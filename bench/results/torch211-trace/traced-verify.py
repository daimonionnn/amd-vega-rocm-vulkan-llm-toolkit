#!/usr/bin/env python3
"""pytorch-verify.py's checks, split into steps that are logged to disk BEFORE
they run, with fsync, so a hard host freeze still leaves a record of which
operation was executing. Two earlier freezes on torch 2.11 lost all output
because container logs and page cache do not survive a hard lockup."""
import os, sys, time, torch, torch.nn as nn, torch.nn.functional as F

LOG = "/trace/trace.log"
_f = open(LOG, "a", buffering=1)
def trace(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    _f.write(line + "\n"); _f.flush(); os.fsync(_f.fileno())
    print(line, flush=True)

trace(f"=== torch {torch.__version__} HIP {torch.version.hip} ===")
dev = torch.device("cuda:0")
trace(f"device {torch.cuda.get_device_properties(0).gcnArchName}")
torch.manual_seed(0)
results = []

def step(name, gpu_fn, cpu_fn, tol):
    trace(f"BEGIN {name}")
    g = gpu_fn(); torch.cuda.synchronize()
    err = (g.float().cpu() - cpu_fn().float()).abs().max().item()
    ok = err <= tol
    results.append(ok)
    trace(f"END   {name}  {'PASS' if ok else 'FAIL'}  max abs {err:.2e}")

x = torch.randn(4096, 1024)
step("pointwise chain", lambda: torch.sigmoid(torch.tanh(x.to(dev).exp().clamp(max=20))),
     lambda: torch.sigmoid(torch.tanh(x.exp().clamp(max=20))), 1e-5)
step("sum 4M", lambda: x.to(dev).sum().reshape(1), lambda: x.sum().reshape(1), 5e-2)
step("softmax", lambda: F.softmax(x.to(dev), -1), lambda: F.softmax(x, -1), 1e-6)
ln = nn.LayerNorm(1024); ln_ref = ln(x)
step("layer_norm", lambda: ln.to(dev)(x.to(dev)), lambda: ln_ref, 1e-4)

q, k, v = (torch.randn(2, 8, 512, 64) for _ in range(3))
sdpa_ref = F.scaled_dot_product_attention(q, k, v)
step("sdpa attention", lambda: F.scaled_dot_product_attention(q.to(dev), k.to(dev), v.to(dev)),
     lambda: sdpa_ref, 1e-4)

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv2d(3, 32, 3, padding=1); s.b1 = nn.BatchNorm2d(32)
        s.c2 = nn.Conv2d(32, 64, 3, padding=1, stride=2); s.b2 = nn.BatchNorm2d(64)
        s.fc = nn.Linear(64 * 16 * 16, 10)
    def forward(s, t):
        t = F.relu(s.b1(s.c1(t))); t = F.relu(s.b2(s.c2(t)))
        return s.fc(t.flatten(1))
net = Net().eval(); img = torch.randn(8, 3, 32, 32)
with torch.no_grad():
    net_ref = net(img)
    step("conv net forward", lambda: net.to(dev)(img.to(dev)), lambda: net_ref, 1e-3)

trace("BEGIN backward pass")
nc = Net(); ng = Net(); ng.load_state_dict(nc.state_dict())
img2 = torch.randn(8, 3, 32, 32); tgt = torch.randint(0, 10, (8,))
F.cross_entropy(nc(img2), tgt).backward()
F.cross_entropy(ng.to(dev)(img2.to(dev)), tgt.to(dev)).backward(); torch.cuda.synchronize()
e = (ng.c1.weight.grad.cpu() - nc.c1.weight.grad).abs().max().item()
results.append(e <= 1e-3)
trace(f"END   backward pass  {'PASS' if e <= 1e-3 else 'FAIL'}  max abs {e:.2e}")

a, b = torch.randn(512, 512), torch.randn(512, 512); ref = a @ b
step("fp16 matmul", lambda: a.half().to(dev) @ b.half().to(dev), lambda: ref, 0.5)
step("bf16 matmul", lambda: a.bfloat16().to(dev) @ b.bfloat16().to(dev), lambda: ref, 2.0)

trace(f"=== DONE: {sum(results)}/{len(results)} passed ===")
