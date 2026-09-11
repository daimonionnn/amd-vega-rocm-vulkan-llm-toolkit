#!/usr/bin/env python3
"""Broader correctness check for PyTorch on gfx90c.

pytorch-smoke.py answers "do the three libraries work". This answers the harder
question: does a real network compute the same thing on this GPU as on the CPU?
Every check compares GPU output against a CPU reference — a kernel returning
zeros, or a dtype silently demoted, fails here rather than passing quietly.
"""
import sys, torch, torch.nn as nn, torch.nn.functional as F

dev = torch.device("cuda:0")
ok = True
def chk(name, got, ref, tol, note=""):
    global ok
    d = (got.float() - ref.float()).abs()
    err = d.max().item()
    rel = (d / ref.float().abs().clamp(min=1e-6)).max().item()
    p = err <= tol
    ok &= p
    print(f"  {'PASS' if p else 'FAIL'}  {name:34} max abs {err:.2e}  max rel {rel:.2e}" +
          (f"  {note}" if note else ""))

print(f"torch {torch.__version__}  HIP {torch.version.hip}")
p = torch.cuda.get_device_properties(0)
print(f"device {torch.cuda.get_device_name(0)}  arch={p.gcnArchName}  CUs={p.multi_processor_count}\n")
torch.manual_seed(0)

# ── element-wise and reductions ───────────────────────────────────────────────
x = torch.randn(4096, 1024)
chk("exp/tanh/sigmoid chain", torch.sigmoid(torch.tanh(x.to(dev).exp().clamp(max=20))).cpu(),
    torch.sigmoid(torch.tanh(x.exp().clamp(max=20))), 1e-5)
chk("sum over 4M elements", x.to(dev).sum().cpu().reshape(1), x.sum().reshape(1), 5e-2)
chk("mean/var (dim=1)", torch.stack(torch.var_mean(x.to(dev), dim=1)).cpu(),
    torch.stack(torch.var_mean(x, dim=1)), 1e-4)
chk("argmax parity", torch.argmax(x.to(dev), dim=1).float().cpu(),
    torch.argmax(x, dim=1).float(), 0)

# ── softmax and layernorm: the classic silent-wrong-answer ops ────────────────
chk("softmax dim=-1", F.softmax(x.to(dev), dim=-1).cpu(), F.softmax(x, dim=-1), 1e-6)
# Reference first, then move: nn.Module.to() is in-place and returns self, so
# computing the CPU reference after the move silently mixes devices.
ln = nn.LayerNorm(1024)
ln_ref = ln(x)
chk("layer_norm", ln.to(dev)(x.to(dev)).cpu(), ln_ref, 1e-4)

# ── attention, the operation that matters for LLM and diffusion work ──────────
q, k, v = (torch.randn(2, 8, 512, 64) for _ in range(3))
chk("scaled_dot_product_attention",
    F.scaled_dot_product_attention(q.to(dev), k.to(dev), v.to(dev)).cpu(),
    F.scaled_dot_product_attention(q, k, v), 1e-4)

# ── a real multi-layer network, forward and backward ──────────────────────────
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv2d(3, 32, 3, padding=1); s.b1 = nn.BatchNorm2d(32)
        s.c2 = nn.Conv2d(32, 64, 3, padding=1, stride=2); s.b2 = nn.BatchNorm2d(64)
        s.fc = nn.Linear(64 * 16 * 16, 10)
    def forward(s, t):
        t = F.relu(s.b1(s.c1(t))); t = F.relu(s.b2(s.c2(t)))
        return s.fc(t.flatten(1))

net = Net().eval()
img = torch.randn(8, 3, 32, 32)
with torch.no_grad():
    net_ref = net(img)
    chk("conv net forward (8 layers)", net.to(dev)(img.to(dev)).cpu(), net_ref, 1e-3)

net_c = Net(); net_g = Net(); net_g.load_state_dict(net_c.state_dict())
img2 = torch.randn(8, 3, 32, 32); tgt = torch.randint(0, 10, (8,))
F.cross_entropy(net_c(img2), tgt).backward()
F.cross_entropy(net_g.to(dev)(img2.to(dev)), tgt.to(dev)).backward()
gc = net_c.c1.weight.grad; gg = net_g.c1.weight.grad.cpu()
chk("backward pass (conv1 grads)", gg, gc, 1e-3, "autograd")

# ── dtypes ───────────────────────────────────────────────────────────────────
a, b = torch.randn(512, 512), torch.randn(512, 512)
ref = a @ b
chk("fp16 matmul", (a.half().to(dev) @ b.half().to(dev)).cpu(), ref, 0.5, "reduced precision")
try:
    chk("bf16 matmul", (a.bfloat16().to(dev) @ b.bfloat16().to(dev)).cpu(), ref, 2.0, "reduced precision")
except Exception as e:
    print(f"  INFO  bf16 matmul unsupported — {type(e).__name__}")

# ── int8 matmul via torch._int_mm (needs hipBLASLt on ROCm) ───────────────────
try:
    ia = torch.randint(-128, 127, (256, 256), dtype=torch.int8)
    ib = torch.randint(-128, 127, (256, 256), dtype=torch.int8)
    r = torch._int_mm(ia.to(dev), ib.to(dev)).cpu()
    chk("torch._int_mm (int8)", r.float(), (ia.int() @ ib.int()).float(), 0)
except Exception as e:
    print(f"  INFO  torch._int_mm unavailable — {type(e).__name__}: {str(e)[:90]}")

print("\n" + ("ALL CORRECTNESS CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
