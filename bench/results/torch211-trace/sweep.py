"""Broad op sweep on torch 2.11 + injected gfx900 rocBLAS, DEFAULT settings
(MIOpen enabled — the inference configuration). One category per process;
every op logged to disk with fsync before it runs, and checked against CPU."""
import sys, os, time, torch, torch.nn.functional as F
cat = sys.argv[1]
f = open("/t/sweep.log", "a", buffering=1)
def tr(m):
    s = f"{time.strftime('%H:%M:%S')} [{cat}] {m}"; f.write(s+"\n"); f.flush(); os.fsync(f.fileno()); print(s, flush=True)
dev = torch.device("cuda:0"); torch.manual_seed(0)
def chk(name, fn, *inp, tol=1e-4):
    tr(f"BEGIN {name}")
    c = fn(*inp); g = fn(*[t.to(dev) for t in inp]); torch.cuda.synchronize()
    err = (g.float().cpu() - c.float()).abs().max().item()
    tr(f"END   {name}  {'PASS' if err <= tol else 'FAIL'}  {err:.2e}")

img = torch.randn(4, 16, 64, 64)
if cat == "pool":
    chk("max_pool2d", lambda x: F.max_pool2d(x, 2), img)
    chk("avg_pool2d", lambda x: F.avg_pool2d(x, 2), img)
    chk("adaptive_avg_pool2d", lambda x: F.adaptive_avg_pool2d(x, 7), img)
elif cat == "upsample":
    chk("interpolate nearest", lambda x: F.interpolate(x, scale_factor=2, mode="nearest"), img)
    chk("interpolate bilinear", lambda x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False), img)
elif cat == "norm":
    chk("group_norm", lambda x: F.group_norm(x, 4), img)
    chk("instance_norm", lambda x: F.instance_norm(x), img)
elif cat == "activation":
    v = torch.randn(4096, 512)
    for n, fn in [("gelu", F.gelu), ("silu", F.silu), ("mish", F.mish), ("leaky_relu", F.leaky_relu)]:
        chk(n, fn, v, tol=1e-5)
elif cat == "index":
    w = torch.randn(1000, 64); idx = torch.randint(0, 1000, (256,))
    tr("BEGIN embedding"); e = (F.embedding(idx.to(dev), w.to(dev)).cpu() - F.embedding(idx, w)).abs().max().item(); tr(f"END   embedding  {'PASS' if e == 0 else 'FAIL'}  {e:.2e}")
    src = torch.randn(64, 128); gi = torch.randint(0, 128, (64, 16))
    tr("BEGIN gather"); e = (torch.gather(src.to(dev), 1, gi.to(dev)).cpu() - torch.gather(src, 1, gi)).abs().max().item(); tr(f"END   gather  {'PASS' if e == 0 else 'FAIL'}  {e:.2e}")
elif cat == "sort":
    v = torch.randn(64, 4096)
    chk("cumsum", lambda x: torch.cumsum(x, 1), v, tol=5e-3)
    chk("sort values", lambda x: torch.sort(x, 1).values, v, tol=0)
    chk("topk values", lambda x: torch.topk(x, 16, 1).values, v, tol=0)
elif cat == "linalg":
    a, b = torch.randn(16, 128, 64), torch.randn(16, 64, 96)
    chk("bmm", torch.bmm, a, b, tol=1e-4)
    chk("einsum", lambda x, y: torch.einsum("bij,bjk->bik", x, y), a, b, tol=1e-4)
elif cat == "loss_backward":
    tr("BEGIN linear+ce backward (rocBLAS, not MIOpen)")
    W = torch.randn(10, 256, requires_grad=True); x = torch.randn(64, 256); y = torch.randint(0, 10, (64,))
    F.cross_entropy(x @ W.T, y).backward(); gc = W.grad.clone()
    Wg = W.detach().to(dev).requires_grad_(True)
    F.cross_entropy(x.to(dev) @ Wg.T, y.to(dev)).backward(); torch.cuda.synchronize()
    e = (Wg.grad.cpu() - gc).abs().max().item(); tr(f"END   linear+ce backward  {'PASS' if e < 1e-5 else 'FAIL'}  {e:.2e}")
elif cat == "conv_transpose":
    chk("conv_transpose2d", lambda x, w: F.conv_transpose2d(x, w, stride=2, padding=1, output_padding=1),
        img, torch.randn(16, 8, 3, 3), tol=1e-3)
