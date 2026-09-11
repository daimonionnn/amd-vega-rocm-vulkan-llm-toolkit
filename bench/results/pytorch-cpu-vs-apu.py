"""PyTorch CPU vs APU on this machine. Forward ops only — safe on torch 2.7.0.
Re-run whenever the iGPU clock changes; the ratios depend on it."""
import time, torch, torch.nn.functional as F
torch.set_num_threads(8)
dev = torch.device("cuda:0")
def bench(fn, warm=3, it=10):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it
rows = []
for n in (1024, 2048, 4096):
    a, b = torch.randn(n, n), torch.randn(n, n); ag, bg = a.to(dev), b.to(dev)
    tc = bench(lambda: a @ b); tg = bench(lambda: ag @ bg); fl = 2 * n**3
    rows.append((f"sgemm fp32 {n}^3", f"{fl/tc/1e9:.1f} GF", f"{fl/tg/1e9:.1f} GF", tc/tg))
for n in (2048, 4096):
    ag, bg = torch.randn(n, n).half().to(dev), torch.randn(n, n).half().to(dev)
    tg = bench(lambda: ag @ bg); fl = 2 * n**3
    rows.append((f"sgemm fp16 {n}^3", "—", f"{fl/tg/1e9:.1f} GF", None))
x, w = torch.randn(16, 64, 128, 128), torch.randn(128, 64, 3, 3); xg, wg = x.to(dev), w.to(dev)
tc = bench(lambda: F.conv2d(x, w, padding=1), 1, 3); tg = bench(lambda: F.conv2d(xg, wg, padding=1))
rows.append(("conv2d 16x64x128x128", f"{tc*1e3:.1f} ms", f"{tg*1e3:.1f} ms", tc/tg))
q, k, v = (torch.randn(4, 12, 1024, 64) for _ in range(3)); qg, kg, vg = q.to(dev), k.to(dev), v.to(dev)
tc = bench(lambda: F.scaled_dot_product_attention(q, k, v), 1, 3); tg = bench(lambda: F.scaled_dot_product_attention(qg, kg, vg))
rows.append(("attention 4x12x1024x64", f"{tc*1e3:.1f} ms", f"{tg*1e3:.1f} ms", tc/tg))
for name, c, g, r in rows:
    print(f"  {name:26} {c:>12} {g:>12} {('%.2fx' % r) if r else '—':>8}", flush=True)
