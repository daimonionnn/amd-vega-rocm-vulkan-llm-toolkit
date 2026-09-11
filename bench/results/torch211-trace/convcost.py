"""Cost of bypassing MIOpen, measured WITHOUT ever running MIOpen's conv backward,
which faults (and once froze the host) on gfx900 in torch 2.11. MIOpen forward
has never failed and is the thing the workaround actually gives up."""
import sys, os, time, torch, torch.nn as nn
mode = sys.argv[1]                      # miopen_fwd | native_fwd | native_fwdbwd
if mode.startswith("native"): torch.backends.cudnn.enabled = False
f = open("/t/convcost.log", "a", buffering=1)
def tr(m):
    s = f"{time.strftime('%H:%M:%S')} [{mode}] {m}"; f.write(s+"\n"); f.flush(); os.fsync(f.fileno()); print(s, flush=True)
dev = torch.device("cuda:0"); torch.manual_seed(0)
def timeit(fn, it=10):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-s)/it*1e3
for cin, cout, H, B in [(3, 32, 64, 16), (64, 128, 64, 16)]:
    m = nn.Conv2d(cin, cout, 3, padding=1).to(dev); x = torch.randn(B, cin, H, H, device=dev)
    tag = f"{cin}->{cout} {B}x{H}x{H}"
    tr(f"BEGIN {tag}")
    if mode.endswith("fwdbwd"):
        xg = x.clone().requires_grad_(True)
        def fb(): xg.grad = None; m.zero_grad(); m(xg).sum().backward()
        ms = timeit(fb)
    else:
        with torch.no_grad(): ms = timeit(lambda: m(x))
    tr(f"END   {tag}  {ms:.2f} ms")
