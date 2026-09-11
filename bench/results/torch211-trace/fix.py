"""Try fixes for the stride-2 conv backward fault. The fix is chosen by the first
argument; a stride-1 conv backward runs first as a control, so a bad GPU state
shows up as the control failing rather than as a false result for the fix."""
import sys, os, time, torch, torch.nn as nn
fix = sys.argv[1]
if fix == "no_miopen":
    torch.backends.cudnn.enabled = False   # bypass MIOpen, use native ATen conv
f = open("/t/fix.log", "a", buffering=1)
def tr(m):
    s = f"{time.strftime('%H:%M:%S')} [{fix}] {m}"; f.write(s+"\n"); f.flush(); os.fsync(f.fileno()); print(s, flush=True)
dev = torch.device("cuda:0"); torch.manual_seed(0)
def back(stride, cin, cout, H):
    mc = nn.Conv2d(cin, cout, 3, padding=1, stride=stride); mg = nn.Conv2d(cin, cout, 3, padding=1, stride=stride)
    mg.load_state_dict(mc.state_dict()); x = torch.randn(8, cin, H, H)
    xc = x.clone().requires_grad_(True); xg = x.clone().to(dev).requires_grad_(True)
    mc(xc).sum().backward()
    mg.to(dev)(xg).sum().backward(); torch.cuda.synchronize()
    return (xg.grad.cpu()-xc.grad).abs().max().item(), (mg.weight.grad.cpu()-mc.weight.grad).abs().max().item()
tr("BEGIN control stride1"); e = back(1, 3, 32, 32); tr(f"END   control stride1  in {e[0]:.1e} w {e[1]:.1e}")
tr("BEGIN stride2");         e = back(2, 32, 64, 32); tr(f"END   stride2  in {e[0]:.1e} w {e[1]:.1e}  <-- FIXED")
