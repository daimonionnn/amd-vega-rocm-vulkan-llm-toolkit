"""Run ONE backward component, named on the command line, to isolate which
kernel faults. Separate processes so one crash does not hide the others."""
import sys, os, time, torch, torch.nn as nn, torch.nn.functional as F
comp = sys.argv[1]
LOG = "/t/isolate.log"
f = open(LOG, "a", buffering=1)
def tr(m):
    s = f"{time.strftime('%H:%M:%S')} [{comp}] {m}"; f.write(s+"\n"); f.flush(); os.fsync(f.fileno()); print(s, flush=True)
dev = torch.device("cuda:0"); torch.manual_seed(0)

def compare(mod_c, mod_g, inp):
    mod_g.load_state_dict(mod_c.state_dict())
    ic = inp.clone().requires_grad_(True)
    ig = inp.clone().to(dev).requires_grad_(True)
    mod_c(ic).sum().backward()
    tr("BEGIN gpu backward")
    mod_g.to(dev)(ig).sum().backward(); torch.cuda.synchronize()
    tr("END   gpu backward")
    e = (ig.grad.cpu() - ic.grad).abs().max().item()
    w = next(mod_c.parameters(), None)
    ew = (next(mod_g.parameters()).grad.cpu() - w.grad).abs().max().item() if w is not None else 0
    tr(f"PASS  input-grad err {e:.2e}  weight-grad err {ew:.2e}")

if comp == "linear":      compare(nn.Linear(256, 128), nn.Linear(256, 128), torch.randn(32, 256))
elif comp == "relu":      compare(nn.ReLU(), nn.ReLU(), torch.randn(32, 256))
elif comp == "batchnorm": compare(nn.BatchNorm2d(32), nn.BatchNorm2d(32), torch.randn(8, 32, 16, 16))
elif comp == "conv":      compare(nn.Conv2d(3, 32, 3, padding=1), nn.Conv2d(3, 32, 3, padding=1), torch.randn(8, 3, 32, 32))
elif comp == "conv_s2":   compare(nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.Conv2d(32, 64, 3, padding=1, stride=2), torch.randn(8, 32, 32, 32))
