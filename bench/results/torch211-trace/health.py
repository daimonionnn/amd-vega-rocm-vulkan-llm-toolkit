import torch
dev = torch.device("cuda:0"); x = torch.randn(4096, 1024)
g = x.to(dev); torch.cuda.synchronize()
for name, fn in [("add", lambda t: t+1.0), ("exp", lambda t: t.exp()), ("tanh", lambda t: t.tanh())]:
    r = fn(g); torch.cuda.synchronize()
    print(f"  OK {name} {(r.cpu()-fn(x)).abs().max().item():.1e}", flush=True)
print("  HEALTH OK", flush=True)
