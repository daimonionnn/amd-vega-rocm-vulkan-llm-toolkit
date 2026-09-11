# torch 2.11.0+rocm7.2 on gfx900 — where it fails

2026-09-11, Vega 8 iGPU **clocked down to 2300 MHz**, 128 gfx900 rocBLAS files
injected from ROCm 6.3.4.

Two earlier runs at 2400 MHz hard-froze the host and lost all output, because
container logs and page cache do not survive a hard lockup. This run logged each
step to disk with `fsync` **before** executing it, so a freeze would still leave
the culprit on disk. [`traced-verify.py`](traced-verify.py), [`trace.log`](trace.log).

## Result

```
BEGIN pointwise chain    END PASS  1.19e-07
BEGIN sum 4M             END PASS  4.88e-04
BEGIN softmax            END PASS  1.49e-08
BEGIN layer_norm         END PASS  7.15e-07
BEGIN sdpa attention     END PASS  3.28e-07
BEGIN conv net forward   END PASS  2.30e-07
BEGIN backward pass      <-- dies here
```

```
Memory access fault by GPU node-1 on address 0x7ab6d7023000. Reason: Unknown.
traps: python3 general protection fault
```

Full error in [`gpu-fault.txt`](gpu-fault.txt). **The host did not freeze and the
GPU was not reset** (0 resets in `dmesg`): the process died cleanly.

## What it means

**The fault is in the software.** A GPU memory access fault on the same operation
is a deterministic bad address from a kernel, not the random corruption or
scattered hangs a marginal clock produces. torch 2.11's autograd backward pass
touches memory it should not on this target. Forward passes — including
attention and a conv net — are fine.

**The clock changed how badly it fails.** At 2300 MHz the fault killed one
process. At 2400 MHz the same workload froze the whole machine twice. Those
earlier runs have no trace, so it is not proven they died at the backward pass —
but it is consistent with it, and with the lower clock letting the driver contain
a fault that the higher one could not.

`torch 2.7.0+rocm6.3` passes the identical backward pass (see
[`2026-09-11-pytorch-correctness.txt`](../2026-09-11-pytorch-correctness.txt)).
The bug is specific to the newer stack.
