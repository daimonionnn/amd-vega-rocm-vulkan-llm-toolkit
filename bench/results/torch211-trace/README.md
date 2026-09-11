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


---

## Follow-up the same day: isolated, worked around, and swept

### It is MIOpen's convolution backward — not stride 2, not the tuning database

Each backward component run in its own process ([`isolate.py`](isolate.py),
[`isolate.log`](isolate.log)):

| Backward pass | Result |
| --- | --- |
| ReLU | pass |
| Linear (rocBLAS) | pass |
| BatchNorm (MIOpen) | pass |
| Conv2d stride 1, 3→32, 8×32×32 | pass |
| **Conv2d stride 2, 32→64, 8×32×32** | **fault** |

That first read as "stride 2". It is not. A later cost measurement ran MIOpen's
backward on a **stride-1** conv at a larger size (64→128, 16×64×64) and **froze
the host**, with the iGPU still overclocked to 2300 MHz; the overclock was removed
only after this freeze. The script had no trace, so it is not proven that the
freeze was the backward rather than the forward; but MIOpen forward at that exact
size was re-run afterwards, at stock clock, and works, which leaves the backward. The trigger is **which
algorithm MIOpen selects**, and that depends on shape and size, not stride.

The obvious suspect — the `gfx900_56.db.txt` tuning database shipped in this
wheel, tuned for a 56-CU Vega 56 rather than this 8-CU APU — was tested and is
**not** the cause: disabling it
(`MIOPEN_DEBUG_DISABLE_FIND_DB=1 MIOPEN_FIND_MODE=1`) still faults
([`fix.py`](fix.py), [`fix.log`](fix.log)).

### The workaround, and what it costs

`torch.backends.cudnn.enabled = False` makes PyTorch use its native convolution
instead of MIOpen. With it, the full verification passes **9/9 including the
backward pass**, zero GPU resets ([`trace-no-miopen.log`](trace-no-miopen.log)).

It is not free. Convolution forward, measured without ever running MIOpen's
backward ([`convcost.py`](convcost.py); logs at [2400 MHz](convcost-2400.log) and
[2000 MHz](convcost.log)):

| Conv forward | iGPU clock | MIOpen | native | |
| --- | --- | ---: | ---: | --- |
| 3→32, 16×64×64 | 2400 MHz | 0.52 ms | 1.31 ms | 2.5× slower |
| | 2000 MHz | 0.52 ms | 1.64 ms | 3.2× slower |
| 64→128, 16×64×64 | 2400 MHz | 5.12 ms | 13.79 ms | 2.7× slower |
| | 2000 MHz | 5.75 ms | 15.51 ms | 2.7× slower |

The small MIOpen conv takes 0.52 ms at both clocks — too short to be bound by
compute — so the workaround's relative cost there depends on the clock; on the
larger conv it is 2.7× at either.

MIOpen forward is correct and fast, so the workaround only earns its cost where
the backward pass is needed:

- **Inference** (ComfyUI, generation, anything `torch.no_grad()`): leave MIOpen
  **on**. No workaround needed.
- **Training**: set `torch.backends.cudnn.enabled = False` and accept 2.5–3×
  slower convolutions.

### Nothing else found

20 ops across 9 categories, default settings with MIOpen enabled, each category
in its own process and checked against CPU ([`sweep.py`](sweep.py),
[`sweep.log`](sweep.log)):

| Category | Ops | Result |
| --- | --- | --- |
| pooling | max, avg, adaptive avg | pass, exact |
| upsampling | nearest, bilinear | pass, exact |
| normalisation | group_norm, instance_norm | pass |
| activations | gelu, silu, mish, leaky_relu | pass |
| indexing | embedding, gather | pass, exact |
| sort / scan | cumsum, sort, topk | pass |
| linear algebra | bmm, einsum | pass, exact |
| backward (non-conv) | linear + cross-entropy | pass |
| **transposed conv** | conv_transpose2d | **pass** |

The last row matters: `conv_transpose2d` is implemented with MIOpen's
backward-data kernels, which is why it was run last and expected to fail. It
works — confirming the fault is specific algorithm selections, not MIOpen's
backward paths as a whole. It is also what VAE decoders in diffusion pipelines
use.

### Hardware state, and what that does and does not establish

| Run | iGPU clock | Outcome |
| --- | --- | --- |
| Full verification, twice | 2400 MHz | host froze, no trace |
| Traced verification, isolation, fix tests | 2300 MHz | memory access fault in conv backward, host fine |
| Original cost measurement | 2300 MHz | host froze |
| Safe cost measurement, 20-op sweep, CPU vs APU | **2000 MHz stock** | all pass |
| Safe cost measurement, CPU vs APU ×4 | 2400 MHz, overclock restored | all pass, 0 GPU faults |

The iGPU overclock and CPU Curve Optimizer were removed **after** the third
freeze, and the stock 2000 MHz clock confirmed under load via `pp_dpm_sclk`. The
iGPU was set back to 2400 MHz the same evening, again confirmed under load, and
the safe cost measurement — which includes native convolution backward — passed
there with no GPU faults. The 20-op sweep has not been repeated at 2400 MHz.

**The conv-backward fault has not been re-run since** — not at stock clock and
not after the overclock was restored — because its failure mode is a host freeze. The case
that it is software rests on its character, not on the hardware state: a memory
access fault at the same operation, isolated to one component, with the forward
pass of that same component working at the same size. Overclock instability does
not produce that. An earlier version of this page claimed the third freeze
happened at stock clock and was therefore unambiguously software; that was wrong
about the order of events.
