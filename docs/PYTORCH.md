# PyTorch on a Vega 8 APU (gfx90c)

It works. Verified 2026-09-11 on a Ryzen 7 5700G / Radeon Vega 8 under Ubuntu
26.04, kernel 7.0 — rocBLAS, rocRAND and MIOpen all producing numerically
correct results. No ROCm installation on the host is needed beyond the kernel
driver, and no patching or source build is involved.

This page is for anyone wanting to run PyTorch, ComfyUI or similar on Vega
silicon. It is separate from the llama.cpp material because almost none of that
applies here: the Tensile backport, `patches/0001` and the ROCm version debates
are all irrelevant to PyTorch, for the reason in the next section.

## The recipe

```bash
git clone https://github.com/daimonionnn/amd-vega-rocm-vulkan-llm-toolkit
cd amd-vega-rocm-vulkan-llm-toolkit
docker build -t pytorch-rocm63-vega -f build/Dockerfile.pytorch-rocm63-vega .
./run/run-pytorch-rocm63.sh                # runs the smoke test
./run/run-pytorch-rocm63.sh python3        # interactive
./run/run-pytorch-rocm63.sh bash           # shell
```

That is the whole thing. The image is about 29 GB unpacked; the wheel download
is 4.5 GB.

## Why the system ROCm version does not matter

The PyTorch ROCm wheel **bundles its own complete ROCm** — `libamdhip64`,
rocBLAS, rocRAND, MIOpen, the lot. The container here is `python:3.12-slim`
with no ROCm installed at all, and it works. The host supplies only the amdgpu
kernel driver and `/dev/kfd`.

So "should I use ROCm 7.2 or 7.14 for PyTorch" is not a question with an answer.
Whatever is in `/opt/rocm` is ignored. What matters is which wheel you install.

## Why `torch 2.7.0+rocm6.3` specifically

It is the last stock wheel that ships gfx900 kernels. Surveyed 2026-09-11 by
reading the wheels' contents:

| Wheel | gfx900 files in `torch/lib/rocblas/library/` |
| --- | ---: |
| `torch 2.7.0+rocm6.3` | **128** |
| `torch 2.8.0+rocm6.4` | 0 |
| `torch 2.10.0+rocm7.0` | 0 |
| `torch 2.10.0+rocm7.1` | 0 |
| `torch 2.11.0+rocm7.2` | 0 |

The newer wheels do contain five files matching "gfx900", but they are leftover
MIOpen tuning databases (`gfx900_56.db.txt`), not kernels.

This mirrors the system packages: ROCm 6.3.4 is the last release whose libraries
carry consumer gfx9 device code at all. See
[the package survey](../bench/results/2026-09-10-rocm-gfx900-library-survey.tsv).

### A newer wheel works — for inference as-is, for training with one switch

`torch 2.11.0+rocm7.2` ships no gfx900 kernels, but injecting the 128 gfx900
rocBLAS files from ROCm 6.3.4 makes nearly all of it work: a
[20-op sweep](../bench/results/torch211-trace/README.md) across pooling,
upsampling, normalisation, activations, indexing, sorting, linear algebra and
transposed convolution is correct against the CPU, and so is backward through
non-convolution layers. [`build/Dockerfile.pytorch-rocm72-vega`](../build/Dockerfile.pytorch-rocm72-vega)
does the injection.

**One thing is broken: MIOpen's convolution backward.** For some shapes it makes
the GPU access an invalid address:

```
Memory access fault by GPU node-1 on address 0x7ab6d7023000
```

— and it froze the host twice before being isolated. Which shapes fail depends
on the algorithm MIOpen selects, not on stride: a small stride-2 conv faulted,
and so, it appears, did a larger stride-1 one. Disabling MIOpen's tuning
database does not help.

So:

| Workload | Setting | Cost |
| --- | --- | --- |
| **Inference** — ComfyUI, generation, anything under `torch.no_grad()` | none; leave MIOpen on | none |
| **Training** | `torch.backends.cudnn.enabled = False` | convolutions ~3× slower |

The workaround makes PyTorch use its native convolution instead of MIOpen, and
with it the full verification passes 9/9 including the backward pass. The cost
is real — conv forward goes from 5.75 ms to 15.51 ms at 64→128 channels,
16×64×64, measured at the stock 2000 MHz — which is why it is worth applying only
when you need gradients.

`torch 2.7.0+rocm6.3` has no such fault and needs neither the injection nor the
switch. Prefer it unless you need something from a newer PyTorch.

### What 2.11 actually brings over 2.7 — on this hardware, very little

Checked against the release notes for 2.8, 2.9, 2.10 and 2.11 (April 2025 to
March 2026).

**The headline features mostly target other hardware.** FlexAttention's
FlashAttention-4 backend is Hopper/Blackwell only; FlexAttention and FP8 work is
Intel XPU; the operator expansion is Apple MPS; differentiable collectives are for
multi-GPU training. None of it helps an 8-CU gfx900 — and on this target PyTorch
cannot use flash or memory-efficient attention at all.

**What reaches AMD** is mostly plumbing: builds against ROCm 7.0, 7.1 and 7.2,
`torch.version.rocm` distinct from `torch.version.hip`, improved pointwise-kernel
heuristics on ROCm, and a run of MIOpen fixes — batchnorm no longer changes
output memory format, convolutions no longer reshape unexpectedly, and MIOpen now
backs CTC loss.

**ComfyUI gains nothing version-gated.** Its code checks the torch version in two
places that matter here:

| Gate | Enables | On this hardware |
| --- | --- | --- |
| `>= 2.10` | mxfp8 compute | off anyway — the function returns `False` unless `is_nvidia()` |
| `>= 2.7` | extended fp16 support | already on with 2.7.0 |

**So the real reasons to upgrade are indirect:** a library that starts requiring a
newer PyTorch, and the security fixes that 2.9 onward list in their release notes.
Neither applies today. Stay on 2.7.0 until one does.

**One lead worth following if you do need 2.11 for training.** 2.9 added
`torch.backends.miopen.immediate`, which switches MIOpen to Immediate Mode — a
different algorithm-selection path from the find mode that picks the faulting
backward algorithm. It might avoid the fault while keeping MIOpen, instead of
paying ~3× for `cudnn.enabled = False`. Untested; 2.11 also made MIOpen
channels-last opt-in again (`PYTORCH_MIOPEN_SUGGEST_NHWC=1`), which touches the
same code. Try it the way the fault was isolated here — traced to disk, one step
per process — because the failure mode can be a host freeze.

### After a GPU hang, reboot — a driver reset is not enough

This cost four wasted test runs and is worth knowing before you debug anything
on this APU. When the GPU hangs, the kernel resets it and reports
`device wedged, but recovered through reset`. **It has not recovered.** After
the first hang, *every* subsequent GPU job hung too — including the
known-good torch 2.7.0 on operations it had passed an hour earlier — and
`dmesg` counted five resets. A reboot restored it immediately: the same test,
run as the first GPU task after boot, passed with zero resets.

The likely reason is that an APU's GPU shares firmware state (SMU, power
management) that a driver-level reset does not reinitialise. Whatever the cause,
the practical rule is: **after any `GPU reset` in `dmesg`, reboot before
trusting another result.** Anything measured in between is measuring the
broken state, not the software under test.

## Two traps

**Pass the render and video groups by numeric gid.** This image is Debian slim
and has no `render` group, so `docker run --group-add=render` makes Docker
refuse to start the container. It fails *before any process runs*, so
`docker logs` is empty and the symptom is indistinguishable from PyTorch
hanging on import. This cost most of the debugging time here.
`run/run-pytorch-rocm63.sh` looks the ids up on the host:

```bash
--group-add "$(getent group render | cut -d: -f3)" \
--group-add "$(getent group video  | cut -d: -f3)"
```

**Check that you are actually on the GPU.** On this hardware a silent fall back
to the CPU is the normal failure mode, not an error — this project has been
caught by three separate mechanisms that each produced plausible numbers. In
PyTorch, verify explicitly:

```python
torch.cuda.is_available()                          # must be True
torch.cuda.get_device_properties(0).gcnArchName    # 'gfx900:xnack-'
```

## What was verified

[`build/pytorch-smoke.py`](../build/pytorch-smoke.py) checks **numbers**, not
the absence of an exception — a kernel silently returning zeros passes a "did it
crash" test. Each check compares against a CPU reference or a known
distribution:

```
torch 2.7.0+rocm6.3, HIP 6.3.42131
device: AMD Radeon Graphics  arch=gfx900:xnack-  CUs=8  mem=64.0 GiB
  PASS  rocBLAS  fp32 matmul 1024^3  — max abs err 2.44e-04
  PASS  rocBLAS  fp16 matmul 1024^3  — max abs err 8.11e-02 (fp16 accum)
  PASS  rocRAND  uniform 2^20  — mean=0.5001 std=0.2884
  PASS  MIOpen   conv2d 8x16x64x64  — max abs err 3.43e-05

  fp32 sgemm 2048^3: 12.2 ms/iter = 1.40 TFLOP/s
```

That throughput line was recorded with the iGPU overclocked to 2400 MHz. At the
stock 2000 MHz the same matmul gives about **1.19 TFLOP/s**; correctness is
unaffected by the clock.

### Beyond the libraries: does a real network compute correctly?

[`build/pytorch-verify.py`](../build/pytorch-verify.py) answers the harder
question — 11 checks, each comparing GPU output against a CPU reference
([full output](../bench/results/2026-09-11-pytorch-correctness.txt)):

| Check | max abs error |
| --- | ---: |
| exp/tanh/sigmoid chain | 1.19e-07 |
| sum over 4M elements | 6.10e-04 |
| mean/var | 2.38e-07 |
| argmax parity | **0** (exact) |
| softmax | 1.49e-08 |
| layer_norm | 7.15e-07 |
| `scaled_dot_product_attention` | 3.28e-07 |
| conv net forward, 8 layers | 2.32e-07 |
| **backward pass (autograd)** | 5.22e-07 |
| fp16 matmul | 5.01e-02 |
| **bf16 matmul** | 3.96e-01 |

Two of those are worth calling out. **Autograd works**, so training is possible
in principle and not just inference. And **bf16 works**, which was not assumed —
it is the more useful reduced-precision format for ML work than fp16.

### What does not work: `torch._int_mm`

int8 × int8 → int32 matrix multiply, used by quantized inference paths. On ROCm
it goes through hipBLASLt, and it fails with:

```
rocblaslt error: Cannot read
  torch/lib/hipblaslt/library/TensileLibrary_lazy_gfx900.dat: No such file
RuntimeError: HIPBLAS_STATUS_INVALID_VALUE when calling hipblasLtMatmulAlgoGetHeuristic
```

Structurally the same problem this repo solves for rocBLAS by copying kernel
files out of ROCm 6.3.4 — except no such file exists to copy. TheRock excludes
hipBLASLt for gfx900 and gfx90c entirely, so int8 quantization is out of reach
rather than merely unpackaged. fp16 and bf16 are unaffected.

rocRAND passing matters beyond itself. [Issue #1](https://github.com/daimonionnn/amd-vega-rocm-vulkan-llm-toolkit/issues/1)
reported it blocked on gfx900 under modular ROCm, where its kernels ship in a
`.kpack` container — but the wheel bundles its own, so that packaging never
enters the picture. PyTorch needs rocRAND for GPU-side RNG, so it would have
been a hard blocker.

## What to expect from the hardware

**About 1.19 TFLOP/s fp32 at the stock 2000 MHz** (1.40 overclocked to 2400),
roughly 58 % of this iGPU's theoretical peak at that clock
(8 CU × 64 lanes × 2 flop × 2.0 GHz = 2.05 TFLOP/s). Larger matrices do better —
4096³ reaches 83 %. For scale, a discrete
Radeon Pro V340 die was reported at ~7 TFLOP/s in issue #1, and a modern dGPU is
another order beyond that.

**Memory is unusual and mostly good news.** `torch.cuda` reports **64 GiB**,
because that is the GTT size set by `amdgpu.gttsize=65536`. There is no separate
VRAM to run out of the way a dGPU does — allocations come from system RAM. On
this machine that means about 43 GB actually usable, which is far more than any
8 GB dGPU offers. Model size is rarely the constraint here; speed is.

Two caveats to that:

- The GTT kernel parameters are **mandatory** for large allocations. Without
  `amdgpu.gttsize=65536 ttm.pages_limit=16777216` in GRUB the iGPU gets ~30 GB
  and overflowing it **hard-freezes the machine**, with no error message.
  `setup/bootstrap-host.sh` sets them; check with
  `grep -o 'amdgpu.gttsize=[0-9]*' /proc/cmdline`.
- Bandwidth is shared DDR4, about 50 GB/s against ~450 GB/s for an HBM2 card.
  Anything bandwidth-bound will feel that.

## How fast is it, really — against this machine's own CPU

Measured 2026-09-11 at the **stock 2000 MHz** iGPU clock
([full results](../bench/results/2026-09-11-pytorch-cpu-vs-apu.md), which also
has the 2400 MHz overclocked figures):

| Workload | CPU (8 threads) | APU | Ratio |
| --- | ---: | ---: | ---: |
| sgemm fp32 4096³ | 677 GF | **1694 GF** | 2.5× |
| sgemm fp16 4096³ | 0.4 GF | **1994 GF** | — |
| conv2d 16×64×128×128 | 90.0 ms | **17.9 ms** | 5.0× |
| **attention 4×12×1024×64** | **20.2 ms** | 56.3 ms | **0.36×** |

Overclocking the iGPU to 2400 MHz is worth about 17 % on convolution, 13–15 % on
matmul and only 10 % on attention, which is memory-bound.

**The fp32 gap is only about 2.5×** — this CPU reaches ~700 GFLOP/s, so the iGPU
is a useful speedup rather than a different league. **fp16 on the CPU is
unusable** (no optimised path; use fp32 or bf16 there), while the APU reaches
1994 GF. **Convolution is the APU's best case at 5×**, which is encouraging
for image work.

**Attention is nearly 3× faster on the CPU**, and that one deserves care. On gfx900
PyTorch has only the `MATH` attention backend — both `FLASH` and
`MEM_EFFICIENT` report "No available kernel" — so it materialises the full
`seq × seq` matrix instead of avoiding it. Transformer work here will be
attention-bound, increasingly so with sequence length. This is PyTorch's
limitation, not the hardware's: llama.cpp has its own flash-attention kernels,
and [`patches/0001`](../patches/README.md) makes them work well on this exact
silicon.

## ComfyUI and Stable Diffusion

**Not yet tested here** — this section is what the measurements imply, not
results.

The compute path is proven and the measurements are mildly encouraging.
Convolution is the APU's strongest case — 5× the CPU at stock clock — and
diffusion models spend most of their time there. Memory is not the constraint it is on an 8 GB
card, so SDXL should fit where it otherwise would not.

The caution is attention. SDXL's transformer blocks will hit the `MATH`
fallback described above, and that cost grows with resolution. Expect this to be
usable for experimentation rather than production, and expect the gap between
SD1.5 and SDXL to be wider here than on supported hardware.

If you try it, the image above is a working PyTorch base to build on. Reporting
real numbers back would be welcome.

## What has not been tried

- ComfyUI, Stable Diffusion, vLLM, or any training workload
- fp16/bf16 throughput beyond the smoke test's single matmul
- `torch.compile` / Triton on this target
- Multi-process or multi-GPU anything
- INT8 paths needing hipBLASLt — confirmed unavailable, see above
