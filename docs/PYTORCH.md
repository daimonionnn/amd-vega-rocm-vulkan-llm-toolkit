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

Whether a newer wheel could be made to work by injecting the rocBLAS files is an
open question — the gate is whether PyTorch's own ATen kernels were compiled for
gfx900, which cannot be settled by inspecting the library (see the TODO in the
[README](../README.md) for why that check fails misleadingly).

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

**1.40 TFLOP/s fp32**, roughly 57 % of this iGPU's theoretical peak
(8 CU × 64 lanes × 2 flop × 2.4 GHz = 2.46 TFLOP/s). For scale, a discrete
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

Measured 2026-09-11
([full results](../bench/results/2026-09-11-pytorch-cpu-vs-apu.md)):

| Workload | CPU (8 threads) | APU | Ratio |
| --- | ---: | ---: | ---: |
| sgemm fp32 4096³ | 699 GF | **1959 GF** | 2.8× |
| sgemm fp16 4096³ | 0.4 GF | **2338 GF** | 5839× |
| conv2d 16×64×128×128 | 96.2 ms | **14.9 ms** | 6.4× |
| **attention 4×12×1024×64** | **20.0 ms** | 50.7 ms | **0.40×** |

**The fp32 gap is only about 3×** — this CPU reaches ~700 GFLOP/s, so the iGPU
is a useful speedup rather than a different league. **fp16 on the CPU is
unusable** (no optimised path; use fp32 or bf16 there), while the APU reaches
2338 GF. **Convolution is the APU's best case at 6.4×**, which is encouraging
for image work.

**Attention is 2.5× faster on the CPU**, and that one deserves care. On gfx900
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
Convolution is the APU's strongest case — 6.4× the CPU — and diffusion models
spend most of their time there. Memory is not the constraint it is on an 8 GB
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
