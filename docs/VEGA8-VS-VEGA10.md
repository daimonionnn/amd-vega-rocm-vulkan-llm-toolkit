# Vega 8 (gfx90c) vs Vega 10 (gfx900) — where "the same chip" stops being true

This whole project rests on one claim: gfx90c and gfx900 share an instruction set,
so `HSA_OVERRIDE_GFX_VERSION=9.0.0` makes gfx900 software run on a Vega 8 APU.
That claim is correct, and it is also narrower than it sounds. The ISA is the same;
almost nothing else is.

This page collects what is actually known to differ, because the differences are
where every surprise in this repo has come from — and because a fix that works on
one will not automatically work on the other. Issue
[#1](https://github.com/daimonionnn/amd-vega-rocm-vulkan-llm-toolkit/issues/1)
confirmed the rocBLAS Tensile backport on a discrete Vega10 (Radeon Pro V340,
MI25-class), which makes the comparison practical rather than theoretical.

## At a glance

| | Vega 8 (this project) | Vega 10 discrete |
| --- | --- | --- |
| Target | **gfx90c** | **gfx900** |
| `gfx_target_version` (KFD) | 90012 | 90000 |
| Silicon | Renoir / Cezanne APU | Vega 10 dGPU |
| Examples | Ryzen 4000G–5000G | RX Vega 56/64, MI25, Radeon Pro V340 |
| Compute units | **8** | 56–64 |
| Memory | UMA, shared DDR4 | dedicated HBM2 |
| Bandwidth | ~50 GB/s | ~410–484 GB/s |
| HSA profile | **`HSA_PROFILE_FULL`** | **`HSA_PROFILE_BASE`** |
| `HSA_OVERRIDE_GFX_VERSION` | **required** | not needed |
| SDMA engines (this host) | 1, 0 XGMI | typically more |

## The ISA really is the same

Code compiled for gfx900 executes correctly on gfx90c. Everything in this repo
depends on it and nothing has contradicted it:

- the rocBLAS gfx900 Tensile kernels backported from ROCm 6.3.4 produce correct
  results here
- `patches/0001` emits `v_mad_mix_f32`, a gfx900 VOP3P instruction, and it works —
  `test-backend-ops -o FLASH_ATTN_EXT` passes 2959/2959, error against an fp64
  reference 1.4e-6
- PyTorch's bundled gfx900 kernels compute correctly (see
  [the smoke test](../bench/results/2026-09-10-pytorch-gfx900-smoke.txt))

The "c" in gfx90c marks an APU variant, not a different instruction set.

## Where it stops

### 1. HSA profile — a different code path inside ROCm

The single most consequential difference found so far, because it changes which
code runs rather than how fast it runs. ROCr's `GpuAgent::InitDma()` guards its
blit setup with:

```cpp
if (use_sdma && (HSA_PROFILE_BASE == profile_)) {
```

A discrete GPU reports `HSA_PROFILE_BASE`. An APU with coherent integrated memory
reports `HSA_PROFILE_FULL` and takes the other branch. So a dGPU and an APU do
**not execute the same code** in the runtime's DMA initialisation.

That looked like the explanation for AMD's ROCm 7.14 wheels segfaulting in
exactly that function on this APU while a V340 runs 7.14-era packages fine. **It
is not.** A second ROCm 7.14 build — `mixa3607/rocm-gfx906:7.14-complete` — runs on
this same APU, through this same `HSA_PROFILE_FULL` branch, without a problem. So
the crash belongs to AMD's wheel build, not to the APU code path. See
[the crash write-up](../bench/results/2026-09-11-rocm714-sdk-gfx900.md) and
[the working 7.14 build](../bench/results/2026-09-11-rocm714-working.md).

**Practical consequence:** "it works on a discrete Vega" is not evidence that it
works on this APU, and vice versa. They diverge inside ROCm itself.

### 2. Memory: the carve-out the APU ignores

A dGPU has VRAM. An APU has a BIOS UMA carve-out plus GTT, and the kernel decides
which to use — `apu_prefer_gtt` rewrites **every** allocation to GTT when
`AMD_IS_APU && real_vram_size < gtt_size`. Measured here: ROCm puts a 20 GB model
entirely in GTT with ~300 MB of VRAM in use, regardless of a 16 GB carve-out.
Vulkan, which does not go through that path, fills the carve-out.

**Practical consequences:**

- Enlarging the BIOS carve-out helps Vulkan and does nothing for ROCm.
- Large models need `amdgpu.gttsize=65536 ttm.pages_limit=16777216` in GRUB. Without
  them the iGPU gets ~30 GB of GTT and a 20 GB model **hard-freezes the machine**
  within seconds. This has no dGPU equivalent.
- Decode is bandwidth-bound at ~50 GB/s rather than ~450, so the two behave
  completely differently on the same model. Do not port performance expectations
  across.

See [ARCHITECTURE.md](ARCHITECTURE.md#uma-memory-model).

### 3. XNACK

`HSA_XNACK=1` **hard-freezes this machine**. The runtime reports the APU as
`gfx900:xnack-`. Not an issue observed on discrete cards, and the reason every
launcher here sets `HSA_XNACK=0`.

### 4. Software support, which is drifting apart

gfx900 was a supported ROCm target for years. gfx90c never was — it is only ever
reached through the override. That gap is widening:

- ROCm **6.3.4** is the last release whose libraries ship consumer gfx9 device
  code at all (rocBLAS, rocRAND, rocFFT, rocSPARSE); 7.x dropped it everywhere.
  See [the package survey](../bench/results/2026-09-10-rocm-gfx900-library-survey.tsv).
- ROCm **7.14.1** lists neither gfx900 nor gfx906 nor gfx90c among supported
  architectures.
- TheRock still *defines* both `gfx900` and `gfx90c` as build targets, with
  identical exclusion lists (hipBLASLt, hipSPARSELt, composable_kernel, rocWMMA,
  hipTensor, rocprofiler-compute), so they can be built even though they are not
  shipped. Only gfx900 is actually published: AMD's nightly index carries
  gfx900, gfx906, gfx908, gfx90a and the RDNA families, while `gfx90c`,
  `gfx90c-igpu` and `igpu-all` return 404 (checked 2026-09-12). For gfx90c a
  ROCm exists as a build target, not as a download.
- `torch 2.7.0+rocm6.3` is the last stock PyTorch wheel carrying gfx900 kernels.
- **But ROCm 7.14 does run here** — mixa3607's TheRock build plus the gfx900
  Tensile files lifted out of AMD's own (unusable) wheel, verified with
  `test-backend-ops` 2959/2959. The support gap is a packaging gap more than a
  capability one.

**Practical consequence:** a discrete Vega10 owner can sometimes use an official
package as-is. On gfx90c there is always a substitution step.

### 5. Scale, which changes what is worth tuning

8 CUs against 56–64 changes which bottleneck you are fighting. Examples measured
here that would not transfer:

- `-ub 256` *hurts* on 8 CUs by under-filling the GEMMs; on a 64-CU part the
  trade-off differs.
- MMQ's 64-column tiles mean an MoE model saturates at `-ub 2048` here while a
  dense model keeps gaining to 4096 — a function of tile geometry and expert
  count, so the crossover moves with CU count.
- Flash attention on this part is register-bound: occupancy 2 caps gfx900 kernels
  at 128 of 256 VGPRs, which is what made `patches/0001` worth 157 % at 32K
  context. The register file is per-CU, so the mechanism carries; the magnitude
  will not.

## If you wanted a native gfx90c ROCm

TheRock defines the target, so this is buildable in principle, and it is the
principled fix for everything above: no `HSA_OVERRIDE_GFX_VERSION=9.0.0`, no
Tensile files copied out of 6.3.4. It has not been attempted here, and as of
2026-09-12 there is no good reason to:

- **Nothing is published.** AMD's nightly index carries gfx900, gfx906, gfx908,
  gfx90a, gfx90X-dcgpu, gfx94X, gfx950 and the RDNA families; `gfx90c`,
  `gfx90c-igpu` and `igpu-all` all return 404. It is a from-source build or
  nothing.
- **It unlocks nothing new.** gfx90c carries the same exclusion list as gfx900 —
  hipBLASLt, hipSPARSELt, composable_kernel, rocWMMA, hipTensor,
  rocprofiler-compute — so INT8 `torch._int_mm` and composable-kernel flash
  attention stay out of reach on either target.
- **Expect no speed.** ROCm 7.14 against 7.2, measured here back to back, was
  noise in 11 of 12 cells and 8 % *worse* in the twelfth. A gfx90c rocBLAS would
  also need Tensile kernels generated for it, untuned for 8 CUs. The gain would
  be architectural — no substitution step, and a road to newer ROCm and
  PyTorch — not throughput.
- **The cost is a night.** A full from-source ROCm build — LLVM, comgr, HIP,
  rocBLAS with kernel generation — on eight cores, plus tens of GB of disk.

If it is ever picked up: build it in a container with
`-DTHEROCK_AMDGPU_FAMILIES=gfx90c-igpu`, make `rocminfo` and `test-backend-ops`
the first milestone rather than benchmarks, and ask
[mixa3607](https://github.com/mixa3607/ML-gfx906) beforehand whether their
pipeline can emit gfx90c — one question instead of a night of compiling.

Reasons to reopen it: AMD starts publishing gfx90c builds, mixa3607 confirms
their pipeline retargets, or something forces this machine past
`torch 2.7.0+rocm6.3` and into the MIOpen conv-backward fault.

## Known unknowns

- Whether `patches/0001` helps as much on a discrete Vega10. This APU is
  memory-bound in a way HBM2 parts are not, so the balance will differ. Raised in
  issue #1; no data yet.
- ~~Whether the ROCm 7.14 `InitDma` crash is the `HSA_PROFILE_FULL` branch or a
  build mismatch.~~ **Settled: the build.** Another 7.14 build runs on this APU
  through the same branch. The corrupt `std::function` manager pointer fits that.
- Whether gfx906 (Radeon VII / MI50 / MI60) behaves like gfx900 for these purposes.
  It is a separate TheRock family with its own exclusions, and
  [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906) treats it separately.
