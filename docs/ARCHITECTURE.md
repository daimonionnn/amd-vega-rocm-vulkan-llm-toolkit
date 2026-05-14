# Architecture Notes

Technical background on GPU inference for this system.

## System Hardware

| Component | Details |
|-----------|---|
| CPU | AMD Ryzen 7 5700G (8C/16T, Zen 3) |
| iGPU | AMD Radeon Vega 8 (gfx90c, 8 CUs, UMA — 16 GB VRAM / ~23 GB GTT from 64 GB system RAM) — `/dev/dri/renderD129`, PCI ID `0x1638` |
| dGPU 1 | AMD Radeon RX 9700 AI Pro (RDNA4, dedicated VRAM) — `/dev/dri/renderD128`, PCI ID `0x7551` |
| dGPU 2 | NVIDIA GeForce RTX 5090 (32 GB dedicated VRAM) |
| RAM | 64 GB DDR4 (shared with Vega 8 iGPU) |
| OS | Ubuntu 25.10 (Questing), kernel 6.17.0-20-generic |

### Vulkan devices

```
Vulkan0: AMD Radeon Graphics (RADV RENOIR)     — Vega 8 iGPU, ~24 GB shared
Vulkan1: AMD Radeon RX 9700 AI Pro (RADV)      — dedicated RDNA4 dGPU
Vulkan2: NVIDIA GeForce RTX 5090               — 32 GB dedicated VRAM
Vulkan3: llvmpipe (LLVM 21.1.2, 256 bits)      — CPU software rasterizer
```

> Device indices may differ depending on PCIe enumeration order. Use `vulkaninfo --summary` to verify.

## Performance Summary

Llama 2 7B Chat Q4_K_S (3.59 GiB), `-ngl 99 -c 2048`:

| Backend | Device | Prompt (t/s) | Generation (t/s) | Status |
|---------|--------|-------------|-------------------|--------|
| **Vulkan** | **RTX 5090** | **2,117** | **273** | **Production** |
| Vulkan | Vega 8 (Vulkan0) | 49 | 14 | ✅ Works |
| ROCm (CPU-only) | Ryzen 5700G | 55 | 12 | ✅ Works (HIP_VISIBLE_DEVICES=-1) |
| ROCm (GPU, host) | Vega 8 (HIP 5.7.1) | — | — | ❌ CRASHES (HIP 5.7.1/Clang-21 mismatch) |
| ROCm 6.4.4 (Docker) | Vega 8 (gfx90c) | — | — | ❌ CRASHES (kernel amdgpu bug) |
| **ROCm 6.2.4 (Docker)** | **Vega 8 (gfx900)** | **40–64 (FA OFF)** | **12–14** | **✅ Working** |
| **ROCm 7.2 (Docker)** | **Vega 8 (gfx900)** | **39–70 (FA OFF)** | **12–15** | **✅ Confirmed working** (2026-05-14) |
| **ROCm 7.2 (Baremetal)** | **Vega 8 (gfx900)** | **TBD** | **TBD** | **✅ Binary verified** — `gfx900:xnack-`, 65536 MiB; benchmarks pending (2026-05-14) |

Qwen3.5-35B-A3B Q4_K_M (`-ngl 99 -c 2048`):

| Backend | Prefill (t/s) | Generation (t/s) | Notes |
|---------|--------------|------------------|-------|
| **CPU FA ON** (`-ngl 0`) | **57–233** | 13–16 | **Best prefill overall** — AVX2 SDPA scales ~4× at large context |
| CPU FA OFF (`-ngl 0`) | 56–226 | 12–14 | Very similar; use `-fa 1` |
| Vulkan native | 45–50 | **19–20** | **Best generation** — stable across all contexts |
| **ROCm 6.2.4 FA OFF** | **40–64** | **12–14** | `-fa 0` recommended — FA ON hurts 33–83% at large context |
| ROCm 6.2.4 FA ON | 35–49 | 11–13 | Suboptimal on Vega 8 |
| **ROCm 7.2 FA OFF** | **39–70** | **12–15** | Best GPU prefill (70 t/s at 1K–4K) |
| ROCm 7.2 FA ON | 36–53 | 12–15 | Default in script; use `-fa 0` |
> Full details in [benchmarks.md](benchmarks.md).

## GPU Architecture Generations

AMD's GPU architectures relevant to ROCm:

| Generation | Codename | GFX ID | Examples | ROCm Status |
|-----------|----------|--------|----------|-------------|
| GCN 5 | Vega | gfx900, gfx906, **gfx90c** | Vega 56/64, **Vega 8 APU** | Legacy — unofficially supported via gfx900 override + tensile backport |
| CDNA 1 | Arcturus | gfx908 | MI100 | Supported (datacenter) |
| CDNA 2 | Aldebaran | gfx90a | MI200 series | Supported (datacenter) |
| RDNA 2 | Navi 2x | gfx1030, gfx1031 | RX 6600-6950 XT | Supported |
| RDNA 3 | Navi 3x | gfx1100, gfx1101, gfx1102 | RX 7600-7900 XTX | Supported |
| RDNA 3.5 | — | gfx1151 | Strix APU (Ryzen AI) | Supported |
| RDNA 4 | Navi 4x | gfx1200, gfx1201 | **RX 9700 AI Pro (this system)**, RX 9060-9070 XT | Supported |

## Why gfx90c → gfx900?

The Vega 8 iGPU in the Ryzen 5700G reports as **gfx90c**. This is a cut-down variant of the Vega architecture:

- **gfx900** = Vega 10 (discrete Vega 56/64)
- **gfx90c** = Vega APU variant (Renoir, Cezanne)

The "c" suffix indicates an APU variant with:
- Fewer Compute Units (8 CUs vs 64 on Vega 64)
- Unified Memory Architecture (shares system RAM)
- Slightly different memory controller

The ISA (instruction set architecture) is identical between gfx900 and gfx90c. Code compiled for gfx900 runs on gfx90c. This is why `HSA_OVERRIDE_GFX_VERSION=9.0.0` works — it tells the ROCm runtime "treat this as gfx900" and the kernels execute correctly.

## Why LM Studio's ROCm Backend Doesn't Work

LM Studio ships pre-built ROCm backends. Checking their `backend-manifest.json`:

```json
{
  "gpu": {
    "targets": ["gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1151", "gfx1200", "gfx1201"]
  }
}
```

These are all RDNA2+ targets. No GCN 5 (gfx900/gfx90c). When the backend tries to launch a compute kernel, it searches for a code object matching the GPU architecture and finds nothing → `hipErrorInvalidDeviceFunction`.

Even with `HSA_OVERRIDE_GFX_VERSION=9.0.0`, the runtime sees "gfx900" but the binary only has code objects for gfx1030+. The actual ISA is completely different between GCN and RDNA — GCN kernels can't run RDNA instructions and vice versa.

## UMA Memory Model

APUs like the Ryzen 5700G use **Unified Memory Architecture** — the GPU shares system RAM with the CPU. There are two memory pools:

### VRAM (Video RAM / Carve-out)
- Configured in BIOS as "UMA Frame Buffer Size"
- Set to **16 GB** on this system
- This is a portion of system RAM reserved for GPU use
- Appears as "VRAM" in `rocm-smi`
- Fastest access for GPU (direct, no translation needed)

### GTT (Graphics Translation Table)
- Dynamically managed by the kernel
- ~23 GB on this system (depends on available RAM)
- Backed by system RAM with GPU-accessible page table mappings
- Slightly slower than VRAM due to translation overhead
- Appears as "GTT" in `rocm-smi`

### Implications for LLM Inference

- `GGML_HIP_UMA=1` tells llama.cpp this is a UMA system — it can use both VRAM and GTT
- `GPU_MAX_ALLOC_PERCENT=100` prevents the runtime from capping allocation at 75%
- Total usable GPU memory: VRAM + GTT ≈ 16 + 23 = **~39 GB** (theoretical)
- Practical limit depends on system RAM pressure from other processes

## Vulkan vs ROCm on This System

### Why Vulkan wins

ROCm/HIP compute is **fundamentally broken** on kernel 6.17 for the Vega 8 iGPU. This was confirmed with both:
- **Ubuntu ROCm packages** (HIP 5.7.1) — segfaults in `libamdhip64.so`
- **Docker ROCm 6.4.4** (official AMD image) — kernel-level compute ring timeouts

Both paths trigger the same kernel-level failure: `no-retry page fault` → `IB test failed on comp_1.1.0 (-110)` → MODE2 GPU reset → display wedge (since the iGPU drives the monitor). This is an **amdgpu kernel driver bug**, not a userspace issue.

Vulkan bypasses the entire ROCm/HIP/HSA stack and uses:
- **RADV** (Mesa) for the Vega 8 — proven, shipped with distro
- **NVIDIA proprietary driver** (580.126.09) for the RTX 5090

### Backend comparison

| Aspect | Vulkan (RADV/NVIDIA) | ROCm host (HIP 5.7.1) | ROCm 6.4.4 (Docker) | ROCm 6.2.4 (Docker) | ROCm 7.2 (Docker) |
|--------|---------------------|----------------------|----------------------|---------------------|-------------------|
| Driver | Mesa RADV / NVIDIA | ROCm/HSA + HIP | ROCm 6.4.4 + HIP | ROCm 6.2.4 + HIP | ROCm 7.2 + HIP |
| gfx90c support | Native (RADV) | Via gfx900 override | Native | Via gfx900 override | Via gfx900 override + tensile backport |
| RTX 5090 support | **Yes** | No (AMD only) | No | No | No |
| Setup complexity | Zero (OOTB) | Build + patches | Docker (crashes) | `./run/run-docker-rocm.sh` | `./run/run-docker-rocm7.sh` |
| Stability | **Rock solid** | Segfaults in libamdhip64 | Kernel crashes (MODE2 reset) | **✅ Stable** | **✅ Stable** |
| RTX 5090 perf | **2,117 / 273 t/s** | N/A | N/A | N/A | N/A |
| Vega 8 perf (35B) | 45–50 / 19–20 t/s | N/A (crashes) | N/A (crashes) | **40–64 / 12–14 t/s** (FA OFF) | **39–70 / 12–15 t/s** (FA OFF best: 70 t/s prefill at 1K–4K) |
| Vega 8 perf (7B) | 49 / 14 t/s | N/A | N/A | TBD | **confirmed working** |
| Crash risk | None | Hard system locks | Hard system locks | None | None |
| Multi-GPU isolation | Vulkan device index | N/A | N/A | PCI ID auto-detect | PCI ID auto-detect |

### Docker ROCm test results

**ROCm 6.4.4 (`rocm/dev-ubuntu-24.04:6.4.4`) — CRASHES:**
- `rocminfo` inside Docker detected **gfx90c natively**
- llama.cpp built targeting native `gfx90c`
- GPU inference immediately triggered: `no-retry page fault` storm → `IB test failed on comp_1.1.0 (-110)` → MODE2 GPU reset → display wedged
- Two separate test runs both hard-crashed the PC
- This appeared to prove a kernel amdgpu driver bug

**ROCm 6.2.4 (`rocm/dev-ubuntu-24.04:6.2.4`) — WORKS ✅:**
- Targets `gfx900` with `HSA_OVERRIDE_GFX_VERSION=9.0.0` (not native gfx90c)
- Required: `HSA_XNACK=0` (XNACK=1 hard-freezes the PC), `GGML_HIP_UMA=0` (UMA requires XNACK)
- Required: FP8 stub patch in `vendors/hip.h` (gfx900 has no FP8 instructions)
- Full 41/41 layer offload of Qwen3.5-35B-A3B-Q4_K_M (20 GB) into 64 GB GTT
- Output: `ggml_cuda_init: found 1 ROCm devices (Total VRAM: 65536 MiB)`
- Confirmed stable across multiple requests

**Why 6.2.4 works but 6.4.4 crashes:** The 6.4.4 image targeted gfx90c natively, triggering a kernel-level compute ring issue. The 6.2.4 image uses `HSA_OVERRIDE_GFX_VERSION=9.0.0` to present as gfx900 and avoids that code path. The FP8 patch resolves the remaining compile error for gfx900. See `Dockerfile.rocm64` for the full solution.

**ROCm 7.2 (`rocm/dev-ubuntu-22.04:7.2`) — WORKS ✅:**
- Targets `gfx900` with `HSA_OVERRIDE_GFX_VERSION=9.0.0`
- Required: gfx900 tensile backport from ROCm 6.3.4 rocBLAS — includes `TensileLibrary_lazy_gfx900.dat` (the index file ROCm 7 looks up first; missing = `Illegal seek for GPU arch: gfx900` crash on first GEMM)
- Backport installed via multi-stage Docker build: `rocm/dev-ubuntu-22.04:6.3.4` stage with `apt-get install rocblas`, then `*gfx900*` files copied to ROCm 7 layer
- `gfx900:xnack-` with Wave Size 64 — correct Vega 8 (GCN5/Wave64) execution
- Confirmed stable 2026-05-14: Qwen3.5-35B-A3B-Q4_K_M, 41/41 layers, 800+ tokens sustained inference, no crash

**Primary recommendation: Vulkan on RTX 5090** for maximum throughput (RTX 5090 is ~20x faster). **Secondary: Docker ROCm 6.2.4 or 7.2** for ROCm-specific testing on Vega 8.

### Multi-GPU isolation (Vega 8 + Radeon 9700 AI Pro)

With two AMD GPUs on the system, ROCm enumerates both as HSA agents. Without explicit selection, it may pick the Radeon 9700 (renderD128, 32 GB) instead of the Vega 8 (renderD129, 64 GB UMA).

`run-docker-rocm.sh` and `run-docker-rocm7.sh` handle this automatically:
1. Scan `/sys/class/drm/renderD*/device/device` for PCI ID `0x1638` (Vega 8)
2. Pass **only** `/dev/dri/renderD129` into the container — the 9700 is invisible
3. Set `ROCR_VISIBLE_DEVICES=0` + `HIP_VISIBLE_DEVICES=0` as a second layer of protection

If auto-detect fails: `VEGA8_RENDER_NODE=/dev/dri/renderD129 ./run/run-docker-rocm.sh model.gguf`

## SDMA and APU Quirks

**SDMA (System DMA)** is the hardware DMA engine for GPU memory transfers. On APU iGPUs, SDMA can cause:
- System hangs during large transfers
- Corrupted data
- Kernel oops/panics

Setting `HSA_ENABLE_SDMA=0` disables the hardware DMA engine and falls back to shader-based copies. This is slower for large transfers but completely reliable.

## xnack (Page Fault Handling)

**xnack** (eXtended NACK) enables GPU page fault handling — the ability for the GPU to handle missing page table entries by requesting pages from the CPU. On UMA APUs, xnack is **always enabled** because the GPU accesses system RAM through the CPU's page tables.

### Code object xnack tagging

GPU code objects (the compiled kernels embedded in `.so` files) are tagged with their xnack compatibility:

| Build Target | Code Object Tag | Runtime Compatibility |
|---|---|---|
| `gfx900` (plain) | **xnack-agnostic** | Works with both `HSA_XNACK=0` and `HSA_XNACK=1` |
| `gfx900:xnack+` | xnack=on only | Requires `HSA_XNACK=1` |
| `gfx900:xnack-` | xnack=off only | Requires `HSA_XNACK=0` |

### rocBLAS convention: plain target (xnack-agnostic)

AMD's own [rocBLAS CMakeLists.txt](https://github.com/ROCm/rocBLAS/blob/develop/CMakeLists.txt) uses plain `gfx900` (no xnack qualifier). Verified via LLVM IR inspection: plain `gfx900` produces code objects with **no** `+xnack` or `-xnack` feature flags — these are xnack-agnostic and work regardless of the `HSA_XNACK` setting.

This disproves the earlier assumption that plain `gfx900` produces `xnack=off(unsupported)` code objects. The build script now uses:
```bash
AMDGPU_TARGETS="gfx900"    # plain, xnack-agnostic (matches rocBLAS convention)
```

### Earlier xnack-related symptoms (no longer relevant)

With the old `gfx900:xnack+` build:
- `HSA_XNACK=1`: GPU page fault storm (`svm_range_restore_pages`, `amdgpu_irq_handle_ih_soft hogged CPU for >10000us`)
- `HSA_XNACK=0`: "invalid device function" (runtime reports `gfx900:xnack-`, no matching code object)

The plain `gfx900` build eliminates both of these issues. The remaining crash is in the HIP runtime itself (see below).

## Code Object Versions (COv5 vs COv6)

AMD GPU code objects have a version number encoded in the ELF `EI_ABIVERSION` field:

| COv | EI_ABIVERSION | Introduced | Notes |
|-----|---------------|------------|-------|
| COv4 | 2 | ROCm 4.x | Legacy |
| COv5 | 3 | ROCm 5.x | Supported by HIP 5.7 |
| COv6 | 4 | ROCm 6.x | **Not supported by HIP 5.7** |

`clang-21` (Ubuntu 25.10) defaults to COv6. The HIP 5.7.1 runtime's COMGR library can only parse up to COv5. When COv6 code objects are embedded in the shared library, COMGR silently fails to load them and GPU operations crash.

### The fix

Force COv5 at build time:
```cmake
-DCMAKE_HIP_FLAGS="-mcode-object-version=5"
```

## ROCm Runtime Crash Analysis

### The problem

After fixing xnack (plain `gfx900` build) and COv5, any GPU offload (`-ngl 1` or more) **segfaults immediately** during slot initialization. The crash is 100% reproducible:

```
dmesg: llama-server[PID]: segfault at 0 ip ...3320e1... error 4
        in libamdhip64.so.5.7.31921[3320e1,...]
```

The crash is always at the **same offset** (`0x3320e1`) inside `libamdhip64.so.5.7.31921` — a NULL pointer dereference during HIP device setup. This is a **bug in the HIP 5.7 runtime** when paired with the kernel 6.14 amdgpu driver and HSA 6.1.2 runtime.

| Configuration | Result |
|---|---|
| `-ngl 1` with `HSA_XNACK=0`, `-fa off` | Segfault at slot init (libamdhip64.so offset 0x3320e1) |
| `-ngl 1` with `HSA_XNACK=0`, `-fa auto` | Segfault at slot init (same offset) |
| `-ngl 1` with `HSA_XNACK=1` | Page fault storm (GPU 99% busy, no progress) — old build |
| `-ngl 0` (ROCm backend active, no offload) | Segfault on first inference |
| `HIP_VISIBLE_DEVICES=-1` (ROCm disabled) | **Works perfectly** — 55 t/s prompt, 12 t/s generation |

CPU-only mode (hiding the GPU entirely with `HIP_VISIBLE_DEVICES=-1`) works perfectly, confirming the bug is in HIP's device/kernel initialization path.

### Root cause: Ubuntu 25.10 ROCm version mismatch

The Ubuntu 25.10 ROCm packages have severe internal version mismatches:

| Component | Ubuntu Version | Role |
|---|---|---|
| clang / LLVM | **21.1.2** | Compiler (very new) |
| hipcc / comgr / device-libs | **7.0.1** | Compiler support (experimental) |
| HSA runtime (libhsa-runtime64) | **6.1.2** | Low-level GPU runtime |
| HIP runtime (libamdhip64) | **5.7.1** | High-level GPU API (**~2 versions behind**) |

In AMD's official ROCm releases, all these components are version-locked (e.g., all 6.2.x or all 7.0.x). Ubuntu repackaged them independently, creating a Frankenstein stack where:

- The compiler (clang-21) generates code using conventions from ROCm 7.x
- The device libraries (7.0.1) provide intrinsics matching the compiler
- The runtime (5.7.1) expects code from the ROCm 5.x era

Simple kernel dispatches happen to work (the ISA is compatible), but the runtime's internal scheduling, graph management, and memory tracking code paths diverge enough to crash during sustained inference workloads.

### Tested models

Both models crash identically, confirming it's not model-specific:
- **Gemma 4 E2B** (Q4_K_M, 3.18 GiB, 4.65B params) — hard crash at `-ngl 35`
- **Llama 2 7B Chat** (Q4_K_S) — hard crash, then segfault at `-ngl 0`

### Resolution

1. ~~**Increase `amdgpu.gttsize`**~~ — Applied (`amdgpu.gttsize=65536 ttm.pages_limit=16777216`), unlocks 64 GB GTT but doesn't fix the HIP crash.
2. ~~**ROCm 6.4.4 Docker**~~ — Crashed at kernel level (MODE2 GPU reset). Tests confirmed gfx90c native targeting triggers an amdgpu driver bug.
3. **Vulkan (RADV/NVIDIA) — ✅ Production** — Zero setup, rock-solid stability, best throughput on RTX 5090.
4. **ROCm 6.2.4 Docker — ✅ Working** — `run-docker-rocm.sh` / `Dockerfile.rocm64`. Coherent ROCm stack avoids the host HIP 5.7.1 mismatch. Uses gfx900 override + FP8 stub patch. Full GPU offload confirmed. 

### Launcher scripts

| Script | Backend | Usage |
|--------|---------|-------|
| `run/start-llama-server.sh` | ROCm 7.2 baremetal (default) | `./run/start-llama-server.sh` |
| `run/start-llama-server.sh --vulkan` | Vulkan (Vega 8, RADV) | Best decode — 20 t/s gen |
| `run/start-llama-server.sh --cpu` | CPU-only | Best prefill at large context |
| `run/start-llama-server.sh --rocm-docker` | ROCm 7.2 Docker | Same perf as baremetal, containerised |
| `run-llamaserver-vulkan.sh` | Vulkan | Direct launcher with full options |
| `run-docker-rocm.sh` | ROCm 6.2.4 (Docker) | **Working ROCm GPU offload — auto-selects Vega 8** |
| `run-docker-rocm7.sh` | ROCm 7.2 (Docker) | **Confirmed working 2026-05-14 — 35B full offload, sustained inference stable** |
| `run-rocm7-baremetal.sh` | ROCm 7.2 baremetal | Direct wrapper — sets all HSA env vars, auto-detects Vega 8 |
| `run-llamaserver-rocm.sh` | ROCm host | Legacy, CPU-only usable |

## ROCm Software Stack on Ubuntu 25.10

Ubuntu doesn't ship full ROCm packages. The available components:

```
hipcc 7.0.1+dfsg     → HIP compiler (wraps clang-21)
libamdhip64-dev       → HIP runtime library
libhipblas-dev        → hipBLAS (wraps rocBLAS)
librocblas-dev        → rocBLAS (BLAS for ROCm)
librocsolver-dev      → rocSOLVER (LAPACK for ROCm)
rocm-device-libs-21   → GPU intrinsics and device library bitcode
rocminfo              → GPU info tool
rocm-smi              → GPU monitoring tool
```

These are the Ubuntu-repackaged versions, not AMD's official ROCm releases. The HIP version (5.7) lags behind AMD's current ROCm (6.x), which is why patches are needed.

## Further Reading

- [llama.cpp HIP/ROCm documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip)
- [ROCm GPU support matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
- [AMD GPU ISA documentation](https://gpuopen.com/documentation/amd-isa-documentation/)
- [Mesa RADV driver](https://docs.mesa3d.org/drivers/radv.html)
