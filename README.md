# VEGA-ROCm-VULKAN-LLM-Toolkit for Linux 

Toolkit for experimental RoCm LLM inference on Vega APUs/GPUs (tested only AMD Ryzen 5700G APU) + tools for dual-GPU LLM management (Vega + nVidia dGPU) - llama.cpp (Llama Server) and LM Studio.

## Hardware

| Component | Detail                                                                   |
| --------- | ------------------------------------------------------------------------ |
| CPU/APU   | AMD Ryzen 7 5700G (8C/16T, Zen 3)                                        |
| iGPU      | Radeon Vega 8 — gfx90c (GCN 5, 8 CUs, 512 MB dedicated + UMA shared RAM) |
| dGPU 1    | AMD Radeon RX 9700 AI Pro (RDNA4, dedicated VRAM)                        |
| dGPU 2    | NVIDIA GeForce RTX 5090 (32 GB VRAM)                                     |
| RAM       | 64 GB DDR4 (shared with Vega 8 iGPU via UMA)                             |
| OS        | Ubuntu 25.10 "Questing", kernel 6.17                                     |

> **GPU targeting note:** Scripts in this toolkit explicitly target the **Vega 8 iGPU** (`/dev/dri/renderD129`, auto-detected by PCI ID `0x1638`, `ROCR_VISIBLE_DEVICES=0`). The Radeon 9700 AI Pro is on `renderD128` (PCI ID `0x7551`). The Radeon 9700 AI Pro and RTX 5090 are not used by these scripts unless you explicitly change device selection.

## Performance

### Vega 8 iGPU — Qwen3.5-35B-A3B Q4_K_M (`-ngl 99 -c 8192`)

| Backend                        | Prefill (t/s) | Generation (t/s) | Notes                                                                      |
| ------------------------------ | ------------- | ---------------- | -------------------------------------------------------------------------- |
| **CPU FA ON** (`-ngl 0 -fa 1`) | **57–233**    | 13–16            | **Best prefill overall.** AVX2 SDPA scales ~4× at large context with FA ON |
| CPU FA OFF (`-ngl 0 -fa 0`)    | 56–226        | 12–14            | Similar to FA ON; use `-fa 1` for best CPU results                         |
| Vulkan native (FA OFF default) | 45–50         | 19–20            | **Best generation throughput** — stable across all context sizes           |
| ROCm 6.2.4 — **FA OFF**        | 40–64         | 12–14            | `-fa 0` recommended — FA ON hurts prefill ~33–83% on Vega 8                |
| ROCm 6.2.4 — FA ON             | 35–49         | 11–13            | Default in old config; suboptimal, use `-fa 0`                             |
| ROCm 7.2 — **FA OFF**          | 39–70         | 12–15            | **Best GPU prefill at large context.** `-fa 0` recommended on Vega 8       |
| ROCm 7.2 — FA ON               | 36–53         | 12–15            | Default in script; weaker prefill than FA OFF at ≥ 1K tokens               |
| LM Studio (Vulkan)             | 49–158*       | 18–19            | *Prefill inflated by LM Studio batching — not directly comparable          |

> Full benchmark data in [docs/benchmarks.md](docs/benchmarks.md).

## Quick Start

```bash
# Vulkan on RTX 5090 (default, fastest)
./start-llm.sh

# Vega 8 iGPU via Vulkan
./start-llm.sh --vega

# ROCm GPU via Docker (Vega 8, full GPU offload)
./run/run-docker-rocm.sh /path/to/model.gguf -ngl 99 -c 2048 --no-warmup

# API endpoint: http://127.0.0.1:8080/v1
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}]}'
```

## Scripts

| Script                                                                   | Purpose                                                                                                           |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| [`start-llm.sh`](start-llm.sh)                                           | **Main launcher.** Vulkan on RTX 5090 by default. Memory safeguards, `--vega`/`--cpu`/`--rocm` modes.             |
| [`run-llamaserver-vulkan.sh`](run-llamaserver-vulkan.sh)                 | Direct Vulkan llama-server wrapper with full device selection (`-dev Vulkan0`/`Vulkan1`).                         |
| [`run-docker-rocm.sh`](run-docker-rocm.sh)                               | **Docker ROCm launcher.** Auto-builds `Dockerfile.rocm64` image on first run, passes GPU devices into container.  |
| [`run-llamaserver-rocm.sh`](run-llamaserver-rocm.sh)                     | Native ROCm wrapper — broken on host (HIP 5.7.1/Clang-21 mismatch); use `run-docker-rocm.sh` instead.             |
| [`build-llamacpp-rocm-vega.sh`](build-llamacpp-rocm-vega.sh)             | Build llama.cpp with ROCm/HIP targeting gfx900 (used inside Docker, or for host experiments).                     |
| [`Dockerfile.rocm7-vega`](Dockerfile.rocm7-vega)                         | **Experimental.** ROCm 7.2 image with gfx900 tensile backport from ROCm 6.3.4.                                    |
| [`run-docker-rocm7.sh`](run-docker-rocm7.sh)                             | **Experimental.** Docker launcher for the ROCm 7.2 image. Same device flags as `run-docker-rocm.sh`.              |
| [`build-llamacpp-rocm7-baremetal.sh`](build-llamacpp-rocm7-baremetal.sh) | **Experimental.** Baremetal ROCm 7 build — downloads tensile backport, no Docker required (needs ROCm 7 on host). |
| [`launch-lmstudio-vulkan.sh`](launch-lmstudio-vulkan.sh)                 | Launch LM Studio with Vulkan env for Vega 8. Has `--diagnose` mode.                                               |
| [`test-server-perf.py`](test-server-perf.py)                             | Benchmark llama-server (port 8080) — prefill and decode t/s across 3 context sizes.                               |
| [`test-lmstudio-perf.py`](test-lmstudio-perf.py)                         | Benchmark LM Studio (port 1234) — streaming time-to-first-token and decode t/s.                                   |
| [`bench/run-all-benchmarks.sh`](bench/run-all-benchmarks.sh)             | **Multi-backend benchmark runner** — iterates all enabled backends×models, collects CSV results, prints summary.  |

## ROCm on Vega 8

**Host ROCm (HIP 5.7.1) is broken** — Ubuntu 25.10 ships HIP 5.7.1 paired with Clang-21, a ~2 major version mismatch. Individual GPU kernel tests pass, but inference segfaults at slot initialization regardless of model, flags, or layer count.

**Docker ROCm 6.2.4 works.** Running llama.cpp in a `rocm/dev-ubuntu-24.04:6.2.4` container provides a coherent stack. Full 41/41 layer GPU offload confirmed on Qwen3.5-35B-A3B-Q4_K_M (20 GB model into 64 GB GTT).

```bash
# Start (auto-builds image on first run, ~10 min)
./run/run-docker-rocm.sh /path/to/model.gguf -ngl 99 -c 2048 --no-warmup
# Server: http://127.0.0.1:8080

# Stop
docker stop $(docker ps -q --filter ancestor=llama-server-rocm-vega)
```

Key env vars baked into `build/Dockerfile.rocm64`:

| Variable                   | Value   | Reason                                                       |
| -------------------------- | ------- | ------------------------------------------------------------ |
| `HSA_XNACK`                | `0`     | `1` hard-freezes the entire PC on Vega 8                     |
| `GGML_HIP_UMA`             | `0`     | UMA mode requires XNACK page-fault handling (disabled above) |
| `HSA_OVERRIDE_GFX_VERSION` | `9.0.0` | Treat gfx90c as gfx900                                       |
| `GPU_MAX_ALLOC_PERCENT`    | `100`   | Allow full GTT allocation                                    |

### Docker & `llama.cpp` Runtime Optimizations

The `run/run-docker-rocm.sh` script applies several crucial flags to maximize inference speed for the ROCm container on your APU:

#### Docker Flags
* `--ipc=host`: Essential for ROCm containers. Bypasses standard shared memory limits, allowing the GPU/CPU to exchange data structures continuously without bottlenecks.
* `--security-opt seccomp=unconfined`: Disables Docker's default syscall filtering. When passing raw character devices (`/dev/kfd`, `/dev/dri`), seccomp adds overhead; removing it grants native bare-metal performance.
* `--ulimit memlock=-1`: Allows unlimited locked memory pages. ROCm relies on memory pinning to stream data between system RAM and the GPU cores without CPU pagetable management. Docker's default limit severely bottlenecks ROCm or causes crashes.

#### `llama.cpp` Flags
* `-fa 1` (Flash Attention): Significantly reduces memory bandwidth overhead and allows larger context sizes. Essential for APUs where memory bandwidth is the primary bottleneck.
* `-ngl 99`: Offloads all layers to the GPU.
* `-t N` (Recommended to add at runtime): Set to your physical CPU core count (e.g., `-t 4` or `-t 8`). Prevents CPU thrashing and saves thermal/power budget for the Vega iGPU.
* `-nkvo` (`--no-kv-offload`): **Do not use unless necessary!** Forces the Key-Value (KV) cache to stay in standard CPU RAM instead of VRAM. Only use this if your model is so large that adding a context window crashes the GPU with Out-of-Memory (OOM) errors.

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md#docker-rocm-workaround-working-solution) for full details and the FP8 stub patch needed for gfx900.

### ROCm 7.2 on Vega 8 — Experimental

**Status: confirmed working** (2026-05-14, Qwen3.5-35B-A3B-Q4_K_M, full 41/41 layer offload, sustained inference stable). ROCm 7.x officially dropped `gfx900` support, but the technique used by
[garymathews/frigate:440056a-rocm-7.2.0](https://github.com/garymathews/frigate/releases/tag/440056a-rocm-7.2.0)
(originally for Frigate NVR / MIGraphX object detection) can be adapted for llama.cpp:

- ROCm 7 LLVM still compiles `gfx900` device code via `hipcc`.
- `rocBLAS` 7.x ships without `gfx900` tensile GEMM kernels — so large-matrix multiply falls back to a slow reference path or fails entirely.
- **Fix:** copy the prebuilt `gfx900` `.co` kernel files **and `TensileLibrary_lazy_gfx900.dat`** from the ROCm **6.3.4** `rocblas` package into ROCm 7's library directory. rocBLAS probes that directory at runtime and picks them up automatically. The lazy `.dat` index file is essential — without it ROCm 7 crashes on the first GEMM with `Illegal seek for GPU arch: gfx900`.

#### Option A — Docker (recommended)

```bash
# Build the image (one-time, ~20–40 min — downloads ROCm 6.3.4 rocblas inside)
docker build -t llama-rocm7-vega -f build/Dockerfile.rocm7-vega build/

# Run (auto-selects Vega 8 render node)
./run/run-docker-rocm7.sh /path/to/model.gguf -ngl 99 -c 2048
```

#### Option B — Baremetal (requires ROCm 7 installed on host)

```bash
./build/build-llamacpp-rocm7-baremetal.sh
./build/build-llamacpp-rocm7-baremetal.sh --skip-backport   # subsequent runs

export HSA_OVERRIDE_GFX_VERSION=9.0.0 HSA_ENABLE_SDMA=0 HSA_XNACK=0 GGML_HIP_UMA=0
./llm/rocm7-vega/bin/llama-server -m /path/to/model.gguf -ngl 99 --host 0.0.0.0 -p 8080
```

## LM Studio (Vulkan)

[`run/launch-lmstudio-vulkan.sh`](run/launch-lmstudio-vulkan.sh) launches LM Studio with the correct Vulkan environment for Vega 8.

```bash
./run/launch-lmstudio-vulkan.sh              # Launch with Vulkan backend
./run/launch-lmstudio-vulkan.sh --diagnose   # Check GPU/memory, show backend targets
./run/launch-lmstudio-vulkan.sh --dry-run    # Print config without launching
```

> LM Studio's bundled ROCm backend only targets RDNA2+ (gfx1030+). Always select **Vulkan** in *Settings → My GPUs*.

## Model Capacity

### Vega 8 — 64 GB UMA (GTT)

| Model Size | Quantization | VRAM Usage | Notes                              |
| ---------- | ------------ | ---------- | ---------------------------------- |
| 3-4B       | Q4_K_M       | ~2-3 GB    | Full offload                       |
| 7-8B       | Q4_K_M       | ~4-5 GB    | Full offload                       |
| 13B        | Q4_K_M       | ~7-8 GB    | Full offload                       |
| 35B (MoE)  | Q4_K_M       | ~20 GB     | Full offload — tested ✓            |
| 70B        | Q4_K_M       | ~35-40 GB  | Should fit in 64 GB GTT — untested |

> 64 GB GTT requires GRUB params: `amdgpu.gttsize=65536 ttm.pages_limit=16777216` — see [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).


## Documentation

| Doc                                                | Contents                                                            |
| -------------------------------------------------- | ------------------------------------------------------------------- |
| [docs/benchmarks.md](docs/benchmarks.md)           | Full benchmark results — ROCm Docker, Vulkan native, LM Studio, CPU |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common errors, Docker ROCm workaround, diagnostic commands          |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)       | GPU architecture, Vulkan vs ROCm analysis, UMA memory model         |
| [docs/BUILD.md](docs/BUILD.md)                     | Build prerequisites, ROCm build from source, HIP patches            |
| [docs/HIP57-PATCHES.md](docs/HIP57-PATCHES.md)     | Technical details of HIP 5.7 compatibility patches                  |

## Project Structure

```
VEGA-ROCm-VULKAN-LLM-Toolkit/
├── README.md
├── start-llm.sh                       ← Main launcher (Vulkan/RTX 5090 default)
│
├── run/                               ← Launch & run scripts
│   ├── run-docker-rocm.sh             ← Docker ROCm 6.2.4 launcher (working, auto-selects Vega 8)
│   ├── run-docker-rocm7.sh            ← Docker ROCm 7.2 launcher (experimental)
│   ├── run-llamaserver-vulkan.sh      ← Vulkan llama-server wrapper
│   ├── run-llamaserver-rocm.sh        ← Native ROCm wrapper (broken on host, kept for reference)
│   └── launch-lmstudio-vulkan.sh      ← LM Studio launcher (Vulkan)
│
├── build/                             ← Dockerfiles & build scripts
│   ├── Dockerfile.rocm64              ← ROCm 6.2.4 image (working)
│   ├── Dockerfile.rocm7-vega          ← ROCm 7.2 image + gfx900 tensile backport (experimental)
│   ├── build-llamacpp-rocm-vega.sh    ← ROCm 6 build script (runs inside Docker)
│   └── build-llamacpp-rocm7-baremetal.sh ← ROCm 7 baremetal build + tensile backport
│
├── bench/                             ← Benchmarks & performance tests
│   ├── bench-rocm.sh                  ← llama-bench (ROCm build)
│   ├── bench-vulkan.sh                ← llama-bench (Vulkan build)
│   ├── run-all-benchmarks.sh          ← Multi-backend runner (ROCm Docker, Vulkan, CPU; multi-model)
│   ├── test-server-perf.py            ← llama-server benchmark (port 8080)
│   └── test-lmstudio-perf.py          ← LM Studio benchmark (port 1234, streaming)
│
├── utils/                             ← Utility scripts
│   ├── search_ddg.py
│   ├── search_github.py
│   └── search_github3.py
│
├── docs/
│   ├── benchmarks.md                  ← Benchmark results (all backends)
│   ├── BUILD.md                       ← Build prerequisites and instructions
│   ├── HIP57-PATCHES.md               ← HIP 5.7 compatibility patches
│   ├── TROUBLESHOOTING.md             ← Common errors and debug tips
│   └── ARCHITECTURE.md               ← GPU architecture, Vulkan vs ROCm analysis
│
├── llm/                               ← llama.cpp build outputs
│   ├── vulkan/                        ← Vulkan build (production)
│   ├── rocm-vega/                     ← ROCm 6 build
│   ├── rocm7-vega/                    ← ROCm 7 build (experimental, created by build script)
│   ├── rocm64/                        ← ROCm 6.4 build
│   └── build/                         ← llama.cpp source workspace
```

## TODO

- [x] Build llama.cpp with ROCm/HIP for gfx900
- [x] Fix xnack (plain gfx900 = xnack-agnostic)
- [x] Fix COv6 incompatibility (force `-mcode-object-version=5`)
- [x] Isolate host crash to HIP 5.7.1 / Clang-21 version mismatch
- [x] Fix GRUB params for 64 GB GTT (`amdgpu.gttsize=65536 ttm.pages_limit=16777216`)
- [x] **Docker ROCm 6.2.4 — working, full GPU offload confirmed**
- [x] **Build llama.cpp with Vulkan backend**
- [x] **Test Vulkan on Vega 8 (stable)**
- [x] **Create Vulkan launcher scripts**
- [x] Benchmark all backends (ROCm Docker, Vulkan native, CPU, LM Studio)
- [x] Document all findings
- [x] **Benchmark flash attention ON vs OFF for all backends** — FA OFF wins for both ROCm 6 and ROCm 7 on Vega 8; FA ON wins for CPU (AVX2 SDPA); see [benchmarks.md](docs/benchmarks.md)
- [x] **Re-run ROCm 6 + CPU benchmarks with consistent settings** — done 2026-05-14, `-c 8192 --no-warmup`, both FA ON and FA OFF
- [ ] Install official AMD ROCm on host to eliminate Docker dependency
- [x] **Housekeeping: reorganised into build/ run/ bench/ utils/ folders**
- [x] **[EXPERIMENTAL] Test ROCm 7.2 Docker build on Vega 8** (`build/Dockerfile.rocm7-vega` + `run/run-docker-rocm7.sh`) — confirmed working 2026-05-14, 35B full offload
- [ ] **[EXPERIMENTAL] Test baremetal ROCm 7.2 build** (`build/build-llamacpp-rocm7-baremetal.sh`)
- [x] **Compare ROCm 6.x vs ROCm 7.x inference speed on Vega 8** — both benefit from `-fa 0`; ROCm 7 FA OFF (70 t/s) edges out ROCm 6 FA OFF (64 t/s) at 1K/4K context; CPU FA ON wins overall (233 t/s at 4K)
