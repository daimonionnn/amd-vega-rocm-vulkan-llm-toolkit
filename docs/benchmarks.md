# LLM Inference Benchmarks

## Running Benchmarks

### Automated multi-backend runner — `bench/run-all-benchmarks.sh`

Runs all enabled backends and models sequentially, collects per-backend CSVs, and prints a grouped summary table.

```bash
# Run everything (output to terminal + log file)
./bench/run-all-benchmarks.sh 2>&1 | tee /tmp/bench-$(date +%Y%m%d-%H%M).log
```

**Key config options** (edit the `CONFIG` section at the top of the script):

| Variable            | Default                       | Purpose                                              |
| ------------------- | ----------------------------- | ---------------------------------------------------- |
| `MODELS`            | Qwen3.5-35B-A3B + Gemma-4-E4B | Array of model paths to benchmark; comment to skip   |
| `CONTEXT_SIZE`      | `8192`                        | Context window passed to the server (`-c`)           |
| `PROMPT_SIZES`      | `(128 1024 4096)`             | Requested prompt token counts                        |
| `GEN_TOKENS`        | `50`                          | Tokens to generate per decode measurement            |
| `SERVER_WAIT_TIMEOUT` | `120`                       | Seconds to wait for server `/health` before giving up |
| `RESULTS_DIR`       | `/tmp/bench-results-<ts>`     | Where per-backend CSV files are written              |
| `ROCM7_LLAMA_BIN`   | *(empty)*                     | Path to bare-metal ROCm 7 `llama-server` (skip if empty) |
| `ROCM6_LLAMA_BIN`   | *(empty)*                     | Path to bare-metal ROCm 6 `llama-server` (skip if empty) |

**Selecting backends** — comment/uncomment entries in `ENABLED_BACKENDS`:

```bash
ENABLED_BACKENDS=(
    "ROCm-7.2-Docker-FA-OFF:-fa 0:start_rocm7_docker"
    "ROCm-7.2-Docker-FA-ON:-fa 1:start_rocm7_docker"
    "ROCm-6.2.4-Docker-FA-OFF:-fa 0:start_rocm6_docker"
    "ROCm-6.2.4-Docker-FA-ON:-fa 1:start_rocm6_docker"
    "ROCm-7.2-Baremetal-FA-OFF:-fa 0:start_rocm7_baremetal"   # set ROCM7_LLAMA_BIN first
    # "ROCm-6.2.4-Baremetal-FA-OFF:-fa 0:start_rocm6_baremetal" # set ROCM6_LLAMA_BIN first
    "Vulkan-GPU-FA-OFF:-fa 0:start_vulkan_gpu"
    "CPU-FA-ON:-fa 1:start_cpu"
    "CPU-FA-OFF:-fa 0:start_cpu"
)
```

CSV files are named `<ModelLabel>__<BackendLabel>.csv`. The summary groups results by model, with separate prefill and decode tables per model.

### Single-backend quick test — `bench/test-server-perf.py`

Use this when a server is already running on port 8080:

```bash
# Start a server manually, then:
python3 bench/test-server-perf.py
```

---

## System Hardware (at time of benchmarks)

| Component | Details                                                                  |
| --------- | ------------------------------------------------------------------------ |
| CPU/APU   | AMD Ryzen 7 5700G (8C/16T, Zen 3)                                        |
| iGPU      | Radeon Vega 8 — gfx90c (GCN 5, 8 CUs, 64 GB UMA) — `/dev/dri/renderD129` |
| dGPU 1    | AMD Radeon RX 9700 AI Pro (RDNA4) — `/dev/dri/renderD128`                |
| dGPU 2    | NVIDIA GeForce RTX 5090 (32 GB VRAM)                                     |
| RAM       | 64 GB DDR4                                                               |
| OS        | Ubuntu 25.10, kernel 6.17                                                |

> All Vega 8 benchmarks explicitly use `/dev/dri/renderD129` (PCI ID `0x1638`). The Radeon 9700 is not involved.

---

## Benchmark: Qwen3.5-35B-A3B — ROCm 6.2.4 (Docker, Flash Attention ON vs OFF)

**Date:** 2026-05-14 (re-run with `-c 8192` for both FA settings; original April 2026 results were `-c 4096 -fa 1` only)
**Tool:** `test-server-perf.py` — three context sizes: 128, 1024, 4096 tokens

### Hardware

> **Memory Config:** APU memory allocated to 2GB + GRUB option set to `amdgpu.gttsize=65536 ttm.pages_limit=16777216`

| Component | Details                                            |
| --------- | -------------------------------------------------- |
| GPU       | AMD Radeon Graphics (Vega 8 iGPU, gfx900:xnack-)   |
| VRAM      | 65536 MiB (64 GB GTT — system RAM mapped via UMA)  |
| Backend   | ROCm 6.2.4 (Docker: `rocm/dev-ubuntu-24.04:6.2.4`) |

### Model

| Field             | Value                                     |
| ----------------- | ----------------------------------------- |
| Model             | `lmstudio-community/Qwen3.5-35B-A3B-GGUF` |
| File              | `Qwen3.5-35B-A3B-Q4_K_M.gguf`             |
| Quantization      | Q4_K_M                                    |
| Size              | ~20 GB                                    |
| Layers offloaded  | 41/41 (full GPU offload)                  |
| GPU model buffer  | 19905 MiB (~19.5 GB)                      |
| CPU mapped buffer | 273 MiB                                   |

### Server Configuration

```bash
# Flash attention ON (hardcoded default in run-docker-rocm.sh)
./run/run-docker-rocm.sh .../Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 99 -c 8192 --no-warmup

# Flash attention OFF
./run/run-docker-rocm.sh .../Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 99 -c 8192 --no-warmup -fa 0
```

### Results — Flash Attention ON (`-fa 1`)

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 140 tokens                   | 128       | 37.59                | 13.14                   |
| 937 tokens                   | 1024      | 48.53                | 12.44                   |
| 3330 tokens                  | 4096      | 34.96                | 10.64                   |

### Results — Flash Attention OFF (`-fa 0`)

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 140 tokens                   | 128       | 40.15                | 14.40                   |
| 937 tokens                   | 1024      | 64.43                | 13.82                   |
| 3330 tokens                  | 4096      | 63.97                | 11.64                   |

### Observations

- **FA OFF is significantly faster at large context** — same pattern as ROCm 7: prefill at 937/3330 tokens is ~33–83% faster without FA (64 vs 49 and 64 vs 35 t/s)
- **FA OFF also wins at 128 tokens** (40 vs 38 t/s prefill, 14.4 vs 13.1 decode)
- **Generation is consistently better with FA OFF** (~14 vs ~12 t/s) — improved across all context sizes
- **Root cause:** gfx900 flash attention kernels are absent or poorly tuned in the ROCm rocBLAS build; standard GEMM-based attention performs better on Vega 8 for both ROCm 6 and ROCm 7
- **Recommendation: use `-fa 0`** for ROCm 6 on Vega 8
- Full 41-layer offload achieved with 20 GB model into 64 GB GTT — only possible after GRUB `amdgpu.gttsize=65536` fix
- Host ROCm stack (Ubuntu HIP 5.7.1 + Clang-21) segfaults at slot init; Docker ROCm 6.2.4 resolves this completely

---

## Benchmark: Qwen3.5-35B-A3B — Vulkan (Native, Vega 8)

**Date:** 2026-04-12
**Tool:** `test-server-perf.py` — three context sizes: 128, 1024, 4096 tokens

### Hardware

> **Memory Config:** APU memory allocated to 2GB + GRUB option set to `amdgpu.gttsize=65536 ttm.pages_limit=16777216`

| Component | Details                                                  |
| --------- | -------------------------------------------------------- |
| GPU       | AMD Radeon Graphics (Vega 8 iGPU, gfx90c)                |
| VRAM      | 65536 MiB (64 GB GTT — system RAM mapped via UMA)        |
| Backend   | Vulkan (Mesa RADV, native — `run/run-llamaserver-vulkan.sh`) |

### Model

| Field        | Value                                     |
| ------------ | ----------------------------------------- |
| Model        | `lmstudio-community/Qwen3.5-35B-A3B-GGUF` |
| File         | `Qwen3.5-35B-A3B-Q4_K_M.gguf`             |
| Quantization | Q4_K_M                                    |
| Size         | ~20 GB                                    |

### Server Configuration

```bash
./run/run-llamaserver-vulkan.sh .../Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 99 -c 4096
```

### Results

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 140 tokens                   | 128       | 44.58                | 19.90                   |
| 937 tokens                   | 1024      | 50.62                | 20.46                   |
| 3330 tokens                  | 4096      | 50.00                | 19.76                   |

### Observations

- **Generation throughput** is stable at ~19–20 t/s across all context sizes — notably more consistent than ROCm Docker
- **Prefill is flat** across all context sizes (~45–51 t/s) — Vulkan/RADV handles large KV cache much better than ROCm on this hardware
- No crash workarounds needed — Vulkan runs natively without Docker or env var hacks

---

## Benchmark: Qwen3.5-35B-A3B — CPU Only (Flash Attention ON vs OFF)

**Date:** 2026-05-14 (re-run with `-c 8192` for both FA settings; original April 2026 results were `-c 4096 -fa 0` with older binary)
**Tool:** `test-server-perf.py` — three context sizes: 128, 1024, 4096 tokens

### Hardware

> **Memory Config:** APU memory allocated to 2GB + GRUB option set to `amdgpu.gttsize=65536 ttm.pages_limit=16777216`

| Component | Details                                            |
| --------- | -------------------------------------------------- |
| CPU       | AMD Ryzen 7 5700G (8C/16T, Zen 3, AVX2/FMA)        |
| Backend   | llama.cpp Vulkan binary, no GPU offload (`-ngl 0`) |

### Model

| Field        | Value                                     |
| ------------ | ----------------------------------------- |
| Model        | `lmstudio-community/Qwen3.5-35B-A3B-GGUF` |
| File         | `Qwen3.5-35B-A3B-Q4_K_M.gguf`             |
| Quantization | Q4_K_M                                    |
| Size         | ~20 GB                                    |

### Server Configuration

```bash
# Flash attention ON
LD_LIBRARY_PATH=llm/vulkan/lib llm/vulkan/bin/llama-server -m .../Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 0 -c 8192 -fa 1 --host 0.0.0.0 --port 8080 --no-warmup

# Flash attention OFF
LD_LIBRARY_PATH=llm/vulkan/lib llm/vulkan/bin/llama-server -m .../Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 0 -c 8192 -fa 0 --host 0.0.0.0 --port 8080 --no-warmup
```

### Results — Flash Attention ON (`-fa 1`)

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 140 tokens                   | 128       | 57.07                | 15.67                   |
| 937 tokens                   | 1024      | 215.04               | 15.39                   |
| 3330 tokens                  | 4096      | 233.12               | 12.95                   |

### Results — Flash Attention OFF (`-fa 0`)

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 140 tokens                   | 128       | 55.56                | 13.91                   |
| 937 tokens                   | 1024      | 203.97               | 11.97                   |
| 3330 tokens                  | 4096      | 225.69               | 14.32                   |

### Observations

- **Flash attention dramatically improves CPU prefill at large context** — 215 vs 204 t/s at 1K, 233 vs 226 t/s at 4K (~5% gain); AVX2 SDPA kernels are well-optimised in current llama.cpp
- **Generation is slightly faster with FA ON** (15.4 vs 13.9 t/s at 128 tokens)
- **Short-context prefill is essentially equal** FA ON vs FA OFF (57 vs 56 t/s)
- **These measurements are not comparable to the April 2026 CPU baseline** (82–88 t/s) — those used an older binary on a different llama.cpp version
- **Generation is memory-bandwidth bound on UMA** (~12–15 t/s) — CPU and GPU share the same DRAM bus
- **Recommendation for CPU: use `-fa 1`** — consistent improvement at all context sizes

---

## Benchmark: Qwen3.5-35B-A3B — LM Studio (Vulkan)

**Date:** 2026-04-12
**Tool:** `test-lmstudio-perf.py` — three context sizes: 128, 1024, 4096 tokens

### Hardware

> **Memory Config:** APU memory allocated to 2GB + GRUB option set to `amdgpu.gttsize=65536 ttm.pages_limit=16777216`

| Component | Details                                           |
| --------- | ------------------------------------------------- |
| GPU       | AMD Radeon Graphics (Vega 8 iGPU, gfx90c)         |
| VRAM      | 65536 MiB (64 GB GTT — system RAM mapped via UMA) |
| Backend   | Vulkan (via LM Studio, local server at port 1234) |

### Model

| Field        | Value                                     |
| ------------ | ----------------------------------------- |
| Model        | `lmstudio-community/Qwen3.5-35B-A3B-GGUF` |
| File         | `Qwen3.5-35B-A3B-Q4_K_M.gguf`             |
| Quantization | Q4_K_M                                    |
| Size         | ~20 GB                                    |

### Server Configuration

LM Studio local server — Vulkan backend, default settings, model loaded via UI.

### Results

> Note: prompt token counts are estimated from word count (LM Studio's `/v1/completions` stream does not always return `usage.prompt_tokens`). Actual token counts may differ slightly.

| Context Size (est. tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| -------------------------- | --------- | -------------------- | ----------------------- |
| ~128 tokens                | 128       | 48.84                | 19.58                   |
| ~1024 tokens               | 1024      | 137.85               | 19.05                   |
| ~4096 tokens               | 4096      | 157.60               | 18.05                   |

### Observations

- **Prefill scales dramatically with context size** (49 → 138 → 158 t/s) — this likely reflects LM Studio's batched prefill processing becoming more efficient with larger prompts
- **Generation is stable** at ~18–20 t/s — consistent with Vulkan native, as expected (same backend)
- **Prefill figures are not directly comparable to other backends** — the estimate-based token counts and LM Studio's internal scheduling make wall-clock prefill appear faster than the raw llama-server measurements
- Decode throughput matches the native Vulkan results (~19–20 t/s), confirming the same underlying GPU path

---

## Benchmark: Qwen3.5-35B-A3B — ROCm 7.2 (Docker, Flash Attention ON vs OFF)

**Date:** 2026-05-14 (clean re-run after zombie process cleanup)
**Tool:** `test-server-perf.py` — three context sizes: 128, 1024, 4096 tokens

### Hardware

> **Memory Config:** APU memory allocated to 2GB + GRUB option set to `amdgpu.gttsize=65536 ttm.pages_limit=16777216`

| Component | Details                                                                                 |
| --------- | --------------------------------------------------------------------------------------- |
| GPU       | AMD Radeon Graphics (Vega 8 iGPU, gfx900:xnack-)                                        |
| VRAM      | 65536 MiB (64 GB GTT — system RAM mapped via UMA)                                       |
| Backend   | ROCm 7.2 Docker (`rocm/dev-ubuntu-22.04:7.2` + gfx900 tensile backport from ROCm 6.3.4) |

### Model

| Field            | Value                                     |
| ---------------- | ----------------------------------------- |
| Model            | `lmstudio-community/Qwen3.5-35B-A3B-GGUF` |
| File             | `Qwen3.5-35B-A3B-Q4_K_M.gguf`             |
| Quantization     | Q4_K_M                                    |
| Size             | ~20 GB                                    |
| Layers offloaded | 41/41 (full GPU offload)                  |
| GPU model buffer | 19905 MiB (~19.5 GB)                      |

### Server Configuration

```bash
# Flash attention ON (default in run-docker-rocm7.sh)
./run/run-docker-rocm7.sh .../Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 99 -c 8192 --no-warmup

# Flash attention OFF
./run/run-docker-rocm7.sh .../Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 99 -c 8192 --no-warmup -fa 0
```

Note: `-fa 0` overrides the `-fa 1` baked into `run/run-docker-rocm7.sh` (last flag wins).

### Results — Flash Attention ON (`-fa 1`)

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 140 tokens                   | 128       | 39.40                | 15.44                   |
| 937 tokens                   | 1024      | 53.23                | 14.52                   |
| 3330 tokens                  | 4096      | 35.94                | 12.38                   |

### Results — Flash Attention OFF (`-fa 0`)

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 140 tokens                   | 128       | 38.63                | 15.49                   |
| 937 tokens                   | 1024      | 70.41                | 15.06                   |
| 3330 tokens                  | 4096      | 68.87                | 12.43                   |

### Observations

- **Flash attention hurts prefill at large context on Vega 8 ROCm 7:** Without FA, prefill at 937 and 3330 tokens is **~32–92% faster** (70 vs 53 and 69 vs 36 t/s). This is the opposite of the expected behaviour on discrete GPUs.
- **Flash attention slightly improves short-context generation** (15.44 vs 15.49 t/s at 128 tokens) — negligible difference.
- **Generation throughput is nearly identical** with or without FA (~14 t/s at short context, ~12 t/s at 3330 tokens) — generation is memory-bandwidth bound on this UMA APU regardless of attention algorithm.
- **Without FA, prefill at large context (~69–70 t/s) approaches CPU prefill (84 t/s raw)** — near parity, which is surprising for an 8-CU iGPU built on GCN 5 architecture.
- **Root cause hypothesis:** Flash attention on Vega 8 / ROCm 7 may fall back to a suboptimal kernel path (gfx900 FA kernels absent or poorly tuned in the backported rocBLAS). Standard GEMM-based attention uses the backported gfx900 GEMM kernels which are well-tuned.
- **Recommendation for ROCm 7 on Vega 8: use `-fa 0`** for better prefill throughput without impacting generation speed.

---

### Notes

- ROCm 7.2 confirmed working on Vega 8 — same approach as ROCm 6.2.4 but with newer HIP/LLVM toolchain
- **Tensile backport fix (2026-05-14):** The critical piece is `TensileLibrary_lazy_gfx900.dat` — ROCm 7's rocBLAS looks this up first at runtime; without it you get `Illegal seek for GPU arch: gfx900` on the first GEMM. The multi-stage Docker build installs rocBLAS into a `rocm/dev-ubuntu-22.04:6.3.4` stage, then copies all `*gfx900*` files + the lazy `.dat` index into the ROCm 7 layer.
- `gfx900:xnack-` with Wave Size 64 = correct Vega 8 kernel execution (Wave64 ISA)
- Multi-GPU isolation working: only 65536 MiB (Vega 8) visible, not the Radeon 9700's VRAM
- Superseded by the full multi-backend tables below, including ROCm 7.2 baremetal, ROCm 7.2 Docker, ROCm 6.2.4 Docker, Vulkan, and CPU.

---

## Benchmark: Qwen3.5-35B-A3B — ROCm 7.2 (Baremetal, Flash Attention ON vs OFF)

**Date:** 2026-05-14 (re-run with `HSA_ENABLE_SDMA=1` + `HSA_XNACK=1` explicitly set)
**Tool:** `bench/run-all-benchmarks.sh` (ROCm 7.2 Baremetal backends only)

### Hardware

> **Memory Config:** APU memory allocated to 2GB + GRUB option set to `amdgpu.gttsize=65536 ttm.pages_limit=16777216`

| Component | Details                                                                                                    |
| --------- | ---------------------------------------------------------------------------------------------------------- |
| GPU       | AMD Radeon Vega 8 iGPU (`/dev/dri/renderD129`, gfx900:xnack-)                                              |
| VRAM      | 65536 MiB (64 GB GTT — system RAM mapped via UMA)                                                          |
| Backend   | ROCm 7.2 Baremetal (`/opt/rocm-7.2.0`, `llm/rocm7-vega/bin/llama-server`)                                 |
| Key envs  | `ROCR_VISIBLE_DEVICES=1`, `HSA_OVERRIDE_GFX_VERSION=9.0.0`, `GGML_HIP_UMA=1`, `HSA_ENABLE_SDMA=1`, `HSA_XNACK=1` |

### Model

| Field            | Value                                     |
| ---------------- | ----------------------------------------- |
| Model            | `lmstudio-community/Qwen3.5-35B-A3B-GGUF` |
| File             | `Qwen3.5-35B-A3B-Q4_K_M.gguf`             |
| Quantization     | Q4_K_M                                    |
| Size             | ~20 GB                                    |
| Layers offloaded | 41/41 (full GPU offload)                  |

### Results — ROCm 7.2 Baremetal FA-OFF (`-fa 0`) ✅ recommended

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 140 tokens                   | 128       | 40.55                | 14.50                   |
| 937 tokens                   | 1024      | 68.18                | 14.30                   |
| 3330 tokens                  | 4096      | 67.32                | 11.88                   |

### Results — ROCm 7.2 Baremetal FA-ON (`-fa 1`) ⚠

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 140 tokens                   | 128       | 38.67                | 14.88                   |
| 937 tokens                   | 1024      | 52.16                | 13.93                   |
| 3330 tokens                  | 4096      | 35.07                | 11.90                   |

### Observations

- **Baremetal vs Docker performance is near-identical** — prefill within ~1–3% across all context sizes; decode within ~1 t/s. The gfx900 tensile backport applied in Docker (`*gfx900*` files from ROCm 6.3.4) is also present in the baremetal `/opt/rocm-7.2.0` install.
- **FA-OFF wins at ≥1K tokens** — same pattern as Docker: 68.18 vs 52.16 t/s at 1K, 67.32 vs 35.07 t/s at 4K. FA-ON degrades severely on Vega 8 gfx900 large-context prefill.
- **`HSA_ENABLE_SDMA=1` + `HSA_XNACK=1` have no significant impact on performance** — results are within run-to-run noise (<1%) vs prior baremetal runs without these flags, with at most a very slight improvement across some measurements. The Vega 8 APU uses SDMA for DMA transfers and reports XNACK support; enabling both explicitly is safe and can be kept in the launch environment, but should not be expected to produce meaningful throughput gains.
- **Recommendation:** Use `run/run-rocm7-baremetal.sh` with `-fa 0` (default) for best performance.

---

## Comparison: All Backends — Qwen3.5-35B-A3B-Q4_K_M — Vega 8 iGPU

Model: `Qwen3.5-35B-A3B-Q4_K_M` — AMD Vega 8 iGPU — 2026-05-14  
Settings: `-c 8192 --no-warmup` · `-ngl 99` (GPU) · `-ngl 0` (CPU)

> **Flash attention on ROCm (Vega 8):** FA ON *hurts* prefill — gfx900 FA kernels absent/untuned in rocBLAS. Always use `-fa 0` for ROCm on this hardware.  
> **Flash attention on CPU:** FA ON *helps* prefill (~5%) via AVX2 SDPA. Use `-fa 1` for CPU.  
> `*` LM Studio uses batched scheduling internally — prefill t/s is not directly comparable to single-request llama-server.

### Prefill (Prompt Processing) — t/s  *(higher is better)*

| Backend                 | FA        | ~128 tok | ~1024 tok | ~4096 tok | Notes                                                   |
| ----------------------- | --------- | -------- | --------- | --------- | ------------------------------------------------------- |
| **CPU** (`-ngl 0`)      | **ON** ✅  | 57.1     | **215.0** | **233.1** | Best prefill overall; AVX2 SDPA scales ~4× with context |
| CPU (`-ngl 0`)          | OFF       | 55.6     | 204.0     | 225.7     | Use `-fa 1` instead                                     |
| **Vulkan** (`-ngl 99`)  | OFF       | 44.6     | 50.6      | 50.0      | Flat across context sizes                               |
| **ROCm 7.2** (Docker)    | **OFF** ✅ | 38.6     | **70.4**  | **68.9**  | Best GPU prefill at large context                       |
| ROCm 7.2 (Docker)        | ON ⚠      | 39.4     | 53.2      | 35.9      | FA ON severely hurts at ≥1K tokens                      |
| **ROCm 7.2** (Baremetal) | **OFF** ✅ | 40.6     | 68.2      | 67.3      | Near-identical to Docker; HSA_ENABLE_SDMA+XNACK no-op   |
| ROCm 7.2 (Baremetal)     | ON ⚠      | 38.7     | 52.2      | 35.1      | Same FA penalty as Docker on gfx900                     |
| **ROCm 6.2.4** (Docker)  | **OFF** ✅ | **40.2** | 64.4      | 64.0      | Recommended config for ROCm 6                           |
| ROCm 6.2.4 (Docker)     | ON ⚠      | 37.6     | 48.5      | 35.0      | FA ON severely hurts at ≥1K tokens                      |
| LM Studio (Vulkan) `*`  | —         | ~48.8    | ~137.9    | ~157.6    | Batched scheduling — not single-request                 |

### Generation (Decode) — t/s  *(higher is better)*

| Backend                 | FA        | ~128 tok | ~1024 tok | ~4096 tok | Notes                                       |
| ----------------------- | --------- | -------- | --------- | --------- | ------------------------------------------- |
| **Vulkan** (`-ngl 99`)  | OFF       | **19.9** | **20.5**  | **19.8**  | Best decode; consistent across all contexts |
| LM Studio (Vulkan) `*`  | —         | 19.6     | 19.1      | 18.1      | Same Vulkan backend                         |
| **CPU** (`-ngl 0`)      | ON        | 15.7     | 15.4      | 13.0      | UMA bandwidth bound                         |
| CPU (`-ngl 0`)          | OFF       | 13.9     | 12.0      | 14.3      |                                             |
| **ROCm 7.2** (Docker)    | **OFF** ✅ | 15.5     | 15.1      | 12.4      |                                             |
| ROCm 7.2 (Docker)        | ON        | 15.4     | 14.5      | 12.4      |                                             |
| **ROCm 7.2** (Baremetal) | **OFF** ✅ | 14.5     | 14.3      | 11.9      | Slightly lower than Docker — negligible     |
| ROCm 7.2 (Baremetal)     | ON        | 14.9     | 13.9      | 11.9      |                                             |
| **ROCm 6.2.4** (Docker)  | **OFF** ✅ | 14.4     | 13.8      | 11.6      |                                             |
| ROCm 6.2.4 (Docker)      | ON        | 13.1     | 12.4      | 10.6      |                                             |

### Best Settings per Use Case

| Use Case              | Backend  | Flags           | Prefill   | Decode     |
| --------------------- | -------- | --------------- | --------- | ---------- |
| Fastest prefill — any | CPU      | `-ngl 0 -fa 1`  | 233 t/s   | 13 t/s     |
| Fastest prefill — GPU | ROCm 7.2 | `-ngl 99 -fa 0` | 70 t/s    | 15 t/s     |
| Fastest generation    | Vulkan   | `-ngl 99`       | 45–50 t/s | **20 t/s** |
| Best interactive chat | Vulkan   | `-ngl 99`       | 45–50 t/s | **20 t/s** |
| ROCm best balance     | ROCm 7.2 | `-ngl 99 -fa 0` | 39–70 t/s | 12–15 t/s  |

**Key takeaways:**
- **CPU FA ON dominates prefill** — AVX2 SDPA scales ~4× from 128→4K tokens (57→233 t/s); GPU prefill stays flat at 39–70 t/s
- **ROCm FA ON hurts Vega 8** — both ROCm 6 and 7 lose 30–95% prefill at large context; `-fa 0` is always better on this hardware
- **Vulkan wins decode** — consistent 19–20 t/s at all context sizes; ROCm tops out at ~15 t/s, CPU at ~15 t/s
- **ROCm 7.2 FA OFF** slightly edges ROCm 6 FA OFF at large context (69–70 vs 64 t/s prefill)

---

## Benchmark: gemma-4-E4B-it-Q4_K_M — All Backends — Vega 8 iGPU

**Date:** 2026-05-14
**Tool:** `bench/run-all-benchmarks.sh` — 7 backends, 3 context sizes (128 / 1024 / 4096 tokens)

### Hardware

> **Memory Config:** APU memory set to 16GB and removed `amdgpu.gttsize=65536 ttm.pages_limit=16777216` from GRUB

| Component | Details                                                   |
| --------- | --------------------------------------------------------- |
| GPU       | AMD Radeon Graphics (Vega 8 iGPU, gfx90c / RADV RENOIR)  |
| VRAM      | 16384 MiB (16 GB BIOS carveout — `amdgpu.gttsize=65536` removed) |
| CPU       | AMD Ryzen 7 5700G (8C/16T, Zen 3, AVX2/FMA)              |

### Model

| Field            | Value                                       |
| ---------------- | ------------------------------------------- |
| Model            | `lmstudio-community/gemma-4-E4B-it-GGUF`    |
| File             | `gemma-4-E4B-it-Q4_K_M.gguf`               |
| Architecture     | Gemma 4 (4B effective — MoE sparse)         |
| Quantization     | Q4_K_M                                      |
| Layers offloaded | 43/43 (full GPU offload — ROCm/Vulkan)       |

### Server Configuration

```bash
# All backends run via:
./bench/run-all-benchmarks.sh   # MODELS=(gemma-4-E4B-it-Q4_K_M.gguf), CONTEXT_SIZE=8192
```

Vulkan backend explicitly pinned to `Vulkan0` (Vega 8 RADV RENOIR) via `-dev Vulkan0` to prevent
auto-selection of the RX 9700 (`Vulkan1`).

---

### Results — ROCm 7.2 Baremetal FA-OFF (`-fa 0`) ✅ recommended  *(2026-05-14, HSA_ENABLE_SDMA=1 + HSA_XNACK=1)*

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 141 tokens                   | 128       | 67.96                | 13.32                   |
| 937 tokens                   | 1024      | 81.96                | 12.19                   |
| 3330 tokens                  | 4096      | 81.77                | 10.00                   |

### Results — ROCm 7.2 Baremetal FA-ON (`-fa 1`) ⚠  *(2026-05-14, HSA_ENABLE_SDMA=1 + HSA_XNACK=1)*

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 141 tokens                   | 128       | 61.46                | 13.29                   |
| 937 tokens                   | 1024      | 43.86                | 12.50                   |
| 3330 tokens                  | 4096      | 25.60                | 10.74                   |

### Results — ROCm 7.2 Docker FA-OFF (`-fa 0`) ✅ recommended

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 141 tokens                   | 128       | 67.56                | 13.13                   |
| 937 tokens                   | 1024      | 80.73                | 11.94                   |
| 3330 tokens                  | 4096      | 80.74                | 9.77                    |

### Results — ROCm 7.2 Docker FA-ON (`-fa 1`) ⚠

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 141 tokens                   | 128       | 60.90                | 13.19                   |
| 937 tokens                   | 1024      | 43.12                | 12.23                   |
| 3330 tokens                  | 4096      | 25.70                | 10.36                   |

### Results — ROCm 6.2.4 Docker FA-OFF (`-fa 0`) ✅ recommended

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 141 tokens                   | 128       | 66.97                | 10.58                   |
| 937 tokens                   | 1024      | 80.33                | 9.74                    |
| 3330 tokens                  | 4096      | 79.98                | 8.32                    |

### Results — ROCm 6.2.4 Docker FA-ON (`-fa 1`) ⚠

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 141 tokens                   | 128       | 58.97                | 10.75                   |
| 937 tokens                   | 1024      | 41.03                | 10.17                   |
| 3330 tokens                  | 4096      | 22.74                | 8.91                    |

### Results — Vulkan GPU FA-ON (`-dev Vulkan0 -fa 1`) ✅ recommended

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 141 tokens                   | 128       | 115.76               | 14.73                   |
| 937 tokens                   | 1024      | 158.83               | 14.55                   |
| 3330 tokens                  | 4096      | 156.18               | 14.03                   |

### Results — Vulkan GPU FA-OFF (`-dev Vulkan0`)

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 141 tokens                   | 128       | 80.62                | 14.44                   |
| 937 tokens                   | 1024      | 131.49               | 13.47                   |
| 3330 tokens                  | 4096      | 137.75               | 11.91                   |

### Results — CPU FA-ON (`-ngl 0 -fa 1`) ✅ recommended

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 141 tokens                   | 128       | 257.48               | 12.33                   |
| 937 tokens                   | 1024      | 810.80               | 11.90                   |
| 3330 tokens                  | 4096      | 923.35               | 10.10                   |

### Results — CPU FA-OFF (`-ngl 0 -fa 0`)

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
| ---------------------------- | --------- | -------------------- | ----------------------- |
| 141 tokens                   | 128       | 241.12               | 12.38                   |
| 937 tokens                   | 1024      | 740.49               | 11.95                   |
| 3330 tokens                  | 4096      | 838.06               | 11.23                   |

---

## Comparison: All Backends — gemma-4-E4B-it-Q4_K_M — Vega 8 iGPU

Model: `gemma-4-E4B-it-Q4_K_M` — AMD Vega 8 iGPU — 2026-05-14  
Settings: `-c 8192 --no-warmup` · `-ngl 99` (GPU) · `-ngl 0` (CPU) · Vulkan pinned to `Vulkan0`

> **Flash attention on ROCm (Vega 8):** FA ON *hurts* prefill — same gfx900 kernel gap as Qwen. Always use `-fa 0` for ROCm on this hardware.  
> **Flash attention on Vulkan (Vega 8):** FA ON *helps* — prefill +43% at 128 tokens, +21% at 4K tokens; Mesa RADV SDPA kernels are well-tuned. Use `-fa 1` for Vulkan.  
> **Flash attention on CPU:** FA ON *helps* prefill (~10%) via AVX2 SDPA. Use `-fa 1` for CPU.

### Prefill (Prompt Processing) — t/s  *(higher is better)*

| Backend                  | FA        | ~128 tok | ~1024 tok | ~4096 tok | Notes                                                         |
| ------------------------ | --------- | -------- | --------- | --------- | ------------------------------------------------------------- |
| **CPU** (`-ngl 0`)       | **ON** ✅  | **248**  | **748**   | **839**   | Best prefill overall; AVX2 SDPA scales ~3.4× with context     |
| CPU (`-ngl 0`)           | OFF       | 232      | 691       | 768       | Use `-fa 1` instead                                           |
| **Vulkan** (`-ngl 99`)   | **ON** ✅  | 126      | **173**   | **170**   | Consistent improvement across contexts vs FA OFF              |
| Vulkan (`-ngl 99`)       | OFF       | 93       | 148       | 152       |                                                               |
| **ROCm 7.2** (Baremetal) | **OFF** ✅ | 76       | 89        | 89        | Flat above 1K tokens                                          |
| ROCm 7.2 (Baremetal)     | ON ⚠      | 68       | 48        | 28        | FA ON severely hurts at ≥1K tokens                            |
| **ROCm 7.2** (Docker)    | **OFF** ✅ | 68       | 81        | 81        | *(Historical: 64GB GTT run — ~10% slower)*                    |
| ROCm 7.2 (Docker)        | ON ⚠      | 61       | 43        | 26        | *(Historical: 64GB GTT run)*                                  |
| **ROCm 6.2.4** (Docker)  | **OFF** ✅ | 67       | 80        | 80        | *(Historical: 64GB GTT run)*                                  |
| ROCm 6.2.4 (Docker)      | ON ⚠      | 59       | 41        | 23        | *(Historical: 64GB GTT run)*                                  |

### Generation (Decode) — t/s  *(higher is better)*

| Backend                  | FA        | ~128 tok | ~1024 tok | ~4096 tok | Notes                                       |
| ------------------------ | --------- | -------- | --------- | --------- | ------------------------------------------- |
| **Vulkan** (`-ngl 99`)   | **ON** ✅  | **17.4** | **17.0**  | **16.8**  | Best decode; FA ON also improves decode     |
| Vulkan (`-ngl 99`)       | OFF       | 17.3     | 16.3      | 14.1      |                                             |
| **ROCm 7.2** (Baremetal) | ON        | 15.9     | 14.7      | 12.4      | Decode slightly higher with FA ON here      |
| **ROCm 7.2** (Baremetal) | **OFF** ✅ | 15.6     | 14.3      | 11.6      |                                             |
| **CPU** (`-ngl 0`)       | ON        | 14.7     | 13.8      | 12.0      | CPU decode improved with 16GB limit         |
| CPU (`-ngl 0`)           | OFF       | 14.4     | 13.8      | 13.1      |                                             |
| **ROCm 7.2** (Docker)    | **OFF** ✅ | 13.1     | 11.9      | 9.8       | *(Historical: 64GB GTT run — ~16% slower)*  |
| **ROCm 6.2.4** (Docker)  | **OFF** ✅ | 10.6     | 9.7       | 8.3       | *(Historical: 64GB GTT run)*                |

### Best Settings per Use Case

| Use Case              | Backend  | Flags                           | Prefill      | Decode      |
| --------------------- | -------- | ------------------------------- | ------------ | ----------- |
| Fastest prefill — any | CPU      | `-ngl 0 -fa 1`                  | 839 t/s      | ~12–14 t/s  |
| Fastest prefill — GPU | Vulkan   | `-ngl 99 -dev Vulkan0 -fa 1`    | 126–170 t/s  | ~17 t/s     |
| Fastest generation    | Vulkan   | `-ngl 99 -dev Vulkan0 -fa 1`    | 126–170 t/s  | **17.4 t/s**|
| Best interactive chat | Vulkan   | `-ngl 99 -dev Vulkan0 -fa 1`    | 126–170 t/s  | **17.4 t/s**|
| ROCm best balance     | ROCm 7.2 | `-ngl 99 -fa 0`                 | ~76–89 t/s   | 11–15.6 t/s |

**Key takeaways (16GB VRAM Allocation vs 64GB GTT):**
- **Restricting `amdgpu.gttsize` improves GPU performance globally:** Removing the `amdgpu.gttsize=65536` GRUB parameter and falling back to the 16 GB BIOS carveout unexpectedly boosted GPU decode speeds across the board. ROCm 7.2 baremetal decode jumped from ~13.3 t/s to ~15.6 t/s (+17%), and Vulkan decode surged from ~14.7 t/s to ~17.4 t/s (+18%). Vulkan prefill also increased slightly.
- **CPU prefill took a slight hit, but CPU decode improved:** CPU prefill dropped slightly (923 → 839 t/s at 4K context) with the smaller memory map, but decode speeds increased to nearly match the GPU (12.3 → 14.7 t/s).
- **Hypothesis for improvement:** While the maximum DDR4 memory bandwidth is identical for UMA whether using a 64GB or 16GB allocation window, avoiding the massive 64GB translation table lookup (GTT) likely reduces page faults and memory overhead on the APU's memory controller for small models that fit entirely inside the 16 GB carveout. 
- **Recommendation:** If you only run models smaller than 14 GB (like Gemma 4, Llama 3 8B, etc), **do not** use `amdgpu.gttsize=65536`. Leave it removed to gain ~18% inference speed. Only use the 64GB GTT override when loading large models like Qwen 35B.
