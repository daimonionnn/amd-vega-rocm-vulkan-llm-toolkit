# DRAFT — not filed

Bug report prepared 2026-09-11, **not submitted**. Review before sending.

**Where to file:** [ROCm/TheRock](https://github.com/ROCm/TheRock/issues) is the
best fit — the binary is a TheRock build, the crash may be a packaging/ABI
problem rather than a source one, and TheRock owns the gfx900 wheels.
[ROCm/rocm-systems](https://github.com/ROCm/rocm-systems/issues) owns the ROCr
source if they redirect. ROCR-Runtime has only 5 open issues and looks dormant
as a standalone repo.

## Tested 2026-09-11: there is no newer build, and gfx900 was dropped

The obvious pre-flight check — re-test on a newer nightly — cannot be done, and
finding that out changes what this report is.

- **gfx900 nightlies stop at `7.14.0a20260612`.** Both `rocm.nightlies.amd.com`
  and the CloudFront origin behind it end there for `rocm-sdk-core` and
  `rocm-sdk-libraries-gfx900`. June 12 is the last build.
- **gfx900 is not in the ROCm 7.14.1 release.** Its notes list gfx908, gfx90a,
  gfx942, gfx950, gfx1030, gfx1100–1103, gfx1150–1153, gfx1200, gfx1201 and
  gfx1250. No gfx900, gfx906 or gfx90c. The sampled release workflow builds only
  `gfx94X-dcgpu`.
- **TheRock can still target them** — `therock_amdgpu_targets.cmake` defines
  both `gfx900` and `gfx90c` with their exclusion lists — so the June wheels were
  presumably CI output for a family that is built but not shipped.

**This is therefore not a regression on a supported target.** It is a crash in a
three-month-old experimental build of an architecture that has since been left
out of the release. Filing it as a defect overstates the case.

**Revised recommendation.** Do not file this as a bug. Either leave it in this
repo's documentation as a dead end others can find, or — if anything is sent at
all — send a short question rather than a defect report: *gfx900 and gfx90c are
defined as TheRock targets and gfx900 wheels were published until 2026-06-12,
but `hsa_init()` segfaults on gfx90c; are these families intended to be usable,
or is the CI output incidental?* That question is answerable in a sentence and
does not ask anyone to debug a dropped architecture. The analysis below stands
if they want it.

---

## Title

`hsa_init()` segfaults in `GpuAgent::InitDma()` on gfx90c APU — corrupt
`std::function` manager pointer (rocm-sdk 7.14.0a20260612, gfx900 wheels)

## Summary

Every ROCm application using the gfx900 SDK wheels dies inside `hsa_init()` on a
Ryzen 5700G / Radeon Vega 8 (gfx90c). `rocminfo` and `hipGetDeviceCount` both
SIGSEGV; 25 of 26 `rocm-sdk test` tests pass, the failing one being the
`rocminfo` invocation.

The same machine runs **classic ROCm 7.2.0 packages without any problem**,
including llama.cpp with full GPU offload, so the hardware, kernel driver and
GTT configuration are known good.

## Environment

| | |
| --- | --- |
| GPU | Radeon Vega 8, gfx90c (Ryzen 7 5700G, Cezanne), PCI `0x1638` |
| OS / kernel | Ubuntu 26.04, kernel 7.0.0-31-generic |
| Failing | `rocm-sdk-core` / `rocm-sdk-libraries-gfx900` / `rocm-sdk-devel` **7.14.0a20260612** from `https://rocm.nightlies.amd.com/v2/gfx900/` |
| Working, same host | classic ROCm **7.2.0**, `libhsa-runtime64.so.1.18.70200` |
| Container | `python:3.12-slim`, `--device=/dev/kfd --device=/dev/dri/renderD128`, render+video gids, `--security-opt seccomp=unconfined` |

## Reproduction

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx900/ "rocm[libraries,devel]"
export LD_LIBRARY_PATH=$SP/_rocm_sdk_core/lib:$SP/_rocm_sdk_libraries_gfx900/lib
$SP/_rocm_sdk_core/bin/rocminfo          # SIGSEGV, exit 139
```

Note: `/usr/local/bin/rocminfo` is a pip console-script wrapper, so gdb reports
"not in executable format". The ELF is at `_rocm_sdk_core/bin/rocminfo`.

## Backtrace

```
Thread 1 "rocminfo" received signal SIGSEGV
#0  0x0000000100000001 in ?? ()
#1  rocr::AMD::GpuAgent::InitDma()        libhsa-runtime64.so.1
#2  rocr::AMD::GpuAgent::PostToolsInit()
#3  rocr::core::Runtime::Load()
#4  rocr::core::Runtime::Acquire()
#5  rocr::HSA::hsa_init()
#6  main
```

## Analysis

Disassembly at the crash site:

```
InitDma+1532:  test %rax,%rax        ; null check
InitDma+1535:  je   +1550            ; taken if null -- it was NOT
InitDma+1537:  mov  %rsp,%rdi        ; dest
InitDma+1540:  mov  %rdi,%rsi        ; source (same object)
InitDma+1543:  mov  $0x3,%edx        ; op = 3
               call *%rax            ; rax = 0x100000001

rax 0x100000001   rdi 0x7fffffffe760   rsi 0x7fffffffe760   rdx 0x3   rbp 0x0
```

This is libstdc++'s
`std::function::_M_manager(_Any_data& dest, const _Any_data& source, _Manager_operation op)`
with `op == __destroy_functor` — so `InitDma` is **destroying a local
`std::function` whose `_M_manager` pointer is `0x100000001`**.

Two observations:

1. The explicit null check immediately before **passed**, so this is not an
   uninitialised-to-zero pointer that a guard would catch.
2. `0x100000001` is `(1 << 32) | 1` — two adjacent 32-bit fields both holding 1,
   read as one 64-bit pointer. That pattern suggests a structure accessed at the
   wrong offset, i.e. an ABI or layout mismatch, rather than ordinary logic
   error.

Comparing `amd_gpu_agent.cpp` between `ROCR-Runtime@rocm-7.2.0` (works here) and
`rocm-systems` (the tree this binary is built from, per the path embedded in the
library), `InitDma` gained wrapper lambdas that create temporary
`std::function` objects which 7.2.0 never creates:

```cpp
queues_[QueueBlitOnly].reset([queue_lambda]() {
  auto queue = queue_lambda();
  queue->SetProfiling(true);
  return queue;
});
```

Those temporaries are the obvious candidate for what is being destroyed here.
`isa_` → `supported_isas()[0]` in the same file is a second candidate if that
vector can be empty for this device.

## Ruled out

**Not an unsupported architecture.** `strings` finds `gfx90c` in the 7.14
runtime. (The working 7.2 runtime contains none of `gfx900`/`gfx90c`/`gfx906`
as strings, deriving names numerically.)

**Not SDMA configuration.** All still SIGSEGV: `HSA_ENABLE_SDMA=0`,
`HSA_ENABLE_SDMA=1`, `HSA_ENABLE_PEER_SDMA=0`, `HSA_ENABLE_SDMA_GANG=0`,
`HSA_ENABLE_SDMA_RECOMMENDED_ENG=0`, `HSA_DISCOVER_COPY_AGENTS=0`. The last
three exist in 7.14 and not in 7.2.

**Not the `HSA_OVERRIDE_GFX_VERSION=9.0.0` this hardware normally needs.**
Identical crash with it set and unset.

**Not missing KFD topology.** Node 1 reports completely:

```
gfx_target_version 90012   simd_count 32     num_xcc 1
num_sdma_engines 1         num_sdma_xgmi_engines 0
num_sdma_queues_per_engine 2                 sdma_fw_version 40
```

## What I could not determine

The actual corruption. No debug symbols are published for these wheels
(`rocm-sdk-core-dbg` and `-debug` both 404), and building ROCr from current
`develop` would not necessarily reproduce a June 2026 binary — particularly if
this is an ABI mismatch, which depends on build configuration rather than
source.

If a `-dbgsym` artifact or the exact `rocm-systems` revision for
`7.14.0a20260612` can be pointed at, I am happy to re-run and report back.

## Why discrete Vega10 works while this APU does not

Reported in
<https://github.com/daimonionnn/amd-vega-rocm-vulkan-llm-toolkit/issues/1>: a
Radeon Pro V340 (discrete Vega10, native gfx900) runs 7.14.1-era packages
successfully. Two differences plausibly account for it, and they compound:

1. **Different code path in this very function.** `InitDma`'s blit setup is
   guarded by `if (use_sdma && (HSA_PROFILE_BASE == profile_))`. A discrete GPU
   reports `HSA_PROFILE_BASE`; an APU with coherent integrated memory reports
   `HSA_PROFILE_FULL` and takes the other path. So the dGPU that works and the
   APU that crashes do not execute the same code here. (Caveat: that branch sits
   inside a lazily-evaluated lambda, so on its own it does not explain a crash
   *during* `InitDma`.)
2. **Different build entirely.** That report used modular `amdrocm*7.14.1`
   **deb packages**; this one uses TheRock **pip wheels** `7.14.0a20260612`.
   Different packaging, different build configuration, and — given the evidence
   points at an ABI or layout mismatch — possibly the more important difference
   of the two.

## Why it may be worth fixing

gfx900 is not a supported target, but these wheels exist and `rocm-sdk targets`
advertises `gfx900`, so they set an expectation. Vega 56/64, MI25 and the
Renoir/Cezanne APUs are numerous and cheap, and ROCm 7.2 demonstrably works on
this silicon — a working 7.14 would let these machines off a two-generation-old
release. Related third-party confirmation that discrete Vega10 works with
7.14-era packages:
<https://github.com/daimonionnn/amd-vega-rocm-vulkan-llm-toolkit/issues/1>
