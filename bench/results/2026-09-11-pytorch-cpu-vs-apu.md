# PyTorch: Vega 8 APU vs the 5700G's own CPU

`torch 2.7.0+rocm6.3`, 8 CPU threads, fp32 unless stated. GFLOP/s for matmul,
wall time for the rest. Re-run with
[`pytorch-cpu-vs-apu.py`](pytorch-cpu-vs-apu.py) — **the iGPU numbers depend on
its clock**, so this page records both configurations it has run at.

| Workload | CPU | APU @ 2400 MHz | APU @ 2000 MHz (stock) | APU Δ | APU÷CPU, 2400 → 2000 |
| --- | ---: | ---: | ---: | ---: | --- |
| sgemm fp32 2048³ | 702 GF | 1402 GF | 1185 GF | −15.4 % | 1.9× → **1.7×** |
| sgemm fp32 4096³ | 677 GF | 1959 GF | 1694 GF | −13.5 % | 2.8× → **2.5×** |
| sgemm fp16 2048³ | 0.4 GF | 2003 GF | 1757 GF | −12.3 % | — |
| sgemm fp16 4096³ | 0.4 GF | 2338 GF | 1994 GF | −14.7 % | — |
| conv2d 16×64×128×128 | 90.0 ms | 14.9 ms | 17.9 ms | **−16.8 %** | 6.4× → **5.0×** |
| **attention 4×12×1024×64** | **20.2 ms** | 50.7 ms | 56.3 ms | **−10.0 %** | 0.40× → **0.36×** |

CPU figures are from the 2000 MHz run; the CPU itself was unaffected by the iGPU
clock, but its boost overclock and Curve Optimizer were disabled at the same
time, which cost it about 3 % on the large matmuls. The 1024³ matmul is omitted:
it runs for a few milliseconds and its CPU figure swung from 504 to 768 GF
between runs, which no clock change explains.

## What the clock change shows

The iGPU clock fell 16.7 % (2400 → 2000 MHz), and not every workload fell with it:

- **Convolution fell 16.8 % — exactly the clock.** It is purely compute-bound.
- **Matmul fell 12–15 %** — mostly compute, a little memory.
- **Attention fell only 10 %.** On this target PyTorch has only the `MATH`
  attention backend (see below), which spends much of its time moving data
  through shared DDR4, and memory bandwidth does not scale with GPU clock.

That is also why these numbers were re-measured rather than scaled: multiplying
everything by 0.833 would have predicted a 17 % attention loss where the real one
is 10 %, and slightly overstated the matmul loss too.

**Efficiency relative to the theoretical peak is unchanged.** Peak fp32 is
8 CU × 64 lanes × 2 flop × clock: 2.46 TFLOP/s at 2400 MHz, 2.05 at 2000. sgemm
2048³ reaches 57 % and 58 % of those respectively; 4096³ reaches 80 % and 83 %.
The GPU does the same work per cycle — it just has fewer cycles.

## Four things worth knowing (at either clock)

**The fp32 gap is only 2–3× at 2400 MHz, 1.7–2.5× at stock.** The CPU reaches
~700 GFLOP/s, which is a respectable AVX2 result on eight cores. Anyone expecting the usual
order-of-magnitude CPU-to-GPU gap will be disappointed: on this hardware the
iGPU is a modest speedup for fp32, not a different league. The APU does reach
1958 GF at 4096³ — about 80 % of its 2.46 TFLOP/s theoretical peak, so it is
working well; the CPU is simply also good.

**fp16 on the CPU is unusable, not merely slow.** 0.4 GFLOP/s is roughly 1700×
slower than the same CPU doing fp32 — PyTorch has no optimised CPU fp16 matmul
and falls back to element-at-a-time conversion. Use fp32 or bf16 on CPU. On the
APU fp16 reaches 2338 GF, above the fp32 peak, consistent with packed fp16
issuing two operations per lane.

**Convolution is where the APU earns its keep** — 6.4× faster at 2400 MHz and
5.0× at stock, the largest honest win here. MIOpen is doing its job, which is good news for anything
image-shaped (ComfyUI, Stable Diffusion, vision models).

**Attention is faster on the CPU**, by 2.5× at 2400 MHz and 2.8× at stock. This is the one result that should
change how you write code for this machine.

## Why attention loses

PyTorch's `scaled_dot_product_attention` picks among several backends. On
gfx900, only the fallback is available:

```
FLASH           unavailable — "Flash attention has been runtime disabled"
MEM_EFFICIENT   unavailable — "No available kernel"
MATH            available
```

The `MATH` backend materialises the full `seq × seq` attention matrix and runs
softmax over it, which is exactly what flash attention exists to avoid. The CPU
path, meanwhile, has an optimised fused kernel. So the comparison is not
CPU-versus-GPU so much as good-algorithm-versus-naive-algorithm, and the
algorithm wins.

**Practical consequences**

- Transformer inference and training on this APU will be attention-bound in a
  way it would not be on supported hardware, and the penalty grows with sequence
  length — the `MATH` backend is quadratic in memory as well as time.
- If a workload is attention-heavy and sequences are long, measure before
  assuming the GPU helps.
- This does not affect llama.cpp, which has its own flash-attention kernels —
  and on this hardware `patches/0001` makes them work properly. The limitation
  is PyTorch's, not the silicon's.
