# PyTorch: Vega 8 APU vs the 5700G's own CPU

2026-09-11, `torch 2.7.0+rocm6.3`, 8 CPU threads, fp32 unless stated.
GFLOP/s for matmul, wall time for the rest.

| Workload | CPU | APU | Ratio |
| --- | ---: | ---: | ---: |
| sgemm fp32 1024³ | 503.8 GF | 1269.5 GF | 2.5× |
| sgemm fp32 2048³ | 723.9 GF | 1401.5 GF | 1.9× |
| sgemm fp32 4096³ | 699.3 GF | **1958.7 GF** | 2.8× |
| sgemm fp16 2048³ | 0.4 GF | 2002.9 GF | 4549× |
| sgemm fp16 4096³ | 0.4 GF | **2338.4 GF** | 5839× |
| conv2d 16×64×128×128 | 96.2 ms | 14.9 ms | 6.4× |
| **attention 4×12×1024×64** | **20.0 ms** | 50.7 ms | **0.40×** |

## Four things worth knowing

**The fp32 gap is only 2–3×.** The CPU reaches ~700 GFLOP/s, which is a
respectable AVX2 result on eight cores. Anyone expecting the usual
order-of-magnitude CPU-to-GPU gap will be disappointed: on this hardware the
iGPU is a modest speedup for fp32, not a different league. The APU does reach
1958 GF at 4096³ — about 80 % of its 2.46 TFLOP/s theoretical peak, so it is
working well; the CPU is simply also good.

**fp16 on the CPU is unusable, not merely slow.** 0.4 GFLOP/s is roughly 1700×
slower than the same CPU doing fp32 — PyTorch has no optimised CPU fp16 matmul
and falls back to element-at-a-time conversion. Use fp32 or bf16 on CPU. On the
APU fp16 reaches 2338 GF, above the fp32 peak, consistent with packed fp16
issuing two operations per lane.

**Convolution is where the APU earns its keep** — 6.4× faster, the largest
honest win here. MIOpen is doing its job, which is good news for anything
image-shaped (ComfyUI, Stable Diffusion, vision models).

**Attention is faster on the CPU**, by 2.5×. This is the one result that should
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
