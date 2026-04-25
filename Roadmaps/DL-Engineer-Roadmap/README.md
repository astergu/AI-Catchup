# DL Engineer Roadmap — Top-Down

The "DL Engineer" / "Research Engineer" / "Applied Scientist" role at frontier labs (DeepMind, Anthropic, OpenAI, Meta-FAIR, Google Research, Mistral, xAI, NVIDIA Research, ByteDance Seed, DeepSeek) is **not** the same as an AI Engineer building products on top of APIs. The DL Engineer **trains the model in the first place**, debugs why the loss spiked at step 47K, owns the parallelism strategy across a 1024-GPU job, and reads a NeurIPS paper on Monday and has it reproduced by Friday.

This roadmap is built top-down: **start from the industrial systems frontier labs actually train and ship; decompose each into sub-systems that solve specific problems; surface the hardware, numerical, and statistical constraints that drive every choice; pull theory in only when a real problem demands it.**

If you also want the Chinese stage-based plan, see [`README_cn.md`](./README_cn.md).

---

## Table of Contents

- [Part 0 — How to Read This Roadmap](#part-0--how-to-read-this-roadmap)
- [Part 1 — What DL Engineers Are Actually Hired To Do](#part-1--what-dl-engineers-are-actually-hired-to-do)
- [Part 2 — The Hardware & Numerical Constraints Behind Every Decision](#part-2--the-hardware--numerical-constraints-behind-every-decision)
- [Part 3 — Industrial Solutions, Decomposed](#part-3--industrial-solutions-decomposed)
  - [3.1 Training a Foundation Model — the meta-task](#31-training-a-foundation-model--the-meta-task)
  - [3.2 Distributed Training & the Parallelism Stack](#32-distributed-training--the-parallelism-stack)
  - [3.3 Numerical Stability & Training Dynamics](#33-numerical-stability--training-dynamics)
  - [3.4 Vision: From CNN to ViT to Modern Hybrid](#34-vision-from-cnn-to-vit-to-modern-hybrid)
  - [3.5 Language Modeling: The Training Stack](#35-language-modeling-the-training-stack)
  - [3.6 Multimodal & Cross-Modal Models](#36-multimodal--cross-modal-models)
  - [3.7 Diffusion & Generative Models](#37-diffusion--generative-models)
  - [3.8 Reinforcement Learning (Classic, RLHF, Robotics, Game-Playing)](#38-reinforcement-learning-classic-rlhf-robotics-game-playing)
  - [3.9 Specialized Architectures (GNNs, Science, Geometric)](#39-specialized-architectures-gnns-science-geometric)
  - [3.10 Inference: From Research Checkpoint to Production](#310-inference-from-research-checkpoint-to-production)
  - [3.11 Evaluation, Benchmarks & Reproducibility](#311-evaluation-benchmarks--reproducibility)
  - [3.12 Mechanistic Interpretability & Scaling Laws](#312-mechanistic-interpretability--scaling-laws)
- [Part 4 — Reading & Reproducing Research (the core craft)](#part-4--reading--reproducing-research-the-core-craft)
- [Part 5 — Foundations to Backfill (Just-in-Time)](#part-5--foundations-to-backfill-just-in-time)
- [Part 6 — Interview Signal at Frontier Labs](#part-6--interview-signal-at-frontier-labs)
- [Part 7 — Suggested Project Track](#part-7--suggested-project-track)
- [Part 8 — DeepMind-Style Research Questions](#part-8--deepmind-style-research-questions)
- [References](#references)

---

## Part 0 — How to Read This Roadmap

For each industrial solution:

1. **The real problem** — what *technical* outcome the lab needs (a converged 70B checkpoint, a vision encoder beating SOTA, an RL agent that doesn't collapse to a degenerate policy) and why naive approaches fail.
2. **Sub-solutions** — the components a production training/inference system decomposes into.
3. **What each resolves** — and the new problem it introduces.
4. **Pros / cons / when it's the wrong tool**.
5. **Hardware & numerical footprint** — VRAM, FLOPs, MFU, communication cost, precision constraints.
6. **Failure modes you'll see** at scale — loss spikes, NaNs, gradient pathologies, eval contamination, throughput collapse.

Theory (linear algebra, calculus, probability, optimization, information theory) is in Part 5 — pull it in when a problem in Part 3 forces you to.

---

## Part 1 — What DL Engineers Are Actually Hired To Do

The job titles vary (Research Engineer, Member of Technical Staff, Applied Scientist, ML Performance Engineer) but the **problem surface is small**:

| Surface | Concrete examples | Primary technical KPI | Where it lives |
|---|---|---|---|
| **Frontier LM pretraining** | Llama, Claude, GPT, Gemini, DeepSeek, Mistral pretraining runs | Eval suite score per FLOP, training stability, MFU | Frontier labs only |
| **Post-training / alignment** | SFT, RLHF, DPO, distillation, model-merge | Win-rate vs base, refusal calibration, capability preservation | Frontier labs + open-weights teams |
| **Vision foundation models** | CLIP, SigLIP, DINOv2, Sapiens, SAM, ViT-22B | ImageNet/ADE20K/COCO + zero-shot transfer | Vision teams |
| **Multimodal foundation models** | GPT-4o, Gemini, Chameleon, Qwen-VL, LLaVA | Cross-modal benchmark suites | Multimodal teams |
| **Diffusion / generative** | SDXL, Flux, Imagen, Veo, Sora | FID/CLIP-score, human preference, sample efficiency | Image/video gen teams |
| **RL & decision-making** | AlphaStar, AlphaZero, Dreamer, RT-2, RLHF for LMs | Skill rating, sample efficiency, stability | DeepMind, OpenAI, robotics teams |
| **Scientific DL** | AlphaFold, AlphaMissense, GraphCast, Materials | Domain benchmark (e.g. CASP, RMSD) | DeepMind, Isomorphic, weather teams |
| **Speech & audio** | Whisper, SeamlessM4T, AudioLM | WER, BLEU, MOS | Audio teams |
| **Robotics / world models** | RT-2, π0, Genie, Dreamer-V3 | Success rate, sim-to-real gap, sample efficiency | Robotics labs |
| **ML systems / kernels / compilers** | FlashAttention, Triton kernels, XLA, Pallas, Inductor | MFU, kernel speedup, memory-traffic reduction | Performance / systems teams |
| **Evaluation & safety** | MMLU, GPQA, MATH, capability + red-team eval suites | Eval reliability, contamination resistance | Evals / safety teams |
| **Interpretability** | SAEs, circuits, probing, activation patching | Feature recovery, causal explanation quality | Interp teams (Anthropic, DeepMind, etc.) |

### What you're actually graded on

A DL Engineer is graded on **four** things, and **none** of them are "knows the most papers":

1. **Can you make the loss go down?** — converge a non-trivial model end-to-end, not just nano-scale toys.
2. **Can you debug at scale?** — when a 1024-GPU job spikes at step 80K, can you find the root cause (data, optimizer, init, mixed precision, hardware, race condition)?
3. **Can you read a paper and produce a working reimplementation?** — ideally that matches reported numbers within noise; bonus if you find their bug.
4. **Do you have hardware/numerical taste?** — can you predict a memory blowup, a kernel inefficiency, or a precision gotcha *before* it happens?

Senior DL engineers add a fifth: **can you look at a noisy loss curve, an ablation table, and a cluster utilization plot — and tell which experiment to run next?** That's research-engineer judgment, and it's what frontier labs are buying.

### The compute / data / quality triangle

```
        Eval quality
              /\
             /  \
            /    \
           / pick \
          / two-ish\
         /__________\
       Compute    Data
```

- **Compute + Data → quality up.** Default scaling (Chinchilla, post-Chinchilla). Expensive.
- **Quality + low data → tons of compute.** Synthetic data, curriculum, RL bootstrapping. Frontier-lab playbook.
- **Quality + low compute → ruthless data curation + distillation.** The Phi / Gemma / Qwen-Small playbook. Where small labs win.

Every architectural and training-recipe decision is a move along this triangle.

---

## Part 2 — The Hardware & Numerical Constraints Behind Every Decision

Skip this part and you'll forever be confused about *why* labs made the choices they made. Almost every "trick" in modern DL training exists to dodge a specific hardware or numerical bottleneck.

### 2.1 GPU memory: where every byte goes (training)

For a Transformer in **training**, VRAM splits roughly as:

```
VRAM = parameters + gradients + optimizer states + activations + framework overhead
```

Adam-style optimizer with mixed precision (BF16 weights + FP32 master + FP32 mom + FP32 var):

| Item | Bytes/param | Notes |
|---|---|---|
| Weights (BF16) | 2 | What forward pass uses |
| Gradients (BF16 or FP32) | 2 or 4 | FP32 master grads are common |
| FP32 master weights | 4 | For numerically stable update |
| Adam `m` (1st moment) | 4 | FP32 |
| Adam `v` (2nd moment) | 4 | FP32 |
| **Total** | **~16** | "16× rule of thumb" |

So a 7B-param model needs ≥112 GB just for parameters/grads/optimizer state — *before* activations. **This is why FSDP / ZeRO / sharded optimizer state are not optional at scale.**

**Activation memory** is the second monster. Without checkpointing, transformer activations grow as `O(B · L · S · H)` where `B`=batch, `L`=layers, `S`=seq, `H`=hidden. For long sequences this can dominate parameters. Activation checkpointing (recompute on backward) trades ~30% extra compute for a large memory cut.

**KV cache** (inference-only) — see §3.10 and the AI-Engineer-Roadmap for the per-token math.

### 2.2 GPU compute hierarchy

| Memory | Bandwidth (H100) | Latency | Capacity |
|---|---|---|---|
| Registers | ~25 TB/s/SM | ~1 cycle | small per SM |
| Shared memory (SRAM) | ~20 TB/s | ~20 cycles | ~228 KB / SM |
| L2 cache | ~7 TB/s | ~150 cycles | 50 MB |
| **HBM (global VRAM)** | **~3.3 TB/s** | ~400 cycles | 80 GB |
| PCIe Gen5 / NVLink | up to ~900 GB/s NVLink | μs | inter-GPU |

The HBM ↔ SRAM gap is the source of most modern kernel work. **FlashAttention** wins by keeping QK softmax in SRAM and never materializing the N² attention matrix in HBM. **Persistent kernels** keep weights in registers across micro-batches. Every meaningful kernel-level speedup is fundamentally a memory-traffic story.

### 2.3 The compute-bound vs memory-bound regimes (the roofline you'll redraw 100 times)

Arithmetic intensity = `FLOPs / bytes loaded`. Plot it against the hardware ridge `peak_FLOPs / peak_bandwidth`:

| Operation | Intensity | Regime | Implication |
|---|---|---|---|
| GEMM (large) | high | compute-bound | Maximize MFU; precision (FP8/FP16) helps |
| Pointwise (LayerNorm, GELU) | low | memory-bound | Fuse with neighbors |
| Softmax | low | memory-bound | Why FlashAttention exists |
| LM decode step | very low | memory-bound | Why batching, MQA/GQA, speculative decoding exist |
| Embedding lookup | very low | memory-bound | Why embedding sharding is its own subfield |

**MFU (Model FLOPs Utilization)** = achieved FLOPs / peak FLOPs. Frontier-lab pretraining targets are 40–60%. Below 30% means you're leaving real money on the table.

### 2.4 GPUs vs TPUs (and why the difference shapes architecture choices)

| Property | NVIDIA H100 / B200 | TPU v5p / v6 (Trillium) |
|---|---|---|
| Compute element | SMs, tensor cores | MXUs (matrix multiply unit), VPUs |
| Native precisions | BF16 / FP8 / FP4 (B200) | BF16 / INT8; FP8 emerging |
| Memory hierarchy | HBM → L2 → SRAM (228KB) → registers | HBM → VMEM (32MB) → vector registers |
| Topology | NVLink islands of 8 (NVL72: 72) | 3D torus pods of 256→8192 chips |
| Programming model | CUDA / Triton | XLA / JAX / Pallas |
| Key win | Mature ecosystem; FP4/FP8 fastest | Massive collective bandwidth; deterministic |
| Key cost | NVLink islands + IB across nodes | Less low-level control; XLA opinions |

Implications for DL engineers:
- **GPU world**: PyTorch + FSDP + custom Triton kernels + NCCL.
- **TPU world**: JAX + `jit` + `pjit` + `shard_map` + collective primitives.
- Frontier labs pick deliberately: Anthropic mostly TPU + GPU, Meta GPU, Google TPU, OpenAI mostly GPU, DeepSeek GPU, Apple TPU+GPU. **You should be able to reason about both.**

### 2.5 Numerical precision — the source of half your training pain

| Format | Bits (sign / exp / mantissa) | Range | Precision (eps) | Where it hurts |
|---|---|---|---|---|
| FP32 | 1 / 8 / 23 | ±1e±38 | ~1e-7 | Default master weights / Adam states |
| **BF16** | 1 / 8 / 7 | ±1e±38 | ~3e-3 | FP32 range, less precision; modern default |
| FP16 | 1 / 5 / 10 | ±6.5e4 | ~5e-4 | Overflows easily — *needs loss scaling* |
| FP8 (E4M3 / E5M2) | 1 / 4 / 3 or 1 / 5 / 2 | small | low | H100+/B200; needs per-tensor scaling |
| FP4 (B200) | 1 / 2 / 1 (variants) | tiny | very low | Frontier; experimental for training |
| INT8 / INT4 | integer | quantized | — | Inference-side; calibration matters |

**Why it matters:**
- **Attention softmax** must accumulate in FP32 — even a single FP16 softmax can overflow at long sequence length and produce NaNs.
- **Loss scaling** (FP16 era) — multiply loss by ~2¹⁵, divide grads back, to keep small gradients above FP16 underflow. Mostly obsolete in BF16.
- **FP8 training** requires careful per-tensor scaling (DelayedScaling, AMAX history); naive FP8 silently loses tail precision.
- **BF16 weight + FP32 master** is the modern standard; when you see a paper *not* doing this, ask why.

### 2.6 Communication: the part that kills throughput at scale

Collective primitives you must internalize:

| Primitive | What it does | When it shows up |
|---|---|---|
| **all-reduce** | Sum across workers, replicate result | DP gradient sync |
| **all-gather** | Concat shards across workers | FSDP forward (gather params) |
| **reduce-scatter** | Sum + shard | FSDP backward |
| **all-to-all** | Permute shards across workers | MoE expert routing, sequence parallelism |
| **broadcast / scatter / gather** | One-to-many / many-to-one | Init, eval |

**Scaling rules of thumb:**
- TP stays inside an NVLink island (≤8 GPUs, or NVL72 if you have it). Cross-NVLink TP is a throughput cliff.
- DP / FSDP scales across nodes via IB / RoCE — bandwidth-tolerant because gradients are reduced once per step.
- PP scales across nodes — but pipeline bubbles can eat 10–30% of compute if not chunked carefully (1F1B, interleaved).
- EP (expert parallelism) for MoE has all-to-all communication — the brutal one; topology matters enormously.
- **Compute-comm overlap** is mandatory at scale. Look up `torch.distributed.fsdp.BackwardPrefetch`, JAX `donate_argnums`, custom NCCL streams.

### 2.7 Cost of training (rough 2024–2025 reference)

| Run | Compute | Wall time | Cost order of magnitude |
|---|---|---|---|
| GPT-2 reproduction (124M) | ~1 GPU-day on a 4090 | hours | <$10 |
| Llama-2 7B from scratch | ~180K H100-hours | weeks on hundreds of GPUs | ~$300K |
| Llama-3 70B from scratch | ~6.4M H100-hours | weeks on 1024+ GPUs | ~$10M |
| Frontier-class LM (~1T-class effective) | ~10⁸ H100-hours | months on 10K+ GPUs | $50M–$500M |

**Implication for the DL engineer at any career level:** every decision that touches a frontier run has dollar-denominated consequences. "Let's just rerun it" is a million-dollar sentence at the top end. The engineering tooling — checkpointing, deterministic data loaders, eval harnesses, telemetry — exists because you cannot afford to lose a long run.

---

## Part 3 — Industrial Solutions, Decomposed

### 3.1 Training a Foundation Model — the meta-task

> The single most consequential thing DL engineers do. Every other section is downstream of this one.

**The real problem.** You have N GPUs/TPUs for T weeks. Produce the best model your compute and data allow. Concretely: maximize eval-suite score per training-FLOP, while keeping the run *stable* (no irrecoverable spikes, no silent quality regression, no checkpoint corruption).

**The training run as a system:**

```
            ┌──────────────────────────────────────────────────┐
            │  Data: web crawl, books, code, math, synthetic   │
            │  Curation, dedup, quality filter, mix, tokenize  │
            └──────────────────────────────────────────────────┘
                                 │
                                 ▼
            ┌──────────────────────────────────────────────────┐
            │  Architecture: depth, width, MQA/GQA/MLA, MoE    │
            │  Embedding tying, RoPE, normalization, init      │
            └──────────────────────────────────────────────────┘
                                 │
                                 ▼
            ┌──────────────────────────────────────────────────┐
            │  Optimizer: AdamW / Adafactor / Lion / Shampoo   │
            │  LR schedule (warmup → cosine / WSD), wd, β1/β2  │
            └──────────────────────────────────────────────────┘
                                 │
                                 ▼
            ┌──────────────────────────────────────────────────┐
            │  Parallelism: DP × TP × PP × SP × CP × EP        │
            │  Activation checkpointing, FSDP, ZeRO            │
            └──────────────────────────────────────────────────┘
                                 │
                                 ▼
            ┌──────────────────────────────────────────────────┐
            │  Precision: BF16 / FP8 with master + scaling     │
            └──────────────────────────────────────────────────┘
                                 │
                                 ▼
            ┌──────────────────────────────────────────────────┐
            │  Telemetry: loss, grad norm, MFU, throughput,    │
            │  per-layer activation/grad norms, power, OOMs    │
            └──────────────────────────────────────────────────┘
                                 │
                                 ▼
            ┌──────────────────────────────────────────────────┐
            │  Eval harness: held-out loss, downstream evals,  │
            │  contamination check, capability tracking        │
            └──────────────────────────────────────────────────┘
                                 │
                                 ▼
            ┌──────────────────────────────────────────────────┐
            │  Checkpoint: sharded, async, fault-recovery      │
            └──────────────────────────────────────────────────┘
```

**Sub-decisions and what each resolves:**

| Decision | What it controls | Default at frontier (2024–2025) |
|---|---|---|
| Data mix | Capability profile, eval scores, multilingual coverage | Heavily curated web + code + math + synthetic; mix tuned by ablation |
| Architecture | Capacity, inference cost, training cost | Decoder-only Transformer, GQA/MLA, RoPE, RMSNorm, SwiGLU |
| Tokenizer | Token efficiency per language/domain | BPE, ~128K–256K vocab, byte-fallback |
| Optimizer | Convergence quality + stability | AdamW (β1=0.9, β2=0.95, wd=0.1) is the unironic default |
| LR schedule | Final loss + when you can stop | Linear warmup → cosine; or WSD (warmup–stable–decay) for "anytime" stops |
| Batch size | Stability vs throughput | Critical batch size scales with model size; tokens/step grows during training |
| Sequence length | Context capability + activation memory | Train short → extend long (RoPE scaling, YaRN); or curriculum |
| Parallelism plan | Throughput, MFU, fits in memory | DP+FSDP at small scale; +TP+PP+SP at frontier; +EP for MoE |
| Precision | Throughput vs stability | BF16 weights + FP32 master + FP8 selectively (matmuls) |
| Init scheme | Stability at depth | Scaled init (μP-aware, GPT-NeoX-style); deep-norm for ultra-deep |
| Loss components | What you actually optimize | Plain CE for pretrain; +z-loss for stability; +aux losses for MoE balance |
| Checkpoint cadence | Recoverability vs cost | Async sharded every N steps; smaller "rolling" + larger "milestone" |

**Pros / cons of frontier-vs-academic-vs-startup recipes:**

| Recipe | Pros | Cons |
|---|---|---|
| Frontier (huge data, huge compute, conservative arch) | Reliable scaling; well-understood failure modes | $$$; little arch novelty |
| Distillation-heavy (Phi, Gemma) | Strong small models; cheap | Bounded by teacher; legal grey area when using closed models |
| MoE (Mixtral, DeepSeek-V3, Qwen-MoE) | Capacity per FLOP; cheap inference | Routing instability; serving complexity; load-balance loss |
| Long-context-from-scratch | Capability included free | Activation memory explodes; RoPE-scaling at extension is cheaper |
| Multimodal-from-scratch | Native cross-modal capability | Data, eval, and tokenization all 2× harder |
| RL-bootstrapped pretrain (R1-style) | Reasoning emerges | Reward design is its own black art |

**Failure modes you will see (the famous ones, in roughly the order they bite):**
- **Slow loss decrease** — almost always data quality / mix or LR schedule.
- **Spike at ~1–10K steps** — usually LR warmup too short, init too aggressive, or bad batch.
- **Spike at ~50K+ steps with recovery** — survivable; log batches that triggered.
- **Spike + divergence** — bad. Roll back to last good checkpoint, lower LR, mask the offending data shard.
- **NaN in attention softmax** — long-seq + FP16; switch to BF16 or accumulate softmax in FP32.
- **MoE expert collapse** — auxiliary load-balance loss missing or too weak.
- **Eval contamination** — your "huge gain" is actually leaked test data.
- **Throughput collapse mid-run** — usually a hardware fault, dataloader stragglers, or NCCL ring degradation.
- **Checkpoint corruption** — "we lost a week" — async writes interrupted by preemption.

Recommended deep reads: **Stanford CS336 (Language Modeling from Scratch)** lecture notes, the **Llama 3 paper**, **DeepSeek-V3 technical report**, **OLMo** training writeups, and the **Hugging Face Ultra-Scale Playbook**.

---

### 3.2 Distributed Training & the Parallelism Stack

> A 70B model does not fit on one GPU. The way you split it across thousands of GPUs *is* the engineering job.

**The real problem.** A frontier model's *parameters + activations + optimizer state* is hundreds of GB to TB. Compute one step in <T seconds on N GPUs. Hide communication behind compute. Recover from any GPU dying mid-step.

**The six parallelism dimensions** (every modern framework composes these):

| Dim | What gets sharded | Communication pattern | Where it lives |
|---|---|---|---|
| **DP (Data Parallelism)** | Same model, different batch shards | all-reduce on grads | Across nodes |
| **FSDP / ZeRO** | Params + grads + optimizer states across DP workers | all-gather (fwd/bwd) + reduce-scatter (bwd) | Across nodes |
| **TP (Tensor Parallelism)** | Each layer's weights split column/row-wise | all-reduce inside each layer | Inside NVLink island |
| **PP (Pipeline Parallelism)** | Layers split across workers | point-to-point sends across stages | Across nodes (typically) |
| **SP (Sequence Parallelism)** | Sequence dimension sharded for activations | all-gather + reduce-scatter | Often combined with TP |
| **CP (Context Parallelism)** | Sequence sharded for very long contexts | ring-attention all-reduce | Long-context training |
| **EP (Expert Parallelism)** | MoE experts placed on different workers | all-to-all per token | MoE only |

**ZeRO stages (DeepSpeed naming):**
- **Stage 1**: shard optimizer states. Cheap, large memory win.
- **Stage 2**: + shard gradients. More win.
- **Stage 3**: + shard parameters. Equivalent to FSDP. Maximum memory savings, more comm.

**Composition rules of thumb:**

| Model size | Typical strategy |
|---|---|
| <1B | Pure DP / FSDP, single node fine |
| 1–10B | FSDP across nodes; gradient accumulation |
| 10–70B | FSDP + TP=8 within node + PP across nodes; activation checkpointing |
| 70B–MoE | + EP, careful all-to-all topology |
| 100B+ dense / 1T+ MoE | Frontier; bespoke per-cluster recipes |

**Frameworks (and where each shines):**

| Framework | Best for | Notes |
|---|---|---|
| **PyTorch FSDP / FSDP2** | General DP+sharding | Modern PyTorch default |
| **DeepSpeed** | ZeRO-3, CPU/NVMe offload | Verbose; rich features |
| **Megatron-LM** | TP+PP+SP for LLMs | Research-grade |
| **Megatron-DeepSpeed** / **Megatron-Core** | Combined | Frontier-grade composition |
| **NVIDIA NeMo** | End-to-end LLM/multimodal | Productized |
| **JAX + `pjit` / `shard_map`** | TPU + multi-host GPU | Functional, composable |
| **TorchTitan** | New PyTorch reference for big training | Worth reading the source |

**Failure modes:**
- **Throughput well below MFU target** — usually comm-bound; check overlap, ring topology, NCCL env.
- **Stragglers** — one slow GPU drags the all-reduce.
- **Mixed-precision instability** with TP — collective ops in FP16; switch to BF16 or accumulate higher.
- **Pipeline bubbles** — micro-batch count too small.
- **OOM on `step()`** — optimizer state not sharded yet (ZeRO-1 vs 2 vs 3).
- **Checkpoint shape mismatch on resume** — you changed parallelism between runs; convert via re-shard.

---

### 3.3 Numerical Stability & Training Dynamics

> **This is the single highest-leverage skill that separates "I know PyTorch" from "I can train at scale."**

**The real problem.** Modern training mixes precisions, sums billions of small numbers, and runs for trillions of steps. Anywhere precision can fail, it will. NaNs are a feature of large-scale training, not a bug.

**Sub-areas every DL engineer must own:**

| Symptom | Likely cause | Diagnosis | Fix |
|---|---|---|---|
| Loss → NaN suddenly | Softmax overflow / invalid log / div-by-zero | Per-layer activation/grad norms blow up | Accumulate softmax in FP32; clip logits |
| Loss spike with recovery | Bad data shard or LR too high | Log batch indices around the spike | Skip-bad-batch + LR cooldown |
| Loss spike → divergence | Same, but past the recovery threshold | Restore from last good checkpoint | Lower LR, mask data, restart |
| Gradient norm exploding | LR too high / bad init / unbounded activations | Track grad norm per layer | Gradient clipping (1.0 typical for LMs) |
| Gradient norm collapsing → 0 | Saturating activations / dead ReLUs / wd too high | Track activation norms | Better init / activation / lower wd |
| Slow convergence | LR schedule, data quality, capacity | Loss vs FLOPs vs scaling-law fit | Compare to scaling law |
| Output identical across batches | Embedding collapse / temp init issue | Cosine sim of activations | Verify init; check embedding tying |
| FP16 silently underflowing | Loss scaling missing / too low | Compare against FP32 / BF16 reference | Use BF16; or dynamic loss scaling |
| FP8 silent quality regression | Per-tensor scaling / AMAX history | Side-by-side eval vs BF16 | Tune scaling recipe (DelayedScaling) |

**Numerical tricks every DL engineer should be able to derive:**

- **log-sum-exp**: `log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i − max(x)))`. Underpins stable softmax and stable cross-entropy.
- **Stable softmax**: subtract max before exp. Without this, FP16 attention logits at long context overflow.
- **Stable cross-entropy from logits** (no explicit softmax): `−x[y] + log-sum-exp(x)`.
- **Stable sigmoid**: `1 / (1 + exp(−x))` overflows for x≪0; piecewise on sign.
- **Welford's online variance** for streaming statistics.
- **Kahan summation** for very long reductions.

**Initialization schemes — what each resolves:**

| Init | Resolves | When |
|---|---|---|
| Xavier / Glorot | Variance preservation in shallow nets | Old MLPs / sigmoid era |
| He / Kaiming | ReLU-aware variance | Most CNN settings |
| Scaled (μP, μ-Transfer) | Hyperparams transfer across model widths | Frontier training; saves sweep cost |
| GPT-NeoX-style scaled init | Stability for deep transformers | Post-2022 LLMs |
| DeepNorm | Stability beyond 100 layers | Deep encoder/decoder models |

**Training-stability checklist before you launch a run:**
- [ ] Init verified: forward pass at init, activation norms ~1 across layers.
- [ ] Gradient flow verified: backward at init, grad norms not vanishing/exploding.
- [ ] LR warmup ≥ 1–2K steps for serious training.
- [ ] Gradient clipping enabled (typically 1.0).
- [ ] Loss accumulation in FP32; softmax in FP32; norms in FP32.
- [ ] Telemetry on: per-layer activation norm, grad norm, weight norm, optimizer state norm.
- [ ] Eval harness ready before step 0; loss-only is not enough.
- [ ] Checkpoint resume tested *before* the long run.

---

### 3.4 Vision: From CNN to ViT to Modern Hybrid

> Vision is the field that taught DL the value of architectural priors — and then the lesson that with enough data, you can drop them.

**The real problem.** Take pixels and produce useful representations: classify, detect, segment, embed for retrieval, condition generative models, solve a downstream task.

**Sub-architectures and what each resolves:**

| Family | Year | Resolves | Key idea |
|---|---|---|---|
| **AlexNet / VGG** | 2012–14 | First DL win on ImageNet | Stack convs + ReLU + dropout |
| **ResNet** | 2015 | Vanishing gradients beyond 20 layers | Residual connections — *the* idea |
| **Inception / EfficientNet / RegNet** | 2014–20 | FLOP-quality tradeoff | Width/depth/resolution co-scaling |
| **U-Net** | 2015 | Dense prediction (segmentation, later diffusion) | Encoder–decoder with skip connections |
| **DETR** | 2020 | Object detection without anchors | Transformer over a set of object queries |
| **ViT** | 2020 | "Vision needs no convolution if you have data" | Patch + position-embed + Transformer |
| **Swin / Hiera / convnext** | 2021–22 | Locality + scale | Hierarchical or pure-conv responses to ViT |
| **MAE** | 2021 | Self-supervised pretraining for vision | Masked patch reconstruction |
| **DINO / DINOv2** | 2021–23 | Strong self-supervised features | Self-distillation + multi-crop |
| **SAM / SAM 2** | 2023–24 | Universal interactive segmentation | Promptable mask predictor over a foundation backbone |
| **CLIP / SigLIP** | 2021–23 | Image–text aligned embeddings | Contrastive (CLIP) / sigmoid (SigLIP) loss over (image, caption) pairs |
| **Hybrid (DiT, Sapiens, Florence-2)** | 2023+ | Specialization on top of generic backbones | Backbone + task-specific heads |

**Pros / cons:**

| Family | Pros | Cons |
|---|---|---|
| CNN | Strong inductive bias; data-efficient at small scale | Doesn't scale as cleanly past ~1B params |
| ViT | Scales beautifully with data; unified with LM stack | Data-hungry; positional encoding choices matter |
| Hybrid (conv-stem + Transformer) | Best of both at modest scale | More moving parts |
| Self-sup (MAE, DINO) | No labels needed | Eval is downstream-task dependent |
| Contrastive (CLIP/SigLIP) | Zero-shot transfer; embedding reuse | Needs huge image-text pairs; caption quality dominates |

**Hardware footprint considerations:**
- ViT-22B trained on TPU pods (Google).
- CLIP/SigLIP large-scale runs are dominated by data loader throughput, not compute (image decoding is CPU/GPU-bound).
- Long-image inputs (high-res) explode activation memory the same way long-sequence LMs do.

**Failure modes:**
- ViT trained on ImageNet-1K alone overfits; needs JFT / LAION / DataComp scale.
- CLIP "modality gap" — text and image embeddings cluster separately; affects downstream metric.
- Caption noise dominates large-scale contrastive training; recaptioning (BLIP, LLaVA-recap) is now a default step.

---

### 3.5 Language Modeling: The Training Stack

> The single most active area in DL today. Most labs above 50 people have a language modeling team.

**The real problem.** Train a model that predicts the next token well enough that downstream eval scores transfer. Then *post-train* it to follow instructions, refuse harmful queries, reason, use tools, and so on.

**The stages of a modern LM:**

| Stage | What it does | Compute share | Tools |
|---|---|---|---|
| **Pretraining** | Next-token prediction on trillions of tokens | ~99% of total compute | Megatron, JAX/XLA, Llama-style trainer |
| **Mid-training / annealing / continued pretrain** | Targeted data (math, code, multilingual) toward end | 1–5% | Same trainer; different mix |
| **SFT (Supervised Fine-Tuning)** | Imitate high-quality demos | <1% | LoRA-friendly; full FT for frontier |
| **Preference tuning (DPO / IPO / KTO / ORPO / SimPO / RLHF)** | Optimize for *preferences* | <1% | TRL, Axolotl, OpenRLHF |
| **RL with verifiable rewards (RLVR / R1-style)** | Reasoning from execution feedback | Variable; growing | OpenRLHF, verl, custom |
| **Distillation** | Compress big-teacher → small-student | Small | Standard SFT loop on teacher outputs |

**Architectural sub-decisions** (and what each resolves):

| Choice | Resolves | Modern default |
|---|---|---|
| **Attention type (MHA / MQA / GQA / MLA)** | KV-cache memory at long context | GQA (Llama, Mistral); MLA (DeepSeek) for ultra-compressed cache |
| **Positional encoding** | Length generalization | RoPE everywhere; YaRN/NTK-aware scaling for extension |
| **Normalization** | Stability at depth | RMSNorm; pre-norm placement |
| **FFN activation** | Throughput + quality | SwiGLU |
| **Tied embeddings** | Parameter saving | Yes for small models; no for frontier |
| **MoE or dense?** | Capacity-per-FLOP vs simplicity | Both; MoE for cheap inference at huge capacity |
| **Vocabulary size** | Token efficiency vs softmax cost | 128K+ for multilingual; bigger for Asian-language coverage |

**Pretraining data — the part papers undersell:**
- **Web crawl** (CommonCrawl, FineWeb, RedPajama, DCLM) — bulk.
- **Code** (StackV1/V2) — big reasoning gains.
- **Math** (OpenWebMath, MegaMath, proof-pile) — direct math gains.
- **Books / academic** (better quality, careful licensing).
- **Synthetic data** (instruction-style, generated by stronger models) — increasingly dominant in mid-training.
- **Filtering & deduplication** (MinHash-LSH, classifier-based quality, contamination removal).

**Pros / cons of post-training methods:**

| Method | Pros | Cons |
|---|---|---|
| SFT | Simple, predictable, format injection | Low ceiling; can hurt diversity |
| RLHF (PPO) | Highest historical quality | Unstable; expensive; reward hacking |
| DPO and friends | Cheap, stable, reproducible | Slightly lower ceiling than well-tuned PPO at scale |
| RLAIF / Constitutional AI | Scales human preferences | Judge bias leaks in |
| RLVR (verifiable rewards) | Real reasoning gains (R1-style) | Reward design is brittle |
| Distillation | Best small-model recipe | Bounded by teacher; legal |

**Failure modes:**
- "MMLU went up, MT-bench went down" — over-aligned for benchmark; lost generality.
- Refusal creep after over-aggressive safety SFT.
- Sycophancy after preference-tuning on preference-of-the-judge.
- RLHF reward hacking — model emits *plausible-looking* but wrong answers.
- Eval contamination — pretraining ate the test set.

---

### 3.6 Multimodal & Cross-Modal Models

> Increasingly the default. "Pure-text-only" models will be the edge case in 5 years.

**Sub-areas:**

| Modality | Key models | What they solve |
|---|---|---|
| Vision–Language (VLM) | CLIP, SigLIP, Flamingo, IDEFICS, LLaVA, Qwen-VL, Chameleon | Image+text reasoning, OCR, doc understanding |
| Speech recognition (ASR) | Whisper, USM, SeamlessM4T | Transcription |
| Speech synthesis (TTS) | VALL-E, NaturalSpeech, OpenAI TTS | Voice gen |
| Audio-LM (general audio) | AudioLM, MusicLM, MusicGen | Music, sound, speech in one model |
| Video understanding | VideoMAE, V-JEPA, Video-LLaVA | Temporal vision |
| Video generation | Sora, Veo, Genmo, Mochi | Text-to-video |
| Native multimodal (text+image+audio+video tokenized) | GPT-4o, Gemini, Chameleon, AnyGPT | Single model across modalities |

**Architectural patterns and tradeoffs:**

| Pattern | Pros | Cons |
|---|---|---|
| Frozen encoder + projector + LM (LLaVA-style) | Cheap; modular; works | Bounded by encoder; weak at fine-grained vision |
| Cross-attention to vision features (Flamingo, IDEFICS) | Better grounding | More complex training; fewer open weights |
| Native multimodal pretraining (GPT-4o, Gemini, Chameleon) | Best quality | Frontier-only; data + eval much harder |
| Tokenize-everything (image VQ-tokens + text tokens) | Unified discrete sequence | Image quality bounded by tokenizer |
| Diffusion for vision generation, AR for text (DiT + LM) | Best image quality + text reasoning | Two stacks to train |

**Hardware reality.** Image patches are tokens — a 224×224 image at patch=14 is ~256 vision tokens; high-res or multi-image inputs push to thousands. Multimodal context windows blow up activation memory. Audio is even worse — Whisper-style chunking exists for a reason.

---

### 3.7 Diffusion & Generative Models

> A separate craft from LMs. Different math, different sampling, different evaluation.

**The real problem.** Generate samples from a target distribution (images, audio, video, molecules). The model learns to denoise from random noise back toward the data manifold.

**Sub-families:**

| Family | What it solves | Pros | Cons |
|---|---|---|---|
| **Autoencoders / VAE** | Latent compression | Simple; informative latents | Blurry generation |
| **GAN (BigGAN, StyleGAN, PGGAN)** | Sharp samples | High quality images | Training instability; mode collapse |
| **Normalizing flows** | Exact likelihood | Tractable density | Architecturally constrained |
| **Diffusion (DDPM, score-based, SDE/ODE)** | High-quality conditional generation at scale | Stable training; SOTA quality | Slow inference unless distilled |
| **Latent diffusion (LDM, SDXL, SD3, Flux)** | Diffusion in compressed latent | Cheaper compute | Quality bounded by VAE |
| **Rectified flow / flow matching** | Cleaner ODE training objective | Faster sampling; simpler | Newer; tooling catching up |
| **Consistency models / LCM / DMD** | One-step / few-step generation | 10–100× faster sampling | Quality slightly behind teacher |
| **Diffusion Transformers (DiT, SD3, Pixart)** | Scaling diffusion with Transformer backbones | Better scaling laws than U-Net | More compute |
| **Autoregressive image (Parti, MUSE, MaskGIT)** | Reuse LM stack for vision gen | Unified with text models | Quality vs diffusion trade |
| **Video diffusion (Sora, Veo)** | Temporal consistency at scale | SOTA video | Brutal compute; data scarcity |

**Sampling sub-decisions:**

| Choice | What it controls |
|---|---|
| Solver (Euler, Heun, DPM-Solver, DPM++ 2M) | Sample steps vs quality |
| Classifier-free guidance scale | Fidelity vs diversity |
| Negative prompt | Quality + safety in image/video gen |
| Schedule (linear, scaled, cosine, EDM) | Where sample gets allocated |
| CFG variants (rescale, dynamic) | Saturated/burnt samples at high CFG |
| Distillation (DMD, LCM, ADD) | Inference latency |

**Hardware reality.** Diffusion training is memory-bound by activations during the U-Net/DiT backbone *across all denoising timesteps*. Video diffusion at 1024×1024 over 100 frames is among the most expensive workloads in DL today.

**Failure modes:**
- Mode collapse in GANs (still relevant for hybrid setups).
- "Burnt" diffusion samples at high CFG.
- Text-image misalignment (CLIP-loss + diffusion combo doesn't learn long captions well — hence T5-XXL conditioning in SD3).
- Sample diversity collapse after distillation to 1–4 steps.

---

### 3.8 Reinforcement Learning (Classic, RLHF, Robotics, Game-Playing)

> The DeepMind heart of the field. RL has reentered the mainstream via RLHF and RLVR, but classical RL is still where the deepest research happens.

**Sub-families and their ladders:**

**Value-based:**

| Method | Resolves | Pros | Cons |
|---|---|---|---|
| Q-learning | Tabular optimal control | Convergence guarantees | Doesn't scale |
| DQN | Q-learning + deep nets + replay | First Atari-from-pixels result | Overestimation; instability |
| Double DQN, Dueling, Rainbow | DQN's failure modes | Better empirically | Many tricks compose |

**Policy gradient / actor-critic:**

| Method | Resolves | Pros | Cons |
|---|---|---|---|
| REINFORCE | Direct policy optimization | Simple | High variance |
| A2C / A3C | Lower variance via baseline + parallelism | Workhorse | Still on-policy |
| PPO | Trust-region simplification (clipped) | The default for ~everything in 2018–2024 | Many implementation gotchas |
| TRPO | Theoretically clean trust region | Good guarantees | Heavy computation |
| SAC | Maximum-entropy off-policy | Sample-efficient continuous control | Hyperparameter sensitive |
| DDPG / TD3 | Deterministic continuous control | Strong baselines | Brittle |
| GRPO (DeepSeek) | RLHF/RLVR without value model | Memory-efficient at LLM scale | Newer; less battle-tested |

**Model-based:**

| Method | Resolves | Pros | Cons |
|---|---|---|---|
| World models | Sample-efficiency via imagined rollouts | Order-of-magnitude data efficiency | World model errors compound |
| Dreamer V1/V2/V3 | Latent world model + actor-critic | SOTA on many continuous-control benchmarks | Latent dynamics tuning |
| MuZero | World model + tree search, no env access at inference | AlphaGo lineage; strong on board games | Compute-heavy |

**Multi-agent / self-play:**
- AlphaGo / AlphaZero / AlphaStar — self-play + tree search.
- Population-based training.
- Open-ended evolution (POET, paired-open-ended).

**RL meets LLMs:**

| Application | Key idea |
|---|---|
| RLHF (PPO) | Reward model learned from preferences; PPO over LM outputs |
| DPO et al. | Closed-form alternative; no explicit reward model |
| RLAIF / Constitutional AI | Preferences generated by another model |
| RLVR (R1, etc.) | Verifiable rewards (compiler, math checker) drive reasoning |
| Process reward models (PRMs) | Step-level rewards for long reasoning chains |
| Agentic RL (browser, coding) | RL over tool-use trajectories with execution feedback |

**Hardware reality.** RL training is *throughput-bound by environment simulation*, not just compute. Simulator throughput, vectorized envs, async actor pools, and GPU-resident environments (Brax, IsaacGym, MJX) are first-class engineering. For LLM RLHF, the *generation step* is the bottleneck (decoding 4–8 rollouts per prompt) — vLLM/SGLang inside the RL loop is now standard.

**Failure modes:**
- Reward hacking (specification gaming).
- Catastrophic collapse to a degenerate policy (always-go-left).
- Off-policy correction breakdown when behavior policy drifts too far.
- KL-from-base divergence in RLHF — model finds reward-model adversarial regions.
- Sample inefficiency masking actual progress.

---

### 3.9 Specialized Architectures (GNNs, Science, Geometric)

> The frontier of DL is increasingly in scientific and structured-data applications.

| Area | Key models | Resolves |
|---|---|---|
| **GNNs** | GCN, GraphSAGE, GAT, MPNN, GIN, Graphormer | Graph-structured data (social, molecules, knowledge graphs) |
| **Geometric DL / equivariant nets** | E(3)NN, EquiformerV2, Allegro, DimeNet | 3D physical symmetries; enforces invariances |
| **Protein structure** | AlphaFold 1/2/3, ESMFold, RosettaFold, OpenFold | Sequence → structure |
| **Function/property prediction** | ESM, ProtBERT, ProtT5 | Protein language models |
| **Materials** | M3GNet, MACE, GNoME | Crystal/structure prediction |
| **Weather / climate** | GraphCast, Pangu-Weather, GenCast, Aurora | Global numerical weather prediction |
| **Differential-equation surrogates** | FNO, PINNs, neural operators | Replacing PDE solvers |
| **Robotics policies** | RT-1/2, Octo, π0, OpenVLA | Robotic action prediction |
| **Decision Transformer / Trajectory Transformer** | Sequence modeling for control | RL via supervised learning over trajectories |

**Pros / cons of going specialized:**

| Pros | Cons |
|---|---|
| Real-world impact (drugs, materials, weather) | Domain expertise required |
| Often less crowded than generic-LLM research | Smaller community; less tooling |
| Inductive biases (equivariance) sample-efficient | Inductive bias can become a ceiling at scale |

---

### 3.10 Inference: From Research Checkpoint to Production

> A model that exists only at training-time checkpoints is not yet useful. The transition is its own engineering problem.

This section is the bridge to the AI-Engineer roadmap. See [`../AI-Engineer-Roadmap/README.md` §3.1](../AI-Engineer-Roadmap/README.md#31-inference-serving--the-foundation-everything-else-rides-on) for the full inference-stack treatment. As a DL engineer you should *understand* these even if you don't operate them:

- KV cache, paged attention (vLLM), continuous batching.
- Quantization (INT8, INT4, FP8, AWQ, GPTQ, SmoothQuant).
- Speculative decoding (Medusa, EAGLE, lookahead).
- TP / PP / EP for large-model serving.
- Inference-only kernel optimizations (FlashAttention 3, persistent kernels).
- Compilation: TorchInductor, TensorRT-LLM, XLA, Pallas.
- Serving frameworks: vLLM, SGLang, TensorRT-LLM, TGI, NeMo Inference.

**What's particularly DL-engineer-flavored:**
- Custom kernels in Triton / CUDA / Pallas for novel architectures (linear-attention, MoE routing, novel activations).
- Distillation from a research model to a production-cost model.
- Quantization-aware training and verification of post-training quantization.

---

### 3.11 Evaluation, Benchmarks & Reproducibility

> "The benchmark went up" is the easiest sentence in DL to fool yourself with.

**The real problem.** Open-ended generative outputs and complex tasks resist clean automated metrics. Benchmark contamination, scoring noise, and selection bias mean published numbers regularly don't match in-house reproductions.

**Sub-areas:**

| Area | Tools / examples | Pros | Cons |
|---|---|---|---|
| **Held-out loss / perplexity** | Pretraining-eval loss on clean shards | Cheap; sensitive | Doesn't predict downstream |
| **Capability benchmarks (MMLU, GPQA, MATH, HumanEval, GSM8K)** | LM-Eval Harness, Helm, BIG-bench | Standard | Contamination risk; saturated benchmarks |
| **Long-context evals (Needle-in-a-haystack, RULER, ZeroSCROLLS)** | dedicated suites | Tests one specific axis | Easy to game |
| **Agentic / tool-use evals (SWE-bench, Aider, GAIA)** | Realistic | Hard | Slow; flaky environments |
| **Multimodal evals (MMMU, MathVista, MMBench)** | VLM | Standard | Contamination; image leakage |
| **Pairwise human / LLM-judge (MT-bench, Arena, AlpacaEval)** | preference quality | Open-ended | Judge bias; verbosity bias |
| **Safety / red-team evals** | HarmBench, JBB, in-house red-team suites | Catches worst tail | Adversarial coverage incomplete |
| **Vision benchmarks (ImageNet, COCO, ADE20K, Kinetics)** | Established | Stable | Saturated |
| **Generative quality (FID, CLIP-score, human MOS)** | Image/video gen | Standard | All proxies, none capture preference fully |

**Reproducibility hygiene** — what frontier labs and serious open teams require:
- Pinned random seeds *and* deterministic CUDA paths *and* deterministic data ordering (or documented non-determinism).
- Environment lock files (uv / pip-tools / nix).
- Git SHA + config hash printed in run logs.
- Eval harness versioned and pinned.
- Data lineage: which mix at which step.
- Contamination check against pretraining corpus before claiming a benchmark gain.

**Failure modes:**
- Benchmark contamination — your "huge gain" is leaked test data.
- Eval-hyperparameter sensitivity — different temperature → different leaderboard ranking.
- Scoring noise — a "1% improvement" is below the variance of the eval.
- Selection bias — papers report best-of-N seeds as if it were median.
- Optimizer-state dependence — checkpoint saved mid-warmup evaluates differently.

---

### 3.12 Mechanistic Interpretability & Scaling Laws

> Two areas where DL has stopped being purely empirical and started being *predictive*.

#### Scaling laws

What they say:
- **Kaplan 2020 / Chinchilla 2022** — for compute-optimal training, params and tokens scale together (roughly 20:1 tokens-to-param post-Chinchilla).
- **Post-Chinchilla shifts** — once inference cost matters, *over-train smaller models on more tokens* (Llama-3 8B on 15T tokens). The right point depends on your *deployment* compute, not just training compute.
- **MoE scaling** — sparse capacity scales differently from dense; effective params ≠ active params.
- **μP / μ-Transfer** — hyperparameters can be transferred from small-width sweeps to large-width runs.

**Implication for the DL engineer:** before launching anything serious, you should be able to predict the ballpark loss for your compute budget. If you land far off the scaling-law line, something is wrong — usually data, init, or LR schedule.

#### Mechanistic interpretability

Sub-areas:

| Area | Key idea |
|---|---|
| **Circuits** | Identify subnetworks computing specific functions |
| **Probing** | Linear classifiers over activations to test "does the model represent X?" |
| **Activation patching / causal scrubbing** | Causally test which components drive a behavior |
| **Sparse autoencoders (SAEs)** | Decompose activations into interpretable monosemantic features |
| **Feature steering** | Edit activations to control behavior |
| **Logit-lens / tuned-lens** | Project intermediate residual stream to the vocabulary |
| **Attribution patching** | Cheap approximation of activation patching at scale |

**Why this matters at frontier labs.** Anthropic's interpretability team, DeepMind's Mechanistic Interp team, and OpenAI's Superalignment work all view this as the path to *understanding* what frontier models compute — which is, ultimately, the safety story. It's also some of the cleanest research-engineer work in DL: tight loop, clear hypotheses, real causal claims.

---

## Part 4 — Reading & Reproducing Research (the core craft)

> If you can read a paper on Monday and have a working reimplementation by Friday with numbers within reasonable noise, you are 90% of the way to a frontier-lab research engineer offer.

**The reading protocol** (compress 80 pages of papers per week):

1. **Title + abstract + figures only.** Decide: skip / skim / deep-read.
2. **Skim:** intro, method figure, main result table, conclusion. ~10 min/paper.
3. **Deep-read:** every equation, every ablation, every hyperparameter. ~2–4 hours.
4. **Adversarial read:** what would make this result fake? Cherry-picked seed? Contaminated eval? Unfair baseline? Missing wall-clock comparison?
5. **Reimplement:** fork a clean baseline, apply the paper's idea, run the smallest reproducible config.
6. **Write up:** one-page summary — what they did, what worked, what didn't reproduce, what surprised you.

**Papers worth deep-reading first** (before specializing):

| Era | Paper |
|---|---|
| 2014 | Adam, GANs, Seq2Seq |
| 2015 | ResNet, Batch Norm, Attention (Bahdanau) |
| 2017 | **Attention Is All You Need** |
| 2018–19 | BERT, GPT-2, T5 |
| 2020 | GPT-3, Scaling Laws (Kaplan), CLIP, ViT |
| 2021 | DDPM, MAE, FlashAttention v1, LoRA |
| 2022 | Chinchilla, InstructGPT, PaLM, Stable Diffusion |
| 2023 | LLaMA, Llama-2, GPT-4 tech report, RWKV, FlashAttention 2, DPO |
| 2024 | Llama-3, Mixtral, Gemini, Claude 3, DeepSeek-V2 (MLA), DPO variants, FlashAttention 3 |
| 2025 | DeepSeek-V3 + R1, Llama-4-class, FP8/FP4 training papers, frontier multimodal |

Read the **technical reports** (Llama-3, DeepSeek-V3, OLMo, Gemma, Qwen) — they're the most information-dense documents in the field, and the ones interviewers most often probe.

**Reimplementation projects (in order of difficulty):**

| Level | Project | Time | What it teaches |
|---|---|---|---|
| 1 | nanoGPT from scratch | 1–2 weeks | Transformer fundamentals |
| 2 | KV cache + speculative decoding on top of it | 1 week | Inference dynamics |
| 3 | LoRA fine-tuning on Llama-3-8B with PEFT | 1 week | Modern adapter tooling |
| 4 | DPO from scratch on a 1B model with TRL | 1–2 weeks | Preference optimization |
| 5 | A small MoE from scratch (8 experts, 2 active) | 2–3 weeks | Routing, load balance, EP |
| 6 | FlashAttention forward in Triton | 1–2 weeks | Kernel programming |
| 7 | Reproduce Chinchilla scaling-law plot at small scale | 2–3 weeks | Experimental discipline |
| 8 | Reproduce a NeurIPS paper of your choice | 4–8 weeks | The end-to-end skill |

---

## Part 5 — Foundations to Backfill (Just-in-Time)

Don't read these front-to-back. Pull each in when a problem in Part 3 forces you to.

### 5.1 Math (the parts that actually bite)

- **Linear algebra:** vector spaces, projections, rank, SVD/eigendecomposition (you'll see these in PCA, attention, low-rank adapters, μP).
- **Calculus:** chain rule (backprop), Jacobians, Hessians (preconditioned optimizers).
- **Probability:** random variables, MLE/MAP, KL divergence, entropy, mutual information.
- **Optimization:** convex vs non-convex, gradient descent dynamics, momentum, second-order methods (LBFGS, Shampoo), constrained optimization.
- **Information theory:** entropy, KL, cross-entropy, Fano's inequality (interp work uses it).
- **Numerical analysis:** condition number, stability, IEEE 754.
- **Statistics:** hypothesis testing, confidence intervals, bootstrapping.

Recommended texts: *Mathematics for Machine Learning* (Deisenroth et al.), Strang *Linear Algebra*, MacKay *Information Theory*.

### 5.2 Classical ML

- Linear / logistic regression — derive the loss and gradient on a whiteboard.
- SVM (margin, kernel trick).
- Decision trees / RF / GBDT (XGBoost, LightGBM).
- KNN, K-means, GMM.
- **Bias–variance tradeoff** — the most important conceptual tool you have.
- Regularization (L1, L2, dropout, early stopping).
- Cross-validation, calibration, ROC/PR curves.

Local references: [`../../ML-Implementations/basics/`](../../ML-Implementations/basics/), [`../../ML-Implementations/optimizers/`](../../ML-Implementations/optimizers/).

### 5.3 Neural network fundamentals

- Backpropagation by hand (one matrix-multiplication NN, then one Transformer block).
- Activations: sigmoid → ReLU → GELU → SiLU/SwiGLU. Why each.
- Initialization: Xavier, He, μP-aware.
- Normalization: BN, LN, RMSNorm, GroupNorm.
- Why pre-norm > post-norm at depth.
- Why residual connections are *the* idea of the 2010s.

### 5.4 Transformer internals

- Self-attention math; QK^T/√d intuition.
- Multi-head; per-head dim; head merging.
- MQA / GQA / MLA — what each compresses, what each costs.
- FFN block; SwiGLU/GeGLU.
- Causal mask, padding mask, document mask (block-diagonal).
- Position: sinusoidal → learned → RoPE → ALiBi → YaRN/NTK-aware.

Local references: [`../../ML-Implementations/transformers/`](../../ML-Implementations/transformers/).

### 5.5 Programming & tooling

- Python: idiomatic, NumPy, PyTorch, JAX, type hints.
- Linux / shell / git proficiency.
- CUDA / Triton fundamentals (read FlashAttention; write a softmax kernel).
- JAX `jit` / `vmap` / `pjit` / `shard_map`.
- Profiling: `torch.profiler`, NVIDIA Nsight, JAX `jax.profiler`.
- Docker / Slurm / Kubernetes for cluster jobs.
- Weights & Biases / Aim / TensorBoard for experiment tracking.

### 5.6 Systems

- File systems and storage I/O for training data.
- NCCL collectives; how all-reduce ring topology works.
- InfiniBand / RoCE; rail-optimized topology.
- Checkpointing strategies (sync vs async, sharded vs flat).
- Fault recovery, preemption handling.

---

## Part 6 — Interview Signal at Frontier Labs

What DeepMind / Anthropic / OpenAI / FAIR / Mistral / DeepSeek actually probe — patterns across many interviews.

### Coding (universal first round)

- Implement attention forward and backward on a whiteboard.
- Implement multi-head attention with masking from scratch.
- Beam search / top-k / top-p sampling.
- KV cache.
- LoRA forward and merge.
- Softmax with numerical stability.
- Pytorch / JAX shape gymnastics: "why does this `.view()` fail?" "What's the difference between `gather` and `index_select`?"

### Hardware / numerical reasoning

- "Why is decode bandwidth-bound and prefill compute-bound?"
- "How much VRAM to train a 7B with FSDP, BF16+FP32 master, Adam, batch=1M tokens?"
- "Where would you put TP vs PP given an 8×H100 node + 32-node cluster?"
- "Why does FP16 attention overflow at long context, and how do you fix it?"
- "What is MFU and what's a good number?"

### Architectural / training-dynamics depth

- "Walk me through pre-norm vs post-norm and why pre-norm survives at 100+ layers."
- "Why GQA? Why MLA? What's the KV-cache math?"
- "Why does the loss spike at step 50K? List five hypotheses ranked by likelihood."
- "What's μP and why do labs use it?"
- "Explain mode collapse in a GAN. Now in an RL policy. Now in MoE routing."

### Research-engineer judgment

- "You ran an ablation showing +0.5% on a benchmark. How do you know it's real?"
- "Two implementations of the same paper give different numbers. Where would you look first?"
- "Your loss is descending too fast — would you celebrate or worry?"
- "I give you 64 H100s for a week. Pick one research question to answer."

### Paper deep-dive

You'll be asked to pick a paper you know well and the interviewer will probe:
- Why the method works (mechanism, not just intuition).
- Where the experiments are weak.
- How it would scale — or break.
- Whether you reproduced it.

---

## Part 7 — Suggested Project Track

> Pick three. Real reimplementation > a long reading list.

### Project 1 — Train a Transformer LM from scratch

Reproduce nanoGPT-class results, then go beyond:
- Train 50M-param GPT on TinyStories or OpenWebText shard.
- Plot loss vs steps; verify it follows a power law over 1+ decade.
- Add **RoPE**, **GQA**, **SwiGLU**, **RMSNorm**, **gradient clipping**, **mixed precision**.
- Implement **KV cache** for inference and benchmark vs no-cache.
- Document every architectural change with an A/B ablation.

### Project 2 — Distributed training mini-frontier

- Train a 1–3B model on a 4-GPU node with FSDP (PyTorch).
- Add **activation checkpointing** and measure memory savings.
- Add **TP=2** and compare throughput vs FSDP-only.
- Add **gradient accumulation** to simulate large batch size; verify loss curves.
- (Optional) port the same model to **JAX with `pjit`** and reproduce.
- Report MFU vs theoretical peak; identify the bottleneck.

### Project 3 — Custom kernel + numerical-stability case study

- Implement a fused softmax kernel in **Triton**.
- Implement **FlashAttention-style** forward in Triton at small head_dim.
- Side-by-side benchmark vs PyTorch SDPA.
- Add a pathological long-sequence test that breaks naive FP16 softmax; show your kernel handles it.

### Project 4 — Reproduce a research result end-to-end

Pick a recent NeurIPS / ICML / ICLR paper with public code:
- Read the paper (deep + adversarial).
- Run their code; verify reported numbers.
- Implement their key contribution into your own training stack from scratch.
- Run your reimplementation; measure the gap.
- Write a 1–2-page reproducibility report.

### Project 5 — RL on a real task

- Solve CartPole + LunarLander with **PPO from scratch** (not just the cleanrl version).
- Move to MuJoCo continuous control with **SAC**.
- (Optional) implement a tiny **Decision Transformer** on D4RL.
- (Stretch) implement **DPO from scratch** on a 1B-parameter base model.

### Project 6 — Mechanistic interpretability mini-project

- Pick a small open model (Pythia-1B, GPT-2 small, Llama-3-8B).
- Find and verify a **circuit** (e.g. indirect object identification, modular addition).
- Write up the activation-patching plot.
- (Stretch) train a small **sparse autoencoder** on a layer's residual stream.

---

## Part 8 — DeepMind-Style Research Questions

These three problems (preserved and translated from [`README_cn.md`](./README_cn.md)) are designed to test research-engineer judgment, not memorization. Treat each as a 30–60-minute open-ended discussion.

### Question 1 (core recommendation): Causal Representation Learning over Multi-Sequence User Behavior

**Setting (DeepMind-style):** You have multiple user behavior sequences (app installs, purchases, views, clicks). Goal: predict whether a conversion will happen.

**Real-world challenges:**
- Sequence lengths are inconsistent.
- Different behaviors causally contribute to conversion to different degrees.
- Naive attention learns *correlation*, not *causation*.

**Task:** design a model that:
- Uses multiple event sequences.
- Learns representations of features that *causally* contribute to conversion.
- Stays robust under spurious correlations.

**Probing questions:**

*Q1: Model design.* How would you model multiple sequences? Self-attention vs cross-attention — when does each apply? Should embeddings be shared?

*Q2: Causal challenges.* Which behaviors are correlated but not causal? How would you design model or training objective to reduce spurious correlation?

*Q3: Experiment design.* How do you verify the model learned causation, not correlation? If offline metrics improve but online metrics drop, how do you explain it?

**Bonus (very DeepMind):**
- Counterfactual masking.
- Temporal interventions (shuffle time order).
- Auxiliary losses (e.g. predict masked event).

**What's being assessed:** representation learning, causal-inference intuition, attention understanding, experiment design.

### Question 2 (RL track): Conversion Modeling under Long-Delay Reward

**Setting:** Conversion may occur hours or days *after* the user behavior sequence ends.

**Limit of supervised learning:** only predicts whether conversion happened; ignores the temporal structure between behavior and reward.

**Task:** reframe conversion prediction as an RL problem.

**Probing questions:**

*Q1:* How are state, action, reward defined?
*Q2:* How do you handle sparse reward?
*Q3:* Why might Policy Gradient fit better than Q-learning here?
*Q4:* Could you use model-based RL?
*Q5:* How do you stabilize training?
*Q6:* How do you evaluate the learned policy?

**Bonus:** TD(λ) credit assignment; hindsight replay; offline RL.

**What's being assessed:** RL modeling skill, ability to abstract real-world problems, theory ↔ engineering bridging.

### Question 3 (foundational research): Does Attention Really Need Softmax?

**Setting:** The Transformer's core is `Attention(Q, K, V) = softmax(QKᵀ / √d) · V`. But:
- Softmax brings numerical instability (overflow at long context).
- Softmax distributions can be over-smooth.

**Task:**
- Is there an attention mechanism without softmax?
- If so, what are its advantages?

**What you should do:**
- Propose an alternative (e.g. linear attention, kernel-based, RetNet-style retention).
- Analyze theoretical properties (normalization, stability, expressivity).
- Validate empirically.

**Bonus:**
- Complexity analysis O(N²) → O(N).
- Long-sequence experiments.
- Compare convergence speed.

**What's being assessed:** mathematical intuition, algorithmic creativity, theory + experiment integration.

---

## References

### Books worth reading cover-to-cover
- *Deep Learning* — Goodfellow, Bengio, Courville. The grounding text.
- *Mathematics for Machine Learning* — Deisenroth, Faisal, Ong. Math you'll actually use.
- *Reinforcement Learning: An Introduction* — Sutton & Barto. The RL bible.
- *Pattern Recognition and Machine Learning* — Bishop. The Bayesian view.
- *Information Theory, Inference, and Learning Algorithms* — MacKay. Free PDF; the most thoughtful book on the field.

### Courses
- **Stanford CS231n** — Vision (still the best CV-from-scratch course).
- **Stanford CS224N** — NLP with Deep Learning.
- **Stanford CS336** — Language Modeling from Scratch (the most current production-relevant DL course).
- **Stanford CS229** — Classical ML.
- **Stanford CS236** — Deep Generative Models.
- **MIT 6.S191** — Intro to DL.
- **Andrej Karpathy: Neural Networks — Zero to Hero** (free YouTube series).
- **Spinning Up in Deep RL** (OpenAI, free).

### Practical / industrial references
- **Hugging Face Ultra-Scale Playbook** — single best free reference on parallelism.
- **Llama 3 paper** (Meta, 2024) — frontier-training engineering details.
- **DeepSeek-V3 / R1 technical reports** — MLA, FP8 training, RLVR.
- **OLMo / OLMo 2** (AI2) — fully open frontier-class training writeups.
- **NVIDIA Megatron-LM** repo + papers.
- **Karpathy's nanoGPT and llm.c** — readable canonical references.
- **JAX Scaling Book** (`jax-ml.github.io/scaling-book`) — the systems counterpart of Ultra-Scale Playbook.

### Foundational papers (read when relevant in Part 3)
- Attention Is All You Need (2017).
- BERT, GPT-2, GPT-3.
- Scaling Laws (Kaplan 2020), Chinchilla (Hoffmann 2022).
- FlashAttention 1/2/3.
- LoRA, QLoRA.
- DPO; ORPO; SimPO.
- DDPM; Latent Diffusion; Rectified Flow.
- AlphaGo / AlphaZero / MuZero.
- AlphaFold 2 / 3.
- DINO / DINOv2; SAM / SAM 2.
- Mixtral; DeepSeek-V3 (MLA, MoE, FP8); Llama-3.
- MAE; CLIP; SigLIP.
- Constitutional AI; InstructGPT.
- Mechanistic interpretability: "Toy Models of Superposition" (Anthropic), "Scaling Monosemanticity", "Tracing Thoughts in LMs."

### Career / research-engineer-path writing
- *How I got a job at DeepMind as a research engineer without an ML degree* — Aleksa Gordić (Medium).
- *Deep Learning Journey Update: What Have I Learned About Transformers and NLP in 2 Months?* — Aleksa Gordić.
- Lilian Weng's blog (`lilianweng.github.io`) — an evolving free DL textbook.

### Reference implementations in this repo
- [`ML-Implementations/basics/`](../../ML-Implementations/basics/) — linear / logistic / softmax classifiers.
- [`ML-Implementations/optimizers/`](../../ML-Implementations/optimizers/) — SGD, Adam, Adagrad.
- [`ML-Implementations/transformers/`](../../ML-Implementations/transformers/) — self-attention, multi-head attention, transformer block.
- [`ML-Implementations/ads/`](../../ML-Implementations/ads/) — applied DL architectures (DCN, DIN, DeepFM, MMoE, PLE, ESMM).
