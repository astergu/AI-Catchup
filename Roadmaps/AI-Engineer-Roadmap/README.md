# AI / LLM Engineer Roadmap — Top-Down

Most roadmaps start with "learn self-attention" and crawl upward to RAG. This one starts where the job actually lives: **what industry ships, what business KPIs it's graded on, what sub-systems each product decomposes into, and what hardware constraints force every architectural decision**. Theory is backfill — pulled in only when a real problem demands it.

---

## Table of Contents

- [Part 0 — How to Read This Roadmap](#part-0--how-to-read-this-roadmap)
- [Part 1 — What AI Engineers Are Actually Hired To Do](#part-1--what-ai-engineers-are-actually-hired-to-do)
- [Part 2 — The Hardware Constraints Behind Every Design Decision](#part-2--the-hardware-constraints-behind-every-design-decision)
- [Part 3 — Industrial Solutions, Decomposed](#part-3--industrial-solutions-decomposed)
  - [3.1 Inference Serving](#31-inference-serving--the-foundation-everything-else-rides-on)
  - [3.2 Chat Products & Alignment](#32-chat-products--alignment)
  - [3.3 RAG (Retrieval-Augmented Generation)](#33-rag-retrieval-augmented-generation)
  - [3.4 Agents & Tool Use](#34-agents--tool-use)
  - [3.5 Domain Adaptation (Fine-Tuning, PEFT, Distillation)](#35-domain-adaptation-fine-tuning-peft-distillation)
  - [3.6 Evaluation & Production Reliability](#36-evaluation--production-reliability)
  - [3.7 Foundation Model Training (What to Understand, Not Build)](#37-foundation-model-training-what-to-understand-not-build)
  - [3.8 Multimodal](#38-multimodal)
- [Part 4 — Picking a Model: 2024–2025 Landscape](#part-4--picking-a-model-20242025-landscape)
- [Part 5 — Foundations to Backfill (Just-in-Time)](#part-5--foundations-to-backfill-just-in-time)
- [Part 6 — Interview Signal: What Top AI Teams Actually Probe](#part-6--interview-signal-what-top-ai-teams-actually-probe)
- [Part 7 — Suggested Project Track](#part-7--suggested-project-track)
- [References](#references)

---

## Part 0 — How to Read This Roadmap

For each industrial solution, the structure is:

1. **The real problem** — what business KPI is being moved, and why naive approaches fail.
2. **Sub-solutions** — the components a production system decomposes into.
3. **What each sub-solution resolves** — and what new problem it introduces.
4. **Pros / cons / when it's the wrong tool**.
5. **Hardware footprint** — VRAM, latency, throughput, $/request.
6. **Failure modes you'll see in production**.

Theory (transformer math, attention variants, scaling laws) sits in Part 5. Pull it in when a problem in Part 3 forces you to.

---

## Part 1 — What AI Engineers Are Actually Hired To Do

The job titles vary (AI Engineer, ML Engineer–GenAI, LLM Engineer, Applied Scientist) but the **product archetypes are surprisingly small**:

| Archetype | Concrete examples | Primary business KPI | Primary technical KPI |
|---|---|---|---|
| **Conversational assistant** | ChatGPT, Claude.ai, customer-support bots | DAU, retention, deflection rate | Helpfulness, refusal rate, P95 latency |
| **Knowledge / Q&A over private data (RAG)** | Glean, Perplexity Enterprise, doc-search copilots | Answer adoption, time-to-answer | Recall@k, factuality, hallucination rate |
| **Code assistant** | Copilot, Cursor, Claude Code | Acceptance rate, edit distance | Completion latency, correctness |
| **Agentic workflows** | Operator, Devin, browser/computer agents | Task completion rate | Steps-to-success, cost-per-task |
| **Content generation** | Marketing copy, ad creative, image/video gen | Conversion lift, content volume | Quality score, brand-safety pass rate |
| **Classifier / extractor** | Fraud, moderation, structured extraction | Precision/recall on the business-critical class | Throughput, $/1M docs |
| **Foundation model training** | Frontier labs only | Benchmark score, eval suite | MFU, time-to-checkpoint, training stability |

### What you're actually graded on

Three numbers control almost every design decision:

1. **Quality** — task-specific eval score (factuality, helpfulness, win-rate vs baseline).
2. **Cost** — $/request or $/1M tokens (driven by model size, context length, caching, batching).
3. **Latency** — P50 / P95 / P99 (time-to-first-token *and* time-to-completion are different metrics).

A senior AI engineer is someone who can **trade these three off explicitly** and explain *why* they made the trade — not someone who knows attention math best.

### The cost/quality/latency triangle

```
        Quality
          /\
         /  \
        /    \
       / pick \
      /  two   \
     /__________\
   Cost      Latency
```

Real shipping examples of where the trade goes:
- **Cheap + fast → quality drop**: Mistral 7B INT4 on a single L40S. Acceptable for classification, weak for reasoning.
- **Quality + fast → cost up**: GPT-4o on every request. Easy quality, $10K/day at 1M requests.
- **Quality + cheap → slow**: a re-ranked RAG pipeline with Claude Sonnet for synthesis. Acceptable batch latency, not great real-time.

Everything in Part 3 is a tool for moving along one axis at the cost of another.

---

## Part 2 — The Hardware Constraints Behind Every Design Decision

If you skip this part you'll forever be confused about *why* the industry made certain choices. **Almost every "trick" in inference and training exists to dodge a specific hardware bottleneck.**

### 2.1 GPU memory: where every byte goes

For a Transformer in inference, VRAM splits roughly as:

```
VRAM = model_weights + KV_cache + activations + framework_overhead
```

**Model weights** — `params × bytes_per_param`:

| Precision | Bytes/param | 7B model | 70B model | 405B model |
|---|---|---|---|---|
| FP32 | 4 | 28 GB | 280 GB | 1.6 TB |
| FP16/BF16 | 2 | 14 GB | 140 GB | 810 GB |
| INT8 | 1 | 7 GB | 70 GB | 405 GB |
| INT4 | 0.5 | 3.5 GB | 35 GB | 203 GB |

So *just to load weights*: 70B at BF16 doesn't fit on a single H100 (80GB). It does fit on an H200 (141GB) — and that fact alone reshapes deployment economics.

**KV cache** — the silent killer:

```
KV_cache_per_token = 2 × num_layers × num_kv_heads × head_dim × bytes_per_param
```

Worked example, LLaMA 3 70B at BF16: 80 layers × 8 KV heads (GQA) × 128 dim × 2 bytes × 2 (K and V) ≈ **320 KB per token**. A 32K-context request alone consumes ~10 GB of KV cache — sometimes more than the activations. This is *why* you can't naively batch long-context requests, *why* PagedAttention exists, *why* MQA/GQA were invented, *why* MLA (Multi-head Latent Attention in DeepSeek-V3) is a big deal.

**Activations** — fluctuate during forward/backward; smaller during inference, dominant during training (recomputation/checkpointing exists to control this).

### 2.2 The two regimes: compute-bound vs memory-bandwidth-bound

LLM inference has two distinct phases:

| Phase | What happens | Bottleneck | What helps |
|---|---|---|---|
| **Prefill** | Process the prompt in parallel | Compute (FLOPs) | Tensor parallelism, bigger batches, FP8 |
| **Decode** | Generate one token at a time | **Memory bandwidth** (must reload weights from HBM every token) | Quantization, speculative decoding, batching, MQA/GQA |

A single decode step is a tiny matmul that re-reads the entire model from HBM — so token/sec is gated by `HBM_bandwidth / model_size`. This is why H100 (3 TB/s HBM) and B200 (8 TB/s HBM) shifted the economics so much. It's also why **decode is what most inference optimizations target** — prefill is "easy."

Mental shortcut: roofline model. Plot arithmetic intensity (FLOPs per byte loaded). Below the ridge → memory-bound (you waste compute). Above the ridge → compute-bound (you waste bandwidth). Decode lives far below the ridge.

### 2.3 Networking: the part nobody mentions until you scale

| Link | Bandwidth | Where it shows up |
|---|---|---|
| Within one GPU (HBM ↔ SMs) | 3–8 TB/s | Sets decode token/sec |
| NVLink (intra-node, ≤8 GPUs) | 600–900 GB/s | Tensor parallelism within a node |
| PCIe Gen5 | ~64 GB/s | Bottleneck if you do TP across PCIe — avoid |
| InfiniBand NDR (inter-node) | 400 Gb/s ≈ 50 GB/s | Pipeline parallelism / data parallelism across nodes |

The rule: **tensor parallelism stays inside an NVLink island** (≤8 GPUs); **pipeline / data parallelism crosses nodes**. Get this backwards and your MFU collapses.

### 2.4 Cost reality (rough 2024–2025 numbers)

| Resource | Typical price | What you can do with it |
|---|---|---|
| H100 SXM, on-demand | $2–4/hr | Serve a 70B FP8 with vLLM at ~30 tok/s/user |
| 8×H100 node | $20–30/hr | Run Llama-3 405B FP8 inference; train a 7B with FSDP |
| GPT-4o API | ~$5 / 1M input, $15 / 1M output | Fast PoC; expensive at production scale |
| Claude Sonnet API | ~$3 / 1M input, $15 / 1M output | Long-context workloads |
| Open-weights self-host | $0.10–0.50 / 1M tokens (amortized) | Break-even vs API around 100M tokens/month |

**Heuristic:** prototype on an API, profile token volume, switch to self-host when monthly bill exceeds ~$10K. Below that, engineering time costs more than the API.

### 2.5 GPU pick guide

| GPU | VRAM | HBM BW | Best for | Worst for |
|---|---|---|---|---|
| H100 80GB | 80 GB | 3.35 TB/s | Frontier serving + training | 70B+ at full precision |
| H200 | 141 GB | 4.8 TB/s | 70B BF16 single-GPU; long context | Newer, less available |
| B200 | 192 GB | 8 TB/s | Frontier training, FP8/FP4 inference | Hard to get, expensive |
| A100 80GB | 80 GB | 2 TB/s | Cost-effective training/serving | New formats (no FP8) |
| L40S | 48 GB | 0.86 TB/s | Cheap inference of small/quantized models | Long context, large models |
| MI300X | 192 GB | 5.3 TB/s | Single-GPU 70B+; ROCm ecosystem maturing | CUDA-only software |
| RTX 4090 / 5090 | 24 / 32 GB | 1+ TB/s | Local dev, QLoRA fine-tuning of ≤13B | Datacenter SLAs |

---

## Part 3 — Industrial Solutions, Decomposed

### 3.1 Inference Serving — the foundation everything else rides on

> Every product in this section ultimately calls `model.generate()`. How efficiently you serve determines whether your product economically exists.

**The real problem.** Autoregressive decoding generates one token at a time, each step re-reading the entire model from HBM. Naive serving wastes 80–95% of GPU time. A poorly served 70B model can cost 10× what a well-served one does for identical quality.

**Sub-solutions:**

| Sub-solution | What it resolves | New problem it creates | Pros | Cons |
|---|---|---|---|---|
| **KV cache** | Avoids recomputing past tokens' K/V at every step | Memory grows with context length | ~10–100× faster decode | Becomes the dominant memory cost at long context |
| **PagedAttention (vLLM)** | KV cache fragmentation across variable-length requests | Adds bookkeeping complexity | 2–4× higher batch size at same VRAM | Tightly coupled to vLLM internals |
| **Continuous batching** | Idle GPU time when one request finishes mid-batch | Scheduler complexity | 5–20× throughput vs static batching | Latency tail-latency variance |
| **Speculative decoding (Medusa, EAGLE, lookahead)** | Decode is bandwidth-bound, not compute-bound | Needs a draft model, target/draft alignment | 2–3× speedup with no quality loss | Acceptance rate is workload-dependent |
| **Quantization (INT8 / INT4 / FP8 / AWQ / GPTQ)** | Memory + bandwidth pressure | Quality regression risk; some kernels missing | 2–4× more throughput, smaller GPUs | Calibration data matters; long-context degradation possible |
| **Tensor parallelism (TP)** | Single-GPU memory ceiling for large models | Requires NVLink; communication every layer | Lets a 70B+ run across 4–8 GPUs | Diminishing returns; latency adds for collective ops |
| **Pipeline parallelism (PP)** | Even larger models across multiple nodes | Pipeline bubbles, scheduling complexity | Scales to 100B+ across nodes | High latency unless prompts are huge |
| **Expert parallelism (EP, for MoE)** | MoE expert weights don't fit per-GPU | Routing and load imbalance | Required for Mixtral / DeepSeek-V3 / GPT-4-class MoE | All-to-all communication is brutal across nodes |
| **Disaggregated prefill/decode** | Prefill (compute-bound) and decode (bandwidth-bound) interfere when colocated | Two separate clusters to operate | Better SLO predictability; cheaper hardware mix | Operational complexity |
| **Prefix / prompt caching** | Repeated system prompts re-prefill on every call | Cache invalidation, memory budget | Massive savings for chat / agent workloads | Limited if prompts are unique |
| **Streaming responses** | TTFT >> TTFCT (first vs final token) hurts UX | Fragmented client handling | UX is responsive; user starts reading at TTFT | Doesn't change total cost |

**Frameworks (industry standard 2024–2025):** vLLM (most common), SGLang (best at structured/agent workloads), TensorRT-LLM (NVIDIA-optimized, best raw perf), Hugging Face TGI (simpler, slipping in performance), Triton + custom kernels (frontier labs).

**Decision tree (paraphrased from real on-call):**
- Single small model, low traffic → **TGI or Ollama**
- Production chat at scale → **vLLM with continuous batching + prefix caching**
- Heavy agent / structured output workload → **SGLang** (RadixAttention, faster constrained decoding)
- Squeezing every last ms on dedicated H100s → **TensorRT-LLM**
- Want to serve a 70B on one consumer GPU → **llama.cpp / ExLlamaV2 with INT4**

**Hardware footprint examples (production reference points):**
- Llama-3-8B BF16 on 1× L40S: ~30 concurrent users at 40 tok/s each.
- Llama-3-70B FP8 on 1× H200: ~50 concurrent users at 30 tok/s each.
- Llama-3-70B BF16 with TP=4 on 4× H100: ~150 concurrent users at 40 tok/s each.

**Production failure modes to know:**
- OOM at peak load because someone sent a 32K-context request — KV cache exploded.
- P99 latency spikes from "noisy neighbor" prefill blocking decode.
- Quantization quality cliff at long context (calibration didn't include long inputs).
- vLLM `max_num_seqs` set too high → throughput up, P99 destroyed.

---

### 3.2 Chat Products & Alignment

> The base pretrained model is a next-token completer. A "chatbot" is built by *aligning* it.

**The real problem.** Pretrained LLMs autocomplete text — they don't follow instructions, refuse harmful requests, stay on persona, or reliably output JSON. Alignment is the layer between "raw model" and "shippable product."

**Sub-solutions:**

| Sub-solution | What it resolves | Cost | Pros | Cons |
|---|---|---|---|---|
| **System prompts** | Teach format, role, constraints at inference | Zero training cost | Fastest iteration loop | Easily jailbroken; eats context window |
| **Few-shot in prompt** | Demonstrate output format | Per-call token cost | No training | Not robust; brittle to phrasing |
| **SFT (supervised fine-tuning)** | Teach instruction-following + format from labeled examples | 100–10K GPU-hours for 7B–70B | Strong baseline; predictable cost | Requires curated data; can overfit to style |
| **RLHF (PPO)** | Optimize for *preferences* not just imitation | 10×+ SFT cost; unstable | Highest quality alignment historically | Reward hacking; engineering complexity |
| **DPO / IPO / KTO / ORPO / SimPO** | Preference tuning without a reward model | ~SFT cost | Stable, reproducible, simple | Slightly weaker than well-tuned PPO at scale |
| **Constitutional AI / RLAIF** | Generates preferences with a model — scales beyond human labelers | SFT-class cost + lots of inference | Scales; reduces human labeling | Bias of judge model leaks in |
| **Distillation from a stronger model** | Cheap quality lift for smaller models | Inference cost on teacher | Often Pareto-best for ≤13B | Legal/TOS issues with closed APIs |
| **Guardrails (input filters, output filters, classifier sandwich)** | Hard constraints (no PII, no policy violations) | Adds latency | Defense in depth | False positives; users find workarounds |
| **Tool use / function calling** | Grounding in verifiable APIs | Per-call cost | Reduces hallucination on factual tasks | Latency, error handling complexity |

**The progression most teams actually follow:** prompt → system prompt + few-shot → SFT on a small high-quality set → DPO on preference pairs → guardrails on top. RLHF (PPO) is rare outside frontier labs because of cost and instability; DPO ate its lunch.

**Hardware reality.** Full SFT of a 7B at BF16 needs ~80GB just for weights+grads+optimizer states (parameters × ~16 bytes for Adam). Use ZeRO-3 / FSDP / DeepSpeed to shard. QLoRA collapses this to 24GB. Full SFT of a 70B → multi-node, days of compute.

**Failure modes:**
- **Alignment tax** — aligned model is dumber than the base on raw benchmarks.
- **Sycophancy** — model agrees with user even when wrong (preference data bias).
- **Reward hacking** — RLHF model gives *plausible-looking* answers, not correct ones.
- **Refusal creep** — model refuses harmless requests after over-aggressive safety tuning.
- **Format collapse** — SFT on a narrow format destroys general capability.

---

### 3.3 RAG (Retrieval-Augmented Generation)

> "The model doesn't know your data" is the #1 enterprise blocker. RAG is how the industry currently solves it.

**The real problem.** Foundation models are frozen at training time, can't see private data, and confidently hallucinate when they don't know. Naive solutions (fine-tuning every doc change, stuffing everything into context) don't scale economically.

**Sub-solutions and what each resolves:**

| Component | What it resolves | Pros | Cons |
|---|---|---|---|
| **Chunking** (fixed, semantic, hierarchical, propositions) | Documents are too long for context | Cheap; pre-computable | Boundary loss; hard to capture cross-chunk reasoning |
| **Embedding model** (BGE, E5, GTE, OpenAI, Voyage, Cohere) | Semantic similarity beats keywords | Generalizes across phrasing | Domain shift hurts; benchmark ≠ your data |
| **Vector database** (FAISS, Milvus, Qdrant, pgvector, Weaviate, Pinecone) | ANN search at scale (>1M vectors) | Sub-ms p50 lookup | Memory-heavy; index choice (HNSW/IVF/ScaNN) is a real decision |
| **Hybrid search** (BM25 + dense fusion) | Pure dense fails on rare keywords, IDs, code | Big recall gains for ~free | Two systems to operate |
| **Query rewriting / multi-query / HyDE** | User queries don't match document phrasing | Recall boost on conversational queries | Adds an LLM call (latency, cost) |
| **Reranker** (cross-encoder, ColBERT, Cohere rerank) | Top-k from ANN is noisy | Massive precision gain | Adds 50–200ms; doesn't help if recall already missed it |
| **Metadata filtering** | Tenant isolation, time/region scoping | Hard correctness for permissions | Index complexity |
| **Long-context model + no retrieval** | Skip the pipeline entirely if doc fits | Simpler; no chunking artifacts | Cost scales linearly with context; "lost in the middle" effect |
| **Cache-augmented generation (CAG)** | Repeated retrieval of same docs | Eliminates retrieval at inference | Only works for small static corpora |
| **Graph RAG / structured retrieval** | Multi-hop, relationship-heavy data | Beats flat retrieval on reasoning queries | Extraction pipeline is fragile |

**Vector DB index choice** (this is a real interview question):

| Index | Memory | Query time | Build time | When |
|---|---|---|---|---|
| **Flat (brute force)** | low | O(N) — slow | none | <100K vectors |
| **HNSW** | high (graph) | very fast | slow | Most production cases ≤100M |
| **IVF + PQ** | low (compressed) | fast, approximate | medium | Billion-scale, tight memory |
| **ScaNN** | medium | fast | medium | Google ecosystem |

**Hardware footprint:** raw vectors at 1024-dim float32 = 4KB each. 10M docs × 4KB = 40GB just for vectors before any index overhead. HNSW typically adds another ~1.5×. **At ~10M+ docs, vector DB sizing dominates infra cost** — at which point IVF+PQ or quantized embeddings (binary, int8) start to look interesting.

**Decision: RAG vs long-context vs fine-tuning?**

| Need | Best tool |
|---|---|
| Knowledge changes daily | **RAG** |
| Knowledge fits in 100K tokens, infrequent queries | **Long context** |
| Format/style/domain (knowledge static) | **Fine-tuning** |
| Strict factual grounding required | **RAG** (with citations) |
| Lowest possible latency | **Fine-tuning** (knowledge baked in) |
| Permission/tenant boundaries | **RAG** with metadata filters |

**Production failure modes:**
- Chunk boundary cuts a critical sentence; reranker can't rescue it.
- Embedding model trained on web data fails on domain jargon (medical, legal codes).
- "Lost in the middle" — relevant chunk is in context but ignored.
- Conversational queries ("what about that other one?") tank retrieval (need query rewriting).
- Stale index — docs updated, embeddings not rebuilt.

---

### 3.4 Agents & Tool Use

> The LLM stops being a text-completer and starts being a *control loop*.

**The real problem.** Many tasks need the model to take actions: search the web, call APIs, edit files, browse a UI. Each step is an LLM call; each failure compounds. Cost and latency explode if not engineered carefully.

**Sub-solutions:**

| Pattern | What it resolves | Pros | Cons |
|---|---|---|---|
| **Function calling / structured outputs** | LLM emits invalid JSON or hallucinated APIs | Reliable schema, machine-parseable | Limited to APIs you've defined |
| **ReAct (Reason + Act loop)** | Need to interleave thinking and tool calls | Simple, strong baseline | Loops; cost grows linearly with steps |
| **Plan-and-execute** | Long-horizon tasks with planning step | Better than greedy ReAct on multi-step | Plan is brittle; replanning is hard |
| **Tree-of-thoughts / self-consistency** | Single-shot reasoning fails | Better correctness on hard problems | Multiplies token cost |
| **Reflection / self-critique** | Model can catch its own errors sometimes | Free quality lift | Doesn't help when model is confidently wrong |
| **MCP (Model Context Protocol)** | Every team reinvented tool schemas | Standardization across hosts | Young ecosystem |
| **Memory (scratchpad, episodic, semantic)** | Context window overflow over long tasks | Enables long-running agents | Memory retrieval is itself a RAG problem |
| **Sandboxing (browser, container, OS)** | Tool calls have side effects | Safety, reproducibility | Infra complexity |

**Why most agent demos don't ship.**
- Error compounding: 90% step accuracy → 35% over 10 steps.
- Cost: an agent doing 20 LLM calls per task at $0.01 each = $0.20/task. At a million tasks → $200K.
- Latency: 20 sequential calls × 2s each = 40s; users abandon.
- Verification: how do you know it succeeded? Often a separate LLM judge — adds another call.

**Where agents do work today.** Bounded domains with clear feedback signals: code (compiler errors), browser tasks (DOM state), data analysis (cell outputs), customer support (ticket resolution).

**Hardware angle.** Agent workloads benefit disproportionately from **prefix caching** (system prompt + tool schemas are identical across calls) and **structured-output engines** (SGLang, Outlines, XGrammar). A 50% prefix-cache hit can halve cost.

---

### 3.5 Domain Adaptation (Fine-Tuning, PEFT, Distillation)

> Reach for fine-tuning *after* prompting and RAG hit their ceiling — not before.

**The real problem.** Sometimes a model needs to internalize a new format, style, or specialized capability that prompting can't reliably elicit. Sometimes you need a small fast model to match a big slow one.

**Sub-solutions:**

| Method | What it resolves | VRAM (7B) | Cost (rel.) | Pros | Cons |
|---|---|---|---|---|---|
| **Full fine-tuning (FSDP/ZeRO-3)** | Maximum capacity change | ~80GB+ | 10× | Best quality ceiling | Expensive; risk of catastrophic forgetting |
| **LoRA** | Low-rank adapters; ~0.1–1% of params trained | ~24GB | 1× | Cheap; multiple adapters per base | Slightly weaker than full FT; rank choice matters |
| **QLoRA** | 4-bit base + LoRA | ~16GB | 0.5× | Fits 7B on consumer GPU | Quantization noise during training |
| **DPO / ORPO** | Preference tuning without reward model | ~LoRA | ~1× | Best for "make it sound right" tasks | Needs preference pairs |
| **Continued pretraining** | New domain corpus (medical, legal, code) | full | 5–20× | Genuine knowledge injection | Needs lots of clean domain text |
| **Distillation** | Compress big-teacher → small-student | depends | medium | Pareto-best small models often come from distill | Teacher must be available; legal constraints |
| **Model merging (TIES, DARE, SLERP)** | Combine multiple fine-tunes | none | ~zero | No training; preserves multi-skill | Quality is hit-or-miss |
| **RAG instead** | Knowledge changes / freshness | none | low | Updates without retraining | Doesn't change behavior or format |

**When fine-tuning is the wrong answer:**
- Knowledge changes faster than you'd retrain → use RAG.
- You only have ~100 examples → use few-shot prompting; FT will overfit.
- The base model already does this well → check first.
- You need to *add facts* — fine-tuning doesn't reliably do this; RAG does.

**Hardware sizing rule of thumb:**
- Full FT memory ≈ params × (2 weights + 4 grads + 8 Adam states) = ~14× params.
- LoRA: weights frozen → only a few % of full FT memory.
- QLoRA: 4-bit weights → another 4× reduction on the largest term.

**Common pipeline in 2025:** QLoRA on 7B–13B models for first iteration → graduate to LoRA on 70B for production → revisit data/eval before scaling further.

---

### 3.6 Evaluation & Production Reliability

> "We added LLM features and shipped them; now nobody on the team can tell if a change made it better." This is the #1 silent killer of GenAI products.

**The real problem.** Generative outputs are open-ended. There's no "test passed / failed" without effort. Without evals you're flying blind, model upgrades become terrifying, and regressions ship undetected.

**Sub-solutions:**

| Method | What it resolves | Pros | Cons |
|---|---|---|---|
| **Golden dataset (50–500 hand-curated examples)** | Catch regressions before deploy | Cheap, interpretable, fast | Doesn't capture distribution drift |
| **LLM-as-judge** | Scaling human eval | 10–100× cheaper than humans | Bias toward verbose/own outputs; low for nuanced tasks |
| **Pairwise preference (Elo, MT-bench)** | Absolute scoring is noisy | More reliable than pointwise | Doesn't tell you absolute quality |
| **Task-specific metrics** | Narrow tasks have ground truth | Cheap, automatable | Don't generalize |
| **Reference-based (BLEU, ROUGE, BERTScore)** | Translation, summarization | Standard, fast | Don't correlate well with helpfulness on open tasks |
| **Online A/B testing** | Real users decide | Ground truth | Slow; needs traffic; risky |
| **Observability / tracing (Langfuse, Helicone, Arize, LangSmith)** | Debug what actually happened | Essential for agents | Yet another system |
| **Red-teaming / jailbreak suites** | Catch safety regressions | Catches the worst tail | Adversarial coverage is incomplete |
| **Shadow deployment** | Test new model on real traffic without exposure | Real-distribution eval, low risk | Infra cost; only catches what you measure |

**What companies actually track in production dashboards:**
- P50 / P95 / P99 latency (TTFT and TTFCT separately).
- $/request, $/active user, $/1M tokens.
- Refusal rate, harmful-content flag rate.
- Task-specific KPIs (deflection, acceptance, conversion).
- Customer-reported issues per 1M sessions.

**Eval best practice (industry-tested):**
1. Build a **regression set** (≤200 examples) and gate every release on it.
2. Pair it with a **diversity set** that explicitly targets your failure modes.
3. Use **LLM-as-judge** for scale, but periodically calibrate against humans.
4. **Trace everything** in production; sample for human review weekly.
5. Treat evals as code — versioned, reviewed, debated.

---

### 3.7 Foundation Model Training (What to Understand, Not Build)

> You probably won't pretrain a model in this job. But you *will* be asked about it.

**Why this matters as an AI Engineer (not training engineer).** Choosing models, debugging quality issues, and reading papers all require you to understand training-side decisions.

**Things to understand at a conceptual level:**

| Concept | What it solves | Why you care |
|---|---|---|
| **Data parallelism (DP)** | Throughput across many workers | Default; the simplest scaling axis |
| **Tensor parallelism (TP)** | Single-GPU memory ceiling | Determines deployment topology |
| **Pipeline parallelism (PP)** | Cross-node memory scaling | Pipeline bubbles = wasted compute |
| **Expert parallelism (EP)** | MoE expert sharding | Why MoE inference is hard |
| **Sequence parallelism (SP)** | Long-context activation memory | Why million-token training is even possible |
| **ZeRO / FSDP** | Optimizer/grad/param sharding | Lets a 70B fit on 8×80GB |
| **Mixed precision (BF16, FP8, FP4)** | Memory + speed | Why H100/B200 matter; FP8 ≈ 2× over BF16 |
| **Activation checkpointing** | Trades compute for memory | Required for long contexts |
| **Scaling laws (Chinchilla, post-Chinchilla)** | Optimal compute allocation | Why "20 tokens per param" was a thing |
| **Data quality > quantity** | Most signal comes from data curation | DCLM, FineWeb, etc. — papers worth reading |
| **MoE (sparse activation)** | Decouple capacity from FLOPs | Mixtral, DeepSeek-V3, GPT-4-class architecture |
| **MFU / HFU** | Are you using the GPU? | Frontier labs target 40–50% MFU |

**The single most valuable artifact to read:** *The Ultra-Scale Playbook* (Hugging Face) — explains every parallelism dimension in production-relevant terms.

---

### 3.8 Multimodal

> Increasingly the default. Pure-text systems are an edge case in many products.

**Sub-areas:**

| Modality | Production model | What it solves | Constraints |
|---|---|---|---|
| **Vision-language (VLM)** | GPT-4o, Claude, Gemini, Qwen-VL, LLaVA | OCR, doc understanding, UI screenshots | Image tokens are expensive (1 image ≈ 1K–2K tokens) |
| **OCR pipelines** | Surya, PaddleOCR, GPT-4V hybrid | Structured doc extraction | Layout parsing is harder than recognition |
| **Speech (ASR)** | Whisper, Deepgram, AssemblyAI | Transcription | Latency for streaming is a craft |
| **Speech (TTS)** | ElevenLabs, OpenAI TTS, OS models | Voice agents | Real-time latency budgets are tight |
| **Image generation** | SDXL, Flux, DALL·E 3, Imagen | Marketing, creative | Separate craft; mostly diffusion |
| **Embeddings (multimodal)** | CLIP, SigLIP, ImageBind | Search across modalities | Quality varies by domain |

**Architectural patterns for VLMs (worth knowing):**
- **Frozen vision encoder + projector + LLM** (LLaVA-style) — cheapest, weakest.
- **Cross-attention from text to image features** (Flamingo, IDEFICS) — better grounding.
- **Native multimodal pretraining** (GPT-4o, Gemini, Chameleon) — best, frontier-only.

---

## Part 4 — Picking a Model: 2024–2025 Landscape

> Knowing which model to pick — and why — is as important as knowing how to use them.

### Comparison

| Model | Provider | Access | Architecture Highlights | Context | Strengths | Weaknesses |
|---|---|---|---|---|---|---|
| **GPT-4o / GPT-4.1 / o3** | OpenAI | Closed API | Dense + reasoning variants; native multimodal; RLHF + RLAIF | 128K–1M | Strongest ecosystem; tool use is mature; reasoning models lead on hard problems | Closed; rate limits; no FT for top tier |
| **Claude 3.5 / 4 / Opus** | Anthropic | Closed API | Constitutional AI + RLHF; strong long-context | 200K | Best long-doc reasoning; safest by default; coding strength | Closed; no image generation |
| **Gemini 1.5 / 2.0 / 2.5** | Google DeepMind | Closed API | Sparse MoE; native multimodal incl. video; TPU-trained | 1M+ | Largest context; native video; deep Google integration | Quality less consistent; Google lock-in |
| **LLaMA 3.1 / 3.3** | Meta | Open weights | Decoder-only; GQA; RoPE; BF16 native | 128K | The default open-weights baseline; broad tooling | Self-host complexity; gap vs frontier |
| **Mistral / Mixtral** | Mistral AI | Open + API | Mixtral: MoE 8×7B; sliding window | 32K–128K | Strong cost/quality; Apache 2.0 | Smaller capability ceiling |
| **DeepSeek V3 / R1** | DeepSeek | Open + API | MoE + Multi-head Latent Attention (MLA); FP8 training | 128K | SOTA open-source; ultra-cheap API; R1 rivals o1 on math | Compliance concerns; ecosystem outside China is thin |
| **Qwen 2.5 / 3 / QwQ** | Alibaba | Open + API | Dense + MoE variants; multilingual focus | 128K | Best multilingual (esp. Chinese); wide size range | Western tooling lag |
| **Gemma 2 / 3** | Google | Open weights | Distilled from Gemini; interleaved attention | 8K–128K | Excellent small-model quality; on-device candidates | Smaller capability than frontier |
| **Phi-3 / Phi-4** | Microsoft | Open weights | Small models trained on heavily curated data | 128K | Punches above weight; great for edge | Narrow data distribution |

### When to pick what

| Need | Recommendation |
|---|---|
| Best overall API | GPT-4o / Claude 4 (Claude for long docs, GPT for multimodal + tool ecosystem) |
| Best reasoning / math | OpenAI o-series, Claude 4 thinking, DeepSeek R1 |
| Self-host quality | Llama 3.3 70B or DeepSeek V3 (MoE) |
| Self-host cheap | Mistral 7B / Gemma 2 / Phi-4 |
| Best long-context | Claude (200K) or Gemini 2.5 (1M+) |
| Best multilingual | Qwen 2.5 / 3 |
| Fine-tuning target | Llama 3 or Mistral — best LoRA/QLoRA tooling, permissive licenses |
| Local dev / on-device | Gemma 2B, Phi-4-mini, Llama 3.2 1B/3B |

**The shipping pattern:** prototype on a closed API → measure token volume + quality requirements → switch to open-weights self-host once monthly bill > engineering time saved.

---

## Part 5 — Foundations to Backfill (Just-in-Time)

> Don't read these front-to-back. Pull each in when a problem in Part 3 forces you to.

### 5.1 Transformer architecture (you should be able to whiteboard it)

- Self-attention: Q, K, V; why divide by √d; why softmax.
- Multi-head attention; per-head dimension; head merging.
- MQA / GQA / MLA — why production models all use one of these (KV cache memory).
- FFN block (gated variants: SwiGLU, GeGLU).
- Pre-norm vs post-norm; RMSNorm vs LayerNorm.
- Residual stream perspective.
- Causal mask, padding mask.
- Positional encoding evolution: sinusoidal → learned → RoPE → ALiBi → YaRN.

### 5.2 Tokenization (more important than you'd think)

- BPE, byte-level BPE, SentencePiece.
- Why vocab size affects latency (output projection cost).
- Tokenizer mismatches across models (same string ≠ same token count).
- Why Chinese / code / numbers tokenize "weirdly."

### 5.3 Sampling and decoding

- Greedy, top-k, top-p (nucleus), temperature, min-p.
- Beam search (and why it's mostly dead for chat).
- Constrained decoding (JSON schema, regex, grammars).
- Speculative decoding (draft/target dynamics).

### 5.4 Loss functions

- Cross-entropy and what perplexity means.
- KL divergence (used in RLHF reference loss).
- DPO loss intuition (why it's "RLHF without RL").

### 5.5 Build it from scratch (the deep backfill — 手搓LLM)

> Optional but disproportionately valuable. After you've shipped a few products, this is what makes you genuinely deep.

**Stage 1 — minimum viable Transformer.** Implement nanoGPT / minGPT from scratch. Token embedding → positional encoding → causal attention → FFN → LayerNorm → autoregressive decode. Milestone: 50M params, generates Shakespeare-like text on a single GPU.

**Stage 2 — attention variants and numerical stability.** FlashAttention paper, RMSNorm, pre-norm vs post-norm; swap softmax attention for linear attention (Performer / RetNet) and compare convergence.

**Stage 3 — tokenizer & embeddings.** Train your own BPE. Compare vocab=8K vs 32K. Understand vocab-size / latency / quality tradeoffs.

**Stage 4 — training tricks.** Gradient clipping, LR warmup, weight decay, mixed precision, gradient accumulation, gradient checkpointing.

**Stage 5 — inference & decoding.** Implement KV cache. Compare with/without. Implement greedy / top-k / top-p / temperature.

**Stage 6 — research-style projects.** Pick one:
- Long-context modeling (sliding window, RoPE scaling, YaRN, RetNet).
- Sequence recommendation as language modeling (Transformer4Rec, decision-transformer-style).
- Speculative decoding with a small draft model.

**Order:** nanoGPT → variants → tokenizer/training → one research project. 2–3 months of evening work if you're committed.

---

## Part 6 — Interview Signal: What Top AI Teams Actually Probe

The pattern across DeepMind / Anthropic / OpenAI / Meta GenAI / Mistral interviews:

### System design (most common, most weighted)

- "Design a doc-Q&A system for 100M documents and 100K daily users."
- "Design an agent that can complete browser-based tasks with <$0.10/task budget."
- "Design inference serving for a 70B model serving 10K concurrent users."

What they're checking: do you reason about retrieval architecture, latency budgets, batching, GPU sizing, eval strategy, failure modes — *together*, not in isolation.

### Hardware reasoning

- "How much VRAM do you need to serve LLaMA-70B at FP16 with 32K context for 16 concurrent users?"
- "Why is decode bandwidth-bound and prefill compute-bound?"
- "Walk me through what happens to KV cache memory as batch size and context length scale."

### Tradeoff questions

- "When would you fine-tune vs use RAG vs just prompt better?"
- "Your DPO model has lower MMLU than the SFT base — is that bad?"
- "Pick one: 8B model fine-tuned, or 70B prompted. How do you decide?"

### Deep-dive (one or two of these)

- "Why is attention O(N²) and is that fundamental? What does FlashAttention change about that?"
- "Why does scaling work? What does Chinchilla tell us, and what's different post-Chinchilla?"
- "Why GQA? Why MLA?"
- "Why does causal LM enable few-shot at all?"

### Failure-mode questions (separates seniors from juniors)

- "Your RAG system was great in dev, hallucinates in prod — diagnose."
- "Your fine-tuned model's MMLU dropped 5 points — what happened, what do you do?"
- "Your agent is in an infinite loop on 2% of tasks — how do you debug?"

---

## Part 7 — Suggested Project Track

> Pick three. Real projects > a long reading list.

### Project 1 — Production-grade RAG

A document Q&A system over a real corpus you care about (your company docs, a paper archive, a code repo). Required to impress:
- Hybrid retrieval (BM25 + dense) with measured recall@k.
- A reranker; ablation showing precision lift.
- A regression eval set with human labels.
- Latency and cost dashboard (P50/P95, $/query).
- A documented failure-mode analysis: 10 cases that fail, why, and what you tried.

### Project 2 — Self-hosted inference at scale

Deploy a 7B–13B open-weights model with vLLM or SGLang on a single GPU. Required:
- Concurrency benchmark (throughput vs latency curve).
- Quantization comparison (BF16 vs INT8 vs INT4 — quality + perf).
- Prefix caching enabled, with measured hit-rate and savings.
- A cost-per-1M-tokens calculation vs the API equivalent.

### Project 3 — Eval-first GenAI feature

Pick a narrow task (classification, structured extraction, summarization). Build:
- A 200-example golden set.
- An LLM-as-judge harness with humanly-calibrated agreement.
- Three implementations (prompt-only, prompt+few-shot, fine-tuned small model).
- A Pareto plot of cost vs quality vs latency.

### Optional Project 4 — Agent with a verifier

A bounded agent (e.g. a SQL-question answerer, a code-edit bot, a browser-task agent) with:
- Function calling / structured outputs.
- Per-step tracing.
- A verifier (programmatic or LLM judge) that catches >50% of failures.
- A cost-per-task budget that doesn't run away.

---

## References

### Practical / industrial
- *The Ultra-Scale Playbook* — Hugging Face. The single best reference on parallelism and large-scale training.
- vLLM, SGLang, TensorRT-LLM documentation — read the design rationale, not just the API.
- Anthropic, OpenAI, DeepSeek model cards and tech reports — these are how the field really communicates now.
- Hamel Husain — *Your AI product needs evals*, *Fuck you, show me the prompt* (blog series on real-world AI engineering).
- Chip Huyen — *Designing Machine Learning Systems*; *AI Engineering* (book).
- Eugene Yan — pragmatic LLM application writing.

### Foundational papers (read when relevant in Part 3)
- Attention is All You Need (2017).
- Scaling Laws / Chinchilla (2020 / 2022).
- FlashAttention (1, 2, 3).
- LoRA, QLoRA.
- DPO, ORPO.
- Mixtral, DeepSeek-V3 (MoE + MLA).
- RAG (Lewis et al., 2020), HyDE, ColBERT.
- ReAct, Toolformer.
- Constitutional AI.

### Career / motivation
- [How I got a job at DeepMind as a research engineer without an ML degree](https://gordicaleksa.medium.com/how-i-got-a-job-at-deepmind-as-a-research-engineer-without-a-machine-learning-degree-1a45f2a781de).
- Lilian Weng's blog (lilianweng.github.io) — the best free LLM textbook in disguise.
