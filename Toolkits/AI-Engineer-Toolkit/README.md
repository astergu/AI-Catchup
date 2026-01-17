- [AI Engineer Toolkit](#ai-engineer-toolkit)
  - [Skill Map](#skill-map)
    - [Must-Learn AI Engineer Toolkit](#must-learn-ai-engineer-toolkit)
    - [Optional](#optional)
  - [90-Day Study Plan](#90-day-study-plan)
    - [Month 1: Core Foundations (Hands-On)](#month-1-core-foundations-hands-on)
      - [Week 1–2: Transformers \& LLM Basics](#week-12-transformers--llm-basics)
      - [Week 3–4: Prompting + APIs](#week-34-prompting--apis)
    - [Month 2: Real Systems (This Is Where You Shine)](#month-2-real-systems-this-is-where-you-shine)
      - [Week 5–6: RAG (Critical Skill)](#week-56-rag-critical-skill)
      - [Week 7–8: Fine-Tuning \& PEFT](#week-78-fine-tuning--peft)
    - [Month 3: Production + Differentiation](#month-3-production--differentiation)
      - [Week 9–10: Evaluation \& Safety](#week-910-evaluation--safety)
      - [Week 11–12: Deployment \& Optimization](#week-1112-deployment--optimization)
    - [Projects That Get You Hired (Very Important)](#projects-that-get-you-hired-very-important)
      - [Project 1: RAG for Real Use Case](#project-1-rag-for-real-use-case)
      - [Project 2: Multi-Task GenAI System](#project-2-multi-task-genai-system)
      - [Project 3: LLM Evaluation Framework](#project-3-llm-evaluation-framework)
  - [手搓LLM](#手搓llm)
    - [阶段 1：从零实现 Transformer（最重要）](#阶段-1从零实现-transformer最重要)
    - [阶段 2：Attention 变体 \& 数值稳定性](#阶段-2attention-变体--数值稳定性)
    - [阶段 3：Tokenizer \& Embedding（很多人忽略）](#阶段-3tokenizer--embedding很多人忽略)
    - [阶段 4：训练技巧（工业级）](#阶段-4训练技巧工业级)
    - [阶段 5：推理 \& 解码](#阶段-5推理--解码)
    - [阶段 6：DeepMind 风格进阶项目](#阶段-6deepmind-风格进阶项目)
      - [项目 A：长序列语言建模](#项目-a长序列语言建模)
      - [项目 B：LLM + 序列推荐](#项目-bllm--序列推荐)
    - [DeepMind 面试级追问（你必须能答）](#deepmind-面试级追问你必须能答)
    - [学习顺序（最推荐）](#学习顺序最推荐)
  - [References](#references)


# AI Engineer Toolkit

## Skill Map

### Must-Learn AI Engineer Toolkit

| Area | Why |
| --- | --- |
| Transformers (practical) | Foundation of GenAI |
| LLM APIs (OpenAI, Anthropic, open models) | Most jobs use them |
| Prompt engineering | Real-world performance driver |
| RAG architectures | Most enterprise GenAI |
| Fine-tuning (LoRA) | Customization |
| Inference optimization | Cost & latency |

### Optional

- Training from scratch
- Deep theoretical proofs
- GAN math
- Reinforcement learning theory (except RLHF intuition)

## 90-Day Study Plan

| Month | Topics | Week breakdowns |
| --- | --- | --- |
| 1 | [Core Foundations (Hands-On)](#month-1-core-foundations-hands-on) | [Week 1–2: Transformers & LLM Basics](#week-12-transformers--llm-basics) <br> [Week 3–4: Prompting + APIs](#week-34-prompting--apis) |
| 2 | [Real Systems (This Is Where You Shine)](#month-2-real-systems-this-is-where-you-shine) | [Week 5–6: RAG (Critical Skill)](#week-56-rag-critical-skill) <br> [Week 7–8: Fine-Tuning & PEFT](#week-78-fine-tuning--peft) |
| 3 | [Production + Differentiation](#month-3-production-differentiation) | [Week 9–10: Evaluation & Safety](#week-910-evaluation--safety) <br> [Week 11–12: Deployment & Optimization](#week-1112-deployment--optimization) |


### Month 1: Core Foundations (Hands-On)

> Goal: Understand how LLMs work + use them fluently

#### Week 1–2: Transformers & LLM Basics

- Learn:
  - [Self-attention (QKV intuition)](./self_attention.md)
  - Tokenization (BPE)
  - Why scaling works
  - Decoder-only models (GPT-style)
- Resources:
  - [Andrej Karpathy: Nerual Networks - Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
  - [Attention is All You Need](https://arxiv.org/pdf/1706.03762)
  - [Understanding and Coding the Self-Attention Mechanism of LLM From Scratch](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)
  - [Understanding Large Language Models -- A Transformative Reading List](https://sebastianraschka.com/blog/2023/llm-reading-list.html)
  - [Visualizing Neural Machine Translation: Mechanics of Seq2Seq Models with Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention)
  - [Understanding Attention Mechanism](https://medium.com/@shashank7.iitd/understanding-attention-mechanism-35ff53fc328e)
  - [Attn Illustrated: Attention](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)
  - [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  - [Self-Attention Mechanisms in Natural Language Processing](https://medium.com/@Alibaba_Cloud/self-attention-mechanisms-in-natural-language-processing-9f28315ff905)
  - [Illustrated Self-Attention](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)
  - [Let’s build GPT](https://www.youtube.com/watch?v=Qz6p0zZ6zZM)
  - [HuggingFace Transformers docs](https://huggingface.co/docs/transformers/en/index)

- Deliverable:
  - Load a LLaMA / Mistral model
  - Generate text locally

#### Week 3–4: Prompting + APIs

- Learn:
  - Prompt templates
  - Few-shot vs zero-shot
  - Chain-of-thought
  - Tool calling
  - Structured outputs (JSON)


- Deliverable:
  - Build a prompt-driven text classifier
  - Compare prompting vs fine-tuning


### Month 2: Real Systems (This Is Where You Shine)

> Goal: Build production-like GenAI systems

#### Week 5–6: RAG (Critical Skill)

- Learn:
    - Embeddings
    - Vector databases (FAISS, Pinecone)
    - Chunking strategies
    - Retrieval evaluation


- Deliverable:
    - Build a document Q&A system
    - Measure recall@k
    - Handle hallucinations

#### Week 7–8: Fine-Tuning & PEFT

- Learn:
    - LoRA
    - Instruction tuning
    - When fine-tuning beats prompting
    - Data quality > data quantity

- Deliverable:
    - Fine-tune a small LLM on a domain task
    - Compare with prompting baseline

### Month 3: Production + Differentiation

> Goal: Be job-ready

#### Week 9–10: Evaluation & Safety

- Learn:
    - LLM evaluation is hard
    - Automatic metrics (BLEU ≠ enough)
    - Human eval frameworks
    - Guardrails

- Deliverable:
    - Build an evaluation harness
    - Track cost / latency / quality trade-offs

#### Week 11–12: Deployment & Optimization

- Learn:
    - Quantization (4-bit / 8-bit)
    - Batch inference
    - Caching
    - Streaming generation

- Deliverable:
    - Deploy a GenAI service
    - Cost breakdown analysis

### Projects That Get You Hired (Very Important)

These are strong signals to hiring managers:
#### Project 1: RAG for Real Use Case

Example:
“AI assistant for app monetization / ads optimization docs”

Show:
- Chunking strategy
- Retrieval metrics
- Failure analysis



#### Project 2: Multi-Task GenAI System

Tie to your background:
- Generate ad copy
- Predict conversion likelihood
- Explain why copy works


This shows hybrid GenAI + ranking expertise.

#### Project 3: LLM Evaluation Framework

Very few candidates do this well.


## 手搓LLM

🧠 总体学习路径（从 0 → LLM 内核）

你不需要一开始就训练 7B 模型，而是：

> 小模型 + 正确结构 + 可控实验 → 扩展到 LLM

### 阶段 1：从零实现 Transformer（最重要）

🎯 目标

- 完全理解并手写：
    - Token embedding
    - Positional encoding
    - Self-attention（QKV）
    - Multi-head attention
    - FFN
    - LayerNorm
    - Mask（causal / padding）
    - Autoregressive decoding

✅ 强烈推荐的 repo（按教学价值）

- 1️⃣ [nanoGPT ⭐⭐⭐⭐⭐（必做）](https://github.com/karpathy/nanoGPT)
  - 为什么是首选：
    - 极简、干净、可一行一行读懂
    - 覆盖 LLM 90% 的关键逻辑
    - 可在 laptop / 单 GPU 跑通
  - 🔨 建议你做的事：
    - 手写一版不看代码
    - 对照实现 causal mask
    - 改写 attention（比如换 RMSNorm）
  - 📘 配套文章：
    - Karpathy: Let's build GPT from scratch
  - 🎯 里程碑：
    - 能用 50M 参数模型生成合理文本
- 2️⃣ [minGPT ⭐⭐⭐⭐⭐（必做）](https://github.com/karpathy/minGPT)
  - 为什么是首选：
    - 这是 nanoGPT 的前身，更教学导向。
  - 🧪 你应该能回答的问题
    - 为什么 attention 要除以 √d？
    - 为什么 LayerNorm 在残差前/后？
    - causal mask 如何实现？

### 阶段 2：Attention 变体 & 数值稳定性

🎯 目标

理解 LLM 的稳定训练与推理细节


推荐文章 / repo
- 🔹 Attention 稳定性
    - FlashAttention paper
    - RMSNorm paper
    - Pre-norm vs Post-norm
- 🔹 Linear Attention
    - Performer
    - Linformer
    - RetNet（DeepMind）
- 📦 推荐 repo：
    - https://github.com/HazyResearch/flash-attention
    - https://github.com/google-research/retention
- 🎯 手搓任务：
    - 把 softmax attention 换成线性 attention
    - 比较 loss / 收敛速度

### 阶段 3：Tokenizer & Embedding（很多人忽略）

🎯 目标

理解：

- BPE / SentencePiece
- subword 影响
- vocab size tradeoff

📦 推荐：

- https://github.com/google/sentencepiece
- HuggingFace tokenizers

🔨 实战：

- 自己训练 tokenizer
- 对比 vocab=8k vs 32k


### 阶段 4：训练技巧（工业级）

🎯 目标

理解 LLM 能跑起来的关键技巧

必须掌握

- Gradient clipping
- Learning rate warmup
- Weight decay
- Mixed precision
- Gradient accumulation

📘 推荐：

- Scaling Laws for Neural Language Models
- HuggingFace training docs


### 阶段 5：推理 & 解码

🎯 目标

理解：

- Greedy / Top-k / Top-p
- Temperature
- KV cache
- 长文本生成

🔨 实现：

- KV cache 加速生成
- 比较有无 cache 的速度

### 阶段 6：DeepMind 风格进阶项目

#### 项目 A：长序列语言建模

> “如何让 GPT 看更长的上下文？”

你可以：

- 实现 sliding window
- 实现 RetNet
- 实现 RoPE scaling

🎯 输出：

- 实验对比
- GitHub repo + README

#### 项目 B：LLM + 序列推荐

> 把 LLM 当作用户行为模型

- 输入：用户事件序列（tokenized）
- 输出：是否转化 / next event

📦 参考：

- Transformer4Rec
- Decision Transformer

### DeepMind 面试级追问（你必须能答）

> 为什么 causal LM 可以 few-shot？

> 为什么 LayerNorm 不用 BatchNorm？

> 为什么 KV cache 有效？

> attention 的 O(N²) 是否本质？

### 学习顺序（最推荐）

- 1️⃣ nanoGPT（2–4 周）
- 2️⃣ Attention 变体 + 数值稳定性（2–3 周）
- 3️⃣ Tokenizer + Training tricks（2 周）
- 4️⃣ 自选一个研究型项目（4–8 周）


## References

- [How I got a job at DeepMind as a research engineer without a machine learning degree](https://gordicaleksa.medium.com/how-i-got-a-job-at-deepmind-as-a-research-engineer-without-a-machine-learning-degree-1a45f2a781de)
  - ML curriculum (read papers, implement, build projects)
    - Neural Style Transfer
    - DeepDream
    - Generative Adversarial Networks (GANs)
    - NLP & Transformers
    - Graph/Geometric ML
    - Reinforcement Learning
  - Write a blog at the end of each macro, summarize what you've learned
  - Open-source a project in the middle of the macro (implementation)
- [Deep Learning Journey Update: What Have I Learned About Transformers and NLP in 2 Months?](https://gordicaleksa.medium.com/deep-learning-journey-update-what-have-i-learned-about-transformers-and-nlp-in-2-months-eb6d31c0b848)