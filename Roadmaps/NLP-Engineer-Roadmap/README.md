# NLP Engineer Roadmap — Top-Down

In 2026 the "NLP Engineer" title means something specific and increasingly distinct from "AI Engineer" or "DL Engineer." NLP Engineers own **production language and speech systems at scale**: search ranking, content moderation, machine translation, ASR/TTS, document AI, dialog, information extraction. The job is to make text and speech *work* in the real world — across hundreds of languages, billions of queries, sub-second latency budgets, and tokenization realities that LLMs alone don't solve cleanly.

This roadmap is built top-down: **start from the industrial systems language teams actually ship; decompose each into sub-systems that solve specific problems; surface the latency, multilingual, tokenization, and cost constraints that drive every choice; pull in theory only when a real problem demands it.**

## Table of Contents

- [NLP Engineer Roadmap — Top-Down](#nlp-engineer-roadmap--top-down)
  - [Table of Contents](#table-of-contents)
  - [Part 0 — How to Read This Roadmap](#part-0--how-to-read-this-roadmap)
  - [Part 1 — What NLP Engineers Are Actually Hired To Do](#part-1--what-nlp-engineers-are-actually-hired-to-do)
    - [What you're actually graded on](#what-youre-actually-graded-on)
    - ["Should I just use an LLM?" — the framework](#should-i-just-use-an-llm--the-framework)
  - [Part 2 — The Constraints That Drive Every Decision](#part-2--the-constraints-that-drive-every-decision)
    - [2.1 Latency budgets — language is real-time](#21-latency-budgets--language-is-real-time)
    - [2.2 Tokenization — the constraint nobody outside NLP appreciates](#22-tokenization--the-constraint-nobody-outside-nlp-appreciates)
    - [2.3 Multilingual reality](#23-multilingual-reality)
    - [2.4 Label sparsity \& domain shift](#24-label-sparsity--domain-shift)
    - [2.5 Audio / speech-specific constraints](#25-audio--speech-specific-constraints)
    - [2.6 Cost economics at NLP scale](#26-cost-economics-at-nlp-scale)
  - [Part 3 — Industrial Solutions, Decomposed](#part-3--industrial-solutions-decomposed)
    - [3.1 Search \& Retrieval](#31-search--retrieval)
    - [3.2 Text Classification at Scale (Moderation, Intent, Spam, Topic)](#32-text-classification-at-scale-moderation-intent-spam-topic)
    - [3.3 Information Extraction (NER, Relations, Events, Structured)](#33-information-extraction-ner-relations-events-structured)
    - [3.4 Machine Translation](#34-machine-translation)
    - [3.5 Speech Recognition (ASR)](#35-speech-recognition-asr)
    - [3.6 Speech Synthesis (TTS) \& Voice Systems](#36-speech-synthesis-tts--voice-systems)
    - [3.7 Dialog Systems \& Conversational AI](#37-dialog-systems--conversational-ai)
    - [3.8 Summarization](#38-summarization)
    - [3.9 Document AI (OCR + Layout + Understanding)](#39-document-ai-ocr--layout--understanding)
    - [3.10 Multilingual \& Low-Resource NLP](#310-multilingual--low-resource-nlp)
    - [3.11 Semantic Parsing \& Text-to-X (SQL, Code, Structured Output)](#311-semantic-parsing--text-to-x-sql-code-structured-output)
    - [3.12 Code Intelligence](#312-code-intelligence)
    - [3.13 Knowledge Graphs \& Structured Knowledge](#313-knowledge-graphs--structured-knowledge)
  - [Part 4 — System Architecture: The Production NLP Stack](#part-4--system-architecture-the-production-nlp-stack)
  - [Part 5 — Foundations to Backfill (Just-in-Time)](#part-5--foundations-to-backfill-just-in-time)
    - [5.1 Linguistic foundations](#51-linguistic-foundations)
    - [5.2 Statistical NLP (the parts still relevant)](#52-statistical-nlp-the-parts-still-relevant)
    - [5.3 Sequence-model fundamentals](#53-sequence-model-fundamentals)
    - [5.4 Modern NLP architectures](#54-modern-nlp-architectures)
    - [5.5 Evaluation literacy](#55-evaluation-literacy)
    - [5.6 Tokenization deep-dive](#56-tokenization-deep-dive)
    - [5.7 Programming \& tooling](#57-programming--tooling)
    - [5.8 Adjacent foundations](#58-adjacent-foundations)
  - [Part 6 — Interview Signal: What NLP Teams Actually Probe](#part-6--interview-signal-what-nlp-teams-actually-probe)
    - [Coding (universal first round)](#coding-universal-first-round)
    - [NLP system design](#nlp-system-design)
    - [Domain depth](#domain-depth)
    - [Statistical / linguistic reasoning](#statistical--linguistic-reasoning)
    - ["Should I just use an LLM?" judgment](#should-i-just-use-an-llm-judgment)
    - [Failure-mode questions](#failure-mode-questions)
  - [Part 7 — Suggested Project Track](#part-7--suggested-project-track)
    - [Project 1 — Hybrid search over a real corpus](#project-1--hybrid-search-over-a-real-corpus)
    - [Project 2 — Multilingual classifier with cascade](#project-2--multilingual-classifier-with-cascade)
    - [Project 3 — End-to-end ASR fine-tune](#project-3--end-to-end-asr-fine-tune)
    - [Project 4 — Document AI extraction pipeline](#project-4--document-ai-extraction-pipeline)
    - [Project 5 — Faithful summarization with eval](#project-5--faithful-summarization-with-eval)
    - [Optional Project 6 — Voice agent under a budget](#optional-project-6--voice-agent-under-a-budget)
  - [References](#references)
    - [Books worth reading](#books-worth-reading)
    - [Courses](#courses)
    - [Foundational papers (read when relevant)](#foundational-papers-read-when-relevant)
    - [Practical / industrial](#practical--industrial)
    - [Tools to actually know](#tools-to-actually-know)
    - [Reference implementations in this repo](#reference-implementations-in-this-repo)

---

## Part 0 — How to Read This Roadmap

For each industrial solution:

1. **The real problem** — what business / product KPI is being moved, and why naive approaches (or "just use an LLM") fail.
2. **Sub-solutions** — the components a production system decomposes into.
3. **What each resolves** — and the new problem it introduces.
4. **Pros / cons / when it's the wrong tool**.
5. **Hardware / latency / cost footprint**.
6. **Failure modes** you'll see in production.

Theory (linguistics, statistical NLP, sequence-model internals) sits in Part 5. Pull it in when a problem in Part 3 forces you to.

---

## Part 1 — What NLP Engineers Are Actually Hired To Do

The role surface is broad but the **product types are well-defined**:

| Surface | Concrete examples | Primary KPI | Latency budget |
|---|---|---|---|
| **Web / e-commerce search** | Google, Amazon, Bing, eBay, Etsy | Click-through, purchase rate | <100 ms p99 |
| **Enterprise / vertical search** | Glean, Slack, Notion AI search | Answer adoption, time-to-answer | <500 ms |
| **Content moderation / trust & safety** | Meta IG, TikTok, YouTube, Reddit | Precision on policy violations, recall on harm | <100 ms (online); minutes (offline review) |
| **Machine translation** | Google Translate, DeepL, Meta NLLB, Apple | BLEU/COMET, human-eval win-rate | <300 ms (UI); real-time (live) |
| **Speech recognition (ASR)** | Whisper, AWS Transcribe, Azure, Deepgram | WER, real-time factor | streaming <300 ms |
| **Speech synthesis (TTS)** | ElevenLabs, OpenAI TTS, Google, Apple Siri | MOS, naturalness, speaker similarity | first-byte <200 ms |
| **Voice assistants / phone agents** | Alexa, Siri, Cortana, call-center bots | Task success, turn latency | end-to-end <800 ms |
| **Document AI / IDP** | Tesseract, AWS Textract, Google Doc AI, Adobe | Extraction F1, layout fidelity | seconds (batch); UI-fast |
| **Dialog & customer support** | Intercom Fin, Zendesk AI, Salesforce Einstein | Deflection rate, CSAT | <2 s per turn |
| **Information extraction** | Bloomberg, Reuters, healthcare/legal NER | Per-entity F1 by class | batch overnight to seconds |
| **Code intelligence** | Copilot, Cursor, Sourcegraph Cody, JetBrains AI | Acceptance rate, edit-distance preserved | <300 ms TTFT |
| **Multilingual at scale** | Wikipedia, Meta NLLB, Google Translate | Per-language quality parity | varies |

### What you're actually graded on

A senior NLP engineer is graded on **five** things:

1. **Quality on production data** — not on benchmarks. Your model wins MNLI but fails on customer support transcripts? Worthless.
2. **Latency at scale** — sub-100ms p99 search ranking. Real-time ASR. First-byte TTS.
3. **Cost** — millions to billions of requests per day. A 10× more expensive model needs to be 10× better, not 10% better.
4. **Multilingual coverage** — quality parity (or honest reporting of gaps) across the 50+ languages your product ships in.
5. **Production reliability** — the long tail of inputs (code-switching, dialects, emoji, broken Unicode, adversarial text) does not crash the pipeline.

Junior engineers chase benchmark wins. Senior engineers ship reliable language at scale.

### "Should I just use an LLM?" — the framework

This is the most common question NLP engineers face in 2026. The honest decision tree:

| Use case | Recommendation |
|---|---|
| One-off, low-volume task with no latency constraint | **LLM API** — engineering time costs more |
| Sub-100ms latency required | **Distilled / classical model** — LLM round-trip is too slow |
| Billions of requests / day | **Smaller model** — LLM cost dominates eng cost |
| Need explainability / auditability | **Classical / structured** — LLMs are opaque |
| Long-tail entity coverage in narrow domain | **Fine-tune small model on domain data** |
| Knowledge changes daily | **RAG over LLM**, not fine-tune |
| Truly novel reasoning needed | **Frontier LLM** — small models can't fake it |
| Multilingual including low-resource | **Multilingual fine-tune** or NLLB-class — frontier APIs uneven for low-resource |

The senior NLP engineer reasons about this trade *for each task* rather than picking a tool tribally.

---

## Part 2 — The Constraints That Drive Every Decision

### 2.1 Latency budgets — language is real-time

| Surface | End-to-end p99 | Model budget |
|---|---|---|
| Search ranking | 50–150 ms | 10–30 ms |
| Type-ahead / autocomplete | 30–80 ms | <20 ms |
| Real-time translation (live captions) | <500 ms incremental | <200 ms |
| Streaming ASR partial | 200–400 ms | <100 ms per partial |
| TTS first-byte | <300 ms | <200 ms |
| Voice agent turn (ASR → NLU → LLM → TTS) | 800–1500 ms total | each stage <300 ms |
| Document AI (per-page extraction) | 1–5 s | seconds OK |
| Content moderation (online) | <200 ms | <100 ms |
| Content moderation (offline / async) | minutes | full reasoning |

**Implication.** Heavy LLMs almost never run in the hot path of search, ASR, or autocomplete. The pattern is *cascade*: cheap classifier → expensive model → human review, with the expensive stage applied only to a tiny fraction.

### 2.2 Tokenization — the constraint nobody outside NLP appreciates

The tokenizer is **the secret tax on every NLP system**. Same string, different tokenizers:

| Tokenizer | "Hello world" | "你好世界" | "مرحبا" | "def foo(x): return x+1" |
|---|---|---|---|---|
| GPT-4o (~o200k) | 2 tokens | 2 tokens | 3 tokens | 11 tokens |
| Llama-3 (128K) | 2 | 5 | 6 | 12 |
| Llama-2 (32K) | 3 | 9 | 10 | 14 |
| GPT-2 / older | 3 | 12+ | 15+ | 16 |
| Whisper | 2 | 2 | 4 | n/a |

**Direct implications:**
- **Cost** — API priced per token; a Chinese sentence may cost 4× the English equivalent on older tokenizers.
- **Latency** — decode is autoregressive; more tokens per response = linearly slower.
- **Context window** — a 128K-token window holds vastly different amounts of content per language.
- **Quality** — over-fragmented tokens hurt reasoning ("2024" as `2`, `0`, `2`, `4` vs `2024`).

**Tokenizer choices an NLP engineer makes:**
- BPE (GPT family), WordPiece (BERT), SentencePiece (T5, Llama), byte-level BPE.
- Vocab size: 32K–256K. Larger = more efficient per language, more parameters in output projection.
- Byte fallback vs unknown token.
- Whether to train your own (almost always yes for code, niche domains, or low-resource languages).

### 2.3 Multilingual reality

There are ~7000 living languages. ~100 have decent training data. ~20 are well-served by frontier models.

| Tier | Languages | Training data | Frontier-model quality |
|---|---|---|---|
| **High-resource** | English, Mandarin, Spanish, French, German, Japanese, Russian, Portuguese, Arabic, Hindi (a few more) | 100B+ tokens | Excellent |
| **Mid-resource** | ~80 languages with Wikipedia + Common Crawl coverage | 1B–100B tokens | OK to good |
| **Low-resource** | ~500 languages with some digital footprint | <1B tokens | Poor; often no model |
| **Endangered / unwritten** | ~6000+ languages | minimal or oral | None |

**Implications:**
- An English-only model may "pass" multilingual eval by translating in/out, with quality cliff for low-resource.
- Tokenization eats low-resource language budgets first (long token counts, off-distribution scripts).
- Code-switching (mixing languages within a sentence) is the production reality everywhere; benchmarks under-represent it.
- Romanization, transliteration, and normalization are domain-specific; no universal pipeline.

### 2.4 Label sparsity & domain shift

Many production NLP tasks have **tiny labeled datasets** by deep-learning standards:

| Task type | Typical labeled set size | Implication |
|---|---|---|
| NER (per domain) | 5K–50K sentences | Active learning, weak supervision, LLM-as-labeler |
| Document classification (vertical) | 1K–100K docs | Fine-tune small encoder |
| Relation extraction | 1K–20K | Distant supervision, prompt LLM |
| Custom intent classifier | 100–10K per intent | Few-shot / data augmentation |
| ASR fine-tune (low-resource) | 10–100 hours | Self-supervised pretrain + fine-tune |
| MT (low-resource pair) | 10K–1M sentence pairs | Backtranslation, multilingual transfer |

The senior NLP playbook: **leverage pretrained backbones + smart data augmentation + active learning + LLM-generated synthetic data.**

### 2.5 Audio / speech-specific constraints

For ASR/TTS you also inherit:

- **Sample rate**: 16 kHz (telephony) vs 24/48 kHz (HiFi). Mismatched rate = catastrophic quality drop.
- **Streaming vs batch**: streaming ASR commits early; can't fix mistakes in past audio.
- **Real-time factor (RTF)**: time-to-process audio / audio duration. <1.0 means faster than real-time. Live products need <0.3 RTF.
- **Endpointing (VAD)**: when did the user stop talking? Get this wrong, the agent interrupts or hangs.
- **Acoustic conditions**: studio vs phone vs car vs windy outdoors → 5× WER variation.
- **Latency vs accuracy tradeoff**: bigger context window = better recognition, slower commits.
- **Speaker variation**: accent, age, gender, child speech, disordered speech — all pull WER apart.

### 2.6 Cost economics at NLP scale

| Resource | Order-of-magnitude price | What you get |
|---|---|---|
| Frontier LLM API per 1M tokens | $1–$15 | Easy quality; expensive at scale |
| Self-hosted 7B model | $0.10–0.50 / 1M tokens | Break-even ~10–100M tokens/month |
| BERT-base classifier on CPU | <$0.01 / 1M docs | Production-cheap baseline |
| Whisper large-v3 on GPU | ~$0.005 / minute audio | Workhorse ASR |
| ElevenLabs TTS API | ~$0.10–0.30 / 1K chars | Premium voice; expensive |
| Self-hosted neural TTS | low; needs GPU + VRAM | Cost-controlled at scale |

The NLP engineer's job is to *cascade*: classical model handles 95% cheaply, LLM handles the 5% that matters. Pure-LLM pipelines look great in demo and bankrupt the team in production.

---

## Part 3 — Industrial Solutions, Decomposed

### 3.1 Search & Retrieval

> The largest, oldest, and most economically important NLP system on Earth. Web search is the original NLP-at-scale problem.

**The real problem.** Given a user query, return the most relevant documents from a corpus of 10⁶–10¹² documents in <100 ms. Make the first result useful — clicks fall off geometrically with rank.

**The classical search funnel** (every search system, modulo names):

```
query → query understanding → retrieval → first-stage ranking → second-stage ranking → rerank/blend → result page
```

**Sub-solutions, by stage:**

| Stage | What it does | Industrial techniques |
|---|---|---|
| **Query understanding** | Intent, entity tagging, spell-correct, expansion, language detection | Classifiers, NER, query rewriting, segmentation |
| **Retrieval (sparse)** | Inverted-index keyword match | BM25, query expansion (RM3), Anserini/Lucene/Elastic |
| **Retrieval (dense)** | Embedding ANN | BGE, E5, GTE, Voyage, OpenAI embeddings + Faiss/HNSW/ScaNN |
| **Hybrid retrieval** | Sparse + dense fusion | RRF, learned-sparse (SPLADE) |
| **First-stage ranking** | Cheap learning-to-rank over retrieved candidates | LightGBM ranker, small bi-encoder |
| **Second-stage rerank** | Heavier cross-encoder | MonoBERT, ColBERT, T5-rerank, LLM-rerank |
| **Diversity / freshness / personalization** | Slate-level | MMR, DPP, freshness boosts |
| **Snippet generation** | Extract answer-bearing passage | Span extraction, abstractive snippets |

**Pros / cons of common approaches:**

| Approach | Pros | Cons |
|---|---|---|
| BM25 alone | Free, sub-ms, language-agnostic, deterministic | No semantic match; "dog" ≠ "puppy" |
| Pure dense retrieval | Semantic match; catches paraphrases | Misses rare keywords, IDs, numbers; tokenizer-bound |
| Hybrid (BM25 + dense + RRF) | Best of both | Two indexes to operate |
| ColBERT (late interaction) | Strong precision; per-token match | High storage cost (token-level vectors) |
| Cross-encoder rerank | Best precision | Slow; only viable on top-k |
| LLM-as-reranker | Highest quality | Latency + cost dominate |
| SPLADE / learned sparse | Inverted-index-fast + semantic | Specialized infra |

**Hardware footprint:**
- Web search index: tens of TB sharded across thousands of servers.
- Dense vector index for 1B docs at 768-d float16: ~1.5 TB before HNSW overhead.
- Reranker GPU cost: a cross-encoder over top-100 at 50 ms is the bulk of the per-query GPU spend.

**Failure modes:**
- "Lost in the middle" for long-tail queries — top-3 retrieval miss is unrecoverable.
- Domain shift: a model trained on web fails on enterprise/code/legal vocabulary.
- ID-heavy queries (SKUs, CVE-IDs) — pure dense fails entirely without hybrid.
- Personalization feedback loops — clicks reinforce popular docs, tail dies.
- Multilingual: query in language A, docs in language B — needs cross-lingual embeddings or translation.

**Cross-reference.** RAG ([`../AI-Engineer-Roadmap/README.md` §3.3](../AI-Engineer-Roadmap/README.md#33-rag-retrieval-augmented-generation)) is essentially "search + LLM synthesis." Most of this section applies there.

---

### 3.2 Text Classification at Scale (Moderation, Intent, Spam, Topic)

> The unsexy core of production NLP. Every messaging platform, social network, ad system, and email provider runs millions of classifiers.

**The real problem.** Given a piece of text (post, message, query, document), assign a label. Sounds trivial. At scale: billions/day, multilingual, adversarial, with class definitions changing weekly.

**Sub-domains:**

| Domain | Examples | Notes |
|---|---|---|
| Content moderation | Hate speech, harassment, CSAM, terrorism, self-harm | Highest stakes; legal/regulatory |
| Spam / abuse | Email spam, comment spam, fake reviews | Adversarial; arms race |
| Intent classification | "What does the user want?" in chat/voice | Bounded label set; fast iteration |
| Topic / category | Routing news, products, ads | Often hierarchical |
| Sentiment | Review polarity, brand mentions | Easy to demo, hard to ship |
| Language detection | Pre-routing | Fast; CLD3, fasttext |
| Quality / clickbait / misinformation | Newsfeed ranking | Fuzzy labels; subjective |

**Sub-solutions:**

| Method | What it resolves | Pros | Cons |
|---|---|---|---|
| **TF-IDF + logistic regression / SVM** | Fast, interpretable baseline | <1 ms inference; no GPU | Weak on paraphrase; manual feature work |
| **fastText** | Cheap multilingual baseline | n-gram embeddings; trains in minutes | Not as strong as Transformers |
| **DistilBERT / MiniLM / TinyBERT fine-tune** | Modern production default | Good quality, CPU-deployable | Still 5–50 ms/inference |
| **mBERT / XLM-R fine-tune** | Multilingual coverage | One model for 100 languages | Quality tail uneven |
| **Modern small encoder (BGE-small, E5-small) + classifier head** | Strong embeddings + tiny head | Cheap and strong | Embedding model latency |
| **LLM zero-shot / few-shot** | New labels with no training data | Instant deploy | Latency, cost, hallucinated labels |
| **LLM as labeler → distill into small model** | Best of both | Cheap inference + LLM-quality labels | Two-step pipeline |
| **Rule + classifier hybrid** | Hard policy needs deterministic rules | Auditable; combines symbolic + statistical | Rule maintenance burden |
| **Active learning** | Label the hardest examples | Maximizes label efficiency | Needs labeling infra |

**Cascading is the production pattern:**

```
input → cheap rule / regex filter → small classifier → big classifier → LLM judge → human review
            (99% filtered)         (90%)            (8%)            (rare)         (rarest)
```

**Hardware/cost footprint.**
- Classical (TF-IDF + LR): millions QPS on a single CPU box.
- Fine-tuned DistilBERT on CPU: ~10K QPS per server.
- LLM-classifier-by-prompt: 1–10 QPS per GPU; only viable with caching.

**Failure modes:**
- **Class imbalance**: 1:10⁶ for rare harm → AUC misleading; PR-AUC and per-class metrics required.
- **Adversarial drift**: spammers iterate against your classifier; weekly retraining is the floor.
- **Multilingual gaps**: trained on English, deployed worldwide → silent quality holes.
- **Concept drift**: "what counts as misinformation" changes; model frozen on old labels.
- **Annotation disagreement**: <0.7 inter-annotator κ on subjective labels; ceiling on attainable accuracy.
- **Calibration**: thresholds tuned on dev set drift in production; recalibrate on rolling window.

---

### 3.3 Information Extraction (NER, Relations, Events, Structured)

> Turn unstructured text into rows in a database. The bridge between language and downstream systems.

**Sub-tasks:**

| Task | Output | Example |
|---|---|---|
| **NER (Named Entity Recognition)** | Span + type | "Apple announced..." → `Apple/ORG` |
| **Entity linking** | Span → KB entity | `Apple/ORG → Q312` (Apple Inc.) |
| **Relation extraction** | Triple | (Apple, founded_by, Steve Jobs) |
| **Event extraction** | Event + arguments | (Acquisition, buyer=Apple, target=Beats, date=2014) |
| **Slot filling** | Form field → value | `flight.destination = SFO` |
| **Coreference resolution** | Mention → entity cluster | "He"/"Tim Cook"/"Apple's CEO" → same entity |
| **Open IE** | Free-form (subject, predicate, object) | (the company, has launched, a new product) |

**Sub-solutions:**

| Approach | What it resolves | Pros | Cons |
|---|---|---|---|
| **CRF / BiLSTM-CRF** | Classical NER baseline | Strong with small data; CPU-fast | Pre-Transformer era; weaker on hard cases |
| **BERT-style token classifier** | Modern default | Strong; fine-tunes with 10K examples | Domain shift hurts |
| **Span-based models (SpanBERT, SpERT)** | Nested entities, relations | Handles overlapping entities | More complex training |
| **Generative IE (T5/UL2-style "extract → JSON")** | Unified format across tasks | Same model for many tasks | Slower; needs careful prompting |
| **LLM zero/few-shot extraction** | New schema, new domain, no labels | Fast prototype | Cost, latency, hallucination |
| **LLM extraction → human review → small model distill** | Production-cost LLM-quality | Best small-budget recipe | Pipeline maintenance |
| **Distant supervision (KB align + filter)** | Massive weak labels | Free training data | Noisy; bias from KB coverage |
| **Weak supervision (Snorkel / labeling functions)** | Programmatic label generation | Domain-expert encoded | Needs careful denoising |

**The 2026 production recipe** (most teams):
1. Define schema (entities, relations, attributes) and edge cases.
2. Use a frontier LLM with a structured-output / function-calling extraction prompt to label 5K–50K examples.
3. Human-review a sample to validate; iterate the prompt.
4. Distill into a small encoder (DistilBERT-class) for production.
5. Add a small held-out human-labeled regression set.
6. Active-learn labels for the failure modes.

**Hardware footprint.**
- BERT-base NER: 5–20 ms per sentence on CPU; faster on small GPU.
- LLM extraction: 200–2000 ms per doc; viable for batch, not online.
- Coref over a long doc: 100–500 ms.

**Failure modes:**
- **Domain shift** — biomedical entities ≠ news entities ≠ legal. Each needs its own labeled set.
- **Long-tail entities** — rare orgs, code-words, abbreviations dominate the harder cases.
- **Nested / overlapping entities** — many models can't represent them.
- **Boundary errors** — "New York Times" vs "New York" + "Times".
- **Multilingual NER** — capitalization-based features fail in CJK; punctuation differs.
- **LLM extraction hallucinations** — model invents attribute values that aren't in the source.

---

### 3.4 Machine Translation

> The classic NLP problem — and the one most reshaped by transformers, then by LLMs.

**The real problem.** Translate text from language A to language B. "Quality" is multidimensional: fidelity, fluency, idiomaticity, register, terminology consistency, length parity. For long-tail language pairs, training data barely exists.

**Architectural lineage:**

| Era | Paradigm | What it solved |
|---|---|---|
| Pre-2014 | Statistical MT (phrase-based, Moses) | Decades of careful feature engineering |
| 2014–17 | RNN encoder-decoder + attention | First neural MT to beat phrase-based |
| 2017+ | Transformer (Vaswani) | Where the field has stayed since |
| 2019+ | Multilingual MT (mBART, M2M-100, NLLB) | One model, 200 languages |
| 2023+ | LLMs as translators (GPT-4, Claude, Gemini, Tower) | Frontier-model quality on high-resource pairs |
| 2024+ | Retrieval-augmented MT, document-level MT | Term consistency, doc-level coherence |

**Sub-problems and what each resolves:**

| Sub-problem | Solution | Notes |
|---|---|---|
| Sentence segmentation | Domain-aware segmenter | Critical for batch; tokenizer can't handle paragraphs |
| Tokenization | Joint multilingual SentencePiece | 32K–256K vocab; balance per-language |
| Vocabulary balance | Sampling temp on multilingual training | Avoid English dominance |
| Terminology consistency | Glossary injection / constrained decoding | Critical for enterprise |
| Document-level coherence | Doc-context windows, coref-aware decoding | Hard to evaluate; harder to ship |
| Quality estimation (no reference) | COMET-Kiwi, BLEURT-Q, LLM-judge | Required for online decisions |
| Domain adaptation | Fine-tune on in-domain | Medical, legal, software UIs each need it |
| Low-resource pairs | Backtranslation, multilingual transfer, pivot through English | Often the only option |
| Code/math/named-entity preservation | Tag-and-restore patterns | LLMs handle better; classical NMT fragile |

**Evaluation reality:**

| Metric | Pros | Cons |
|---|---|---|
| BLEU | Standard; cheap; reproducible | Penalizes valid paraphrases; saturated |
| chrF / chrF++ | Better for morphologically rich languages | Less interpretable |
| BERTScore | Semantic match | Slower; biased to model's training |
| **COMET / COMET-22 / COMET-Kiwi** | Strong correlation with human judgment | The current production default |
| BLEURT | Trained on human ratings | Smaller community |
| Human eval (MQM, error severity) | Ground truth | Slow, expensive |
| LLM-as-judge | Cheap human-eval proxy | Bias; varies by judge model |

**LLM-vs-NMT decision in 2026:**

| Pair | High-resource | Mid-resource | Low-resource |
|---|---|---|---|
| Quality leader | Frontier LLM | LLM ≈ specialized NMT | Specialized multilingual NMT (NLLB, M2M-100) |
| Cost leader | Specialized NMT | Specialized NMT | Specialized NMT |
| Latency leader | Specialized NMT | Specialized NMT | Specialized NMT |
| Best for live/streaming | Specialized NMT | Specialized NMT | n/a |
| Best for document-level | Frontier LLM | Frontier LLM | Specialized + LLM postedit |

**Failure modes:**
- **Hallucination** — confident output disconnected from source. LLMs are *worse* than NMT here on niche content.
- **Length collapse / explosion** — output too short / too long.
- **Off-target translation** in multilingual MT — outputs in the wrong target language.
- **Terminology drift** — "patient" translated three ways in one document.
- **Cultural / register mismatch** — formal source, casual target.
- **Code / numbers / dates getting "translated"** — most common production bug.

---

### 3.5 Speech Recognition (ASR)

> Audio in, text out. The most performance-critical NLP system in voice products.

**The real problem.** Convert audio waveform to text accurately, fast, and robustly across acoustic conditions, accents, languages, and domains.

**Architectural families:**

| Family | Era | Key idea | Pros | Cons |
|---|---|---|---|---|
| **HMM-GMM hybrid** | classical | Phone-state HMM + GMM acoustic model | Decades of decoder maturity | Outperformed by NN; complex |
| **HMM-DNN hybrid (Kaldi)** | 2012+ | DNN replaces GMM | Dominated until 2020 | Pipeline complex; many components to tune |
| **CTC** | 2014+ | End-to-end alignment-free | Streaming-friendly | Conditional independence assumption |
| **RNN-T (Transducer)** | 2017+ | Streaming + autoregressive | Production gold standard for streaming (Google, Apple) | Training memory cost |
| **Attention encoder-decoder (LAS)** | 2016+ | Pure seq2seq | Strong offline | Not streaming-friendly out of box |
| **Conformer** | 2020+ | Conv + self-attention | The best-in-class encoder | More compute |
| **Whisper / Whisper-style** | 2022+ | Multilingual encoder-decoder + huge weak supervision | Robust; multilingual; offline | Not naturally streaming |
| **Self-supervised pretrain (wav2vec2, HuBERT, BEST-RQ)** | 2020+ | Pretrain on unlabeled audio | Massive low-resource gains | Two-stage training |
| **Audio LLMs (AudioLM, SeamlessM4T, Voxtral)** | 2024+ | Unified speech-to-X | Translation + recognition in one | Compute cost |

**Sub-systems of an ASR pipeline:**

| Stage | What it does |
|---|---|
| Audio input + resampling | 16 kHz mono PCM is the standard |
| Voice activity detection (VAD) | Skip silence; segment streams |
| Feature extraction | Mel filterbank, log-mel, MFCC |
| Acoustic encoder | Conformer / Transformer / LSTM |
| Decoder | RNN-T joint network, attention decoder, or CTC + LM rescoring |
| LM rescoring | N-gram or neural LM to boost lexical accuracy |
| Endpointing | When did the user stop talking? |
| Inverse text normalization (ITN) | "$5.20" not "five dollars and twenty cents" |
| Punctuation + capitalization | Often a separate post-processor |
| Diarization | Who spoke when (multi-speaker) |
| Confidence scoring | For downstream gating / human review |

**Pros / cons of the two production paradigms:**

| Paradigm | Pros | Cons |
|---|---|---|
| **Streaming RNN-T** | Low-latency partials; phone-grade real-time | Training is memory-hungry; harder to multilingual |
| **Whisper-style offline** | Robust; multilingual; cheap to fine-tune | Can't stream natively; longer first-output latency |

**Hardware footprint:**
- Whisper-large-v3 on a single A10/L4: ~0.1–0.3 RTF. Cheap to host.
- Streaming Conformer-RNN-T on phones: feasible at 100MB-class quantized models; on-device on Pixel/iPhone.
- Server-side ASR: a single H100 can serve hundreds of concurrent streams of small Conformer.

**Failure modes:**
- **Domain mismatch**: trained on read speech, deployed on phone calls (8 kHz, codec artifacts, crosstalk).
- **Accent gap**: WER on non-native speech can be 2–3× the baseline.
- **Code-switching**: speakers mixing languages mid-sentence.
- **Hot-word issues**: brand names, niche vocabulary; usually patched with biasing / contextual LM.
- **Hallucinated transcripts** in encoder-decoder ASR (Whisper) on silence / music / long inputs.
- **Punctuation/capitalization drift** when post-processor lags model upgrade.

---

### 3.6 Speech Synthesis (TTS) & Voice Systems

> Text in, audio out. The other half of voice products.

**Sub-systems of a modern TTS:**

| Stage | What it does |
|---|---|
| Text frontend | Normalize numbers, dates, abbreviations; pronunciation lookups |
| G2P (grapheme-to-phoneme) | "tomato" → /təˈmeɪtoʊ/ (or learned end-to-end) |
| Acoustic model / spectrogram predictor | FastSpeech, Tacotron, VITS, Glow-TTS |
| Vocoder | Mel-spectrogram → waveform: HiFi-GAN, BigVGAN, WaveGlow, neural codecs |
| Voice cloning / speaker embedding | One-shot or few-shot speaker reproduction |
| Prosody / emotion control | Pitch, rate, emphasis, emotion |
| Streaming generation | Output chunks as soon as ready (first-byte) |

**Architectural families:**

| Family | Examples | Pros | Cons |
|---|---|---|---|
| **Concatenative** | classical | Natural snippets | Big footprints; can't generalize |
| **Parametric** | classical | Small footprints | Robotic |
| **Tacotron / Tacotron2 + WaveNet** | 2017–18 | First convincingly neural | Slow; teacher-forcing artifacts |
| **FastSpeech / FastSpeech2** | 2019+ | Non-autoregressive; fast | Needs duration alignment |
| **VITS** | 2021 | End-to-end; high quality | Training complexity |
| **Diffusion TTS (NaturalSpeech 2/3)** | 2023+ | High naturalness | Slower inference unless distilled |
| **Neural codec LMs (VALL-E, SpeechX)** | 2023+ | Voice cloning from 3 seconds | Quality variance |
| **Audio-LM / multimodal (GPT-4o voice)** | 2024+ | Unified text+speech with reasoning | Compute |
| **Streaming TTS (StreamSpeech)** | 2024+ | First-byte latency | Complexity |

**Failure modes:**
- **Mispronunciations** of names, place names, brand names — G2P + lexicon needed.
- **Robotic prosody** on long utterances or unusual punctuation.
- **Code-switching breaks pronunciation** — wrong phoneset chosen.
- **Speaker drift** in voice cloning — voice identity shifts within a long output.
- **Hallucinated content** in audio-LM TTS on out-of-distribution input.
- **First-byte latency** killing UX in voice agents — bigger model + slower vocoder = abandonment.

---

### 3.7 Dialog Systems & Conversational AI

> The product surface where ASR, NLU, retrieval, generation, and TTS all collide.

**Sub-architectures:**

| Architecture | When it fits |
|---|---|
| **Rule-based (decision trees)** | IVR, narrow phone menus; deterministic |
| **Slot-filling NLU + business logic** | Bounded domain (booking, ordering); auditable |
| **Retrieval-based** | FAQ-style; cheap, controllable |
| **Generative (frontier LLM, RAG)** | Open-domain support, complex queries |
| **Hybrid (intent + RAG + LLM)** | The 2026 production default |

**Sub-components every production dialog system has:**

| Component | What it does |
|---|---|
| **Intent classifier** | What does the user want this turn? |
| **Entity / slot extractor** | Pull "tomorrow", "JFK", "$50" out |
| **Dialog state tracker** | What's been said, what's confirmed, what's missing |
| **Policy / planner** | What's the next system action? |
| **Response generator** | LLM, retrieval, or template |
| **Grounding / RAG layer** | Pull verified facts from KB |
| **Guardrails** | PII redaction, tone control, refusal |
| **Logging + analytics** | Turn-level metrics, conversation analytics |

**Pros / cons:**

| Pattern | Pros | Cons |
|---|---|---|
| Pure LLM in the loop | Flexible, fast to ship | Hard to control; cost; hallucination |
| Slot-filling + LLM-as-fallback | Auditable; bounded cost | Brittle on out-of-scope inputs |
| Tool-calling LLM (function calling) | Real action-taking | Latency; cascading errors |
| Retrieval-grounded LLM | Best of both | Retrieval miss → bad answer |

**Hardware reality.** Voice-agent total budget = ASR (~200ms) + NLU/RAG/LLM (~600ms) + TTS first-byte (~200ms) ≈ 1s. Anything over 1.5s end-to-end and users hang up. **Streaming everything** (partial ASR, streaming LLM, streaming TTS) is mandatory.

**Failure modes:**
- "Sorry I didn't catch that" loops — ASR + endpointing failures cascade.
- Repetitive responses — LLM stuck in a generation pattern.
- Out-of-scope handling — "I can't help with that" fires too often or not enough.
- Memory loss across turns — forgot the user's name three turns ago.
- Tool-call failures — LLM calls API, API errors, LLM doesn't recover.
- Privacy leaks — model recites training data or other users' info.

**Cross-reference.** Agentic patterns (ReAct, plan-and-execute, MCP) live in [`../AI-Engineer-Roadmap/README.md` §3.4](../AI-Engineer-Roadmap/README.md#34-agents--tool-use).

---

### 3.8 Summarization

> Reduce a long document to a short faithful version. Sounds simple. Isn't.

**Sub-problem flavors:**

| Flavor | Example |
|---|---|
| **Extractive** | Pick top-k sentences; classical baseline |
| **Abstractive** | Generate a summary in new words |
| **Query-focused** | Summarize answer to user question |
| **Multi-document** | Summarize a topic across many sources |
| **Long-document** | Books, transcripts, court filings |
| **Real-time / streaming** | Live meeting / call summaries |

**Sub-solutions:**

| Method | Pros | Cons |
|---|---|---|
| **TextRank / LexRank** | Free, no training | Just picks sentences; no rephrasing |
| **BART / Pegasus / T5 fine-tuned** | Strong abstractive baseline | Limited input length; needs fine-tune |
| **Long-context LLM (Claude 200K, Gemini 1M)** | Whole-document in one call | Cost; "lost-in-the-middle" |
| **Map-reduce summarization** | Long docs in pieces, then combined | Some context loss across chunks |
| **Hierarchical (chunk → section → doc)** | Scales to books | Engineering complexity |
| **Retrieval-augmented summarization** | Multi-doc topic summaries | Needs retrieval quality |
| **Faithfulness-aware decoding** | Reduce hallucination | Slower; constrained decoding |

**Evaluation:**

| Metric | Reality |
|---|---|
| **ROUGE-1/2/L** | Standard but correlates poorly with quality |
| **BERTScore / BLEURT** | Better but still proxy |
| **QAGS / FactScore** | Faithfulness-specific |
| **LLM-as-judge** | Industry default in 2026; calibrate vs human |
| **Human eval** | Ground truth; expensive |

**Failure modes:**
- **Hallucination** — summary asserts facts not in the source. The dominant problem.
- **Coverage holes** — missing the key result from page 4.
- **Length non-compliance** — "give me 3 bullet points" → 8 bullets.
- **Tone / register drift** — summarizing legal docs in casual tone.
- **Lost-in-the-middle** for long-context LLMs — middle of doc underrepresented.

---

### 3.9 Document AI (OCR + Layout + Understanding)

> Forms, invoices, contracts, IDs, receipts, scientific papers. The interface between paper and computers.

**The pipeline:**

```
PDF / image input
     │
     ▼
   Preprocessing (deskew, denoise, binarize)
     │
     ▼
   Layout analysis (regions, columns, tables, figures)
     │
     ▼
   OCR (text recognition per region)
     │
     ▼
   Reading order reconstruction
     │
     ▼
   Structured understanding (key-value, tables, entities)
     │
     ▼
   Domain extraction / downstream
```

**Sub-solutions and recent shifts:**

| Stage | Classical | Modern (2024–26) |
|---|---|---|
| Layout | Heuristic + small CNN (LayoutParser) | Detectron-style detector; **DocLayout-YOLO**, **Surya** |
| OCR | Tesseract, ABBYY | **Surya**, **PaddleOCR**, **TrOCR**, **Got-OCR2** |
| Reading order | Heuristic | Learned (LayoutLMv3) |
| Structured understanding | Rule-based + NER | **LayoutLMv3**, **Donut**, **Pix2Struct**, **Nougat** (LaTeX-aware) |
| End-to-end VLM | n/a | **GPT-4o**, **Gemini**, **Qwen-VL**, **InternVL** treat the whole page as input |

**Pros / cons of approaches:**

| Approach | Pros | Cons |
|---|---|---|
| Modular (layout → OCR → IE) | Auditable; cheap per stage | Errors compound across stages |
| End-to-end (Donut / Pix2Struct) | One model, one error mode | Less transparent; heavier |
| VLM (GPT-4o / Gemini / Qwen-VL) | Strong on hard layouts; zero-shot for new doc types | Cost; latency; long-document scaling |
| Hybrid (modular + VLM fallback) | Pareto-best in 2026 | Pipeline complexity |

**Hardware footprint:**
- OCR-per-page (Surya / PaddleOCR): 50–500 ms on CPU; faster on GPU.
- VLM-per-page (GPT-4o-class): 1–5 s; expensive at high volume.
- LayoutLMv3 fine-tuned: 100–500 ms per page on GPU.

**Failure modes:**
- **Multi-column reading order** — text reads top-to-bottom, left column then right. Easy to get wrong.
- **Tables** — merged cells, nested headers, totals rows. Still hard.
- **Handwriting** — ~3× WER vs printed; specialized models needed.
- **Multilingual scripts** — Arabic RTL, CJK vertical text, Devanagari complex glyph stacks.
- **Low-resolution scans** — sub-150-DPI inputs degrade everything.
- **Form orientation** — scans rotated 90°/180°; need preprocessor.

---

### 3.10 Multilingual & Low-Resource NLP

> The single most under-served axis of NLP. Where senior NLP engineers earn their keep.

**The reality.** A model trained primarily on English does not "just work" in Hindi, Yoruba, or Tagalog. Production teams that ship globally need *deliberate* multilingual strategy.

**Sub-strategies:**

| Strategy | What it resolves | Pros | Cons |
|---|---|---|---|
| **Translate-and-process** | English-only model + translate I/O | Easy to bolt on | Translation errors compound; cost |
| **Multilingual pretrained backbone (mBERT, XLM-R, NLLB)** | Cross-lingual transfer | One model, many languages | Quality variance; "curse of multilinguality" |
| **Per-language fine-tune** | Domain + language | Best per-language quality | N models to maintain |
| **Adapters (MAD-X, LoRA per language)** | One backbone + small per-language adapter | Best of both | Adapter selection at inference |
| **Cross-lingual transfer (zero/few-shot)** | New language with no labels | Cheap | Quality cliff for low-resource |
| **Backtranslation (MT for data)** | Generate parallel data | Free training data | Quality bounded by MT |
| **Self-supervised + small fine-tune (wav2vec2 → ASR)** | Low-resource ASR | Massive WER gains | Needs unlabeled audio |
| **Frontier LLM zero-shot** | Many languages on day 1 | Easy | Quality cliff at low-resource |

**The "curse of multilinguality."** Adding more languages to a single model improves transfer but eventually *degrades* per-language quality vs a monolingual model of the same size. Bigger models partly fix it (XLM-R XL, NLLB-3.3B), but the tradeoff is real.

**Tokenizer trap (revisit §2.2 with deep-funnel implications).** A "multilingual" model with a tokenizer trained mostly on English tokenizes Hindi/Burmese/Khmer terribly — every byte is its own token, latency 4×, context window 4× smaller. Tokenizer choice **is** a multilingual decision.

**Production hygiene for multilingual systems:**
- Per-language eval set, *human-curated*, refreshed.
- Per-language quality dashboard with regression alerts.
- Tokenizer efficiency check (tokens per word) per language.
- Code-switching test set.
- Romanization normalization (Hindi in Devanagari vs Latin).
- Honest quality reporting — better to ship "we don't support X" than ship X badly.

**Failure modes:**
- "We launched in 30 languages" — 24 of them are unusably bad and the team doesn't know.
- Translation pivot through English silently mangles low-resource pairs.
- Tokenizer trained on English fragments low-resource scripts to the point of unusability.
- Code-switched inputs (Hinglish, Spanglish, Singlish) treated as the worst-case of either language.
- Cultural/locale mismatch — date formats, addresses, name order.

---

### 3.11 Semantic Parsing & Text-to-X (SQL, Code, Structured Output)

> Turn natural language into a precise machine-readable form.

**Sub-problems:**

| Task | Output |
|---|---|
| **Text-to-SQL** | SQL query against a schema |
| **Text-to-code** | Code in target language |
| **Text-to-API** | Function call with args |
| **Text-to-structured output (JSON, schema)** | Validated typed output |
| **Slot filling** | Form values |
| **Question parsing for KG** | SPARQL or graph query |
| **Math word problem → expression** | Computable expression |

**Architectural families:**

| Family | Pros | Cons |
|---|---|---|
| **Grammar-constrained generation** | Output guaranteed valid | Grammar maintenance |
| **LLM + JSON schema constraint (Outlines, XGrammar, SGLang)** | Modern default | Prompt-tuning still needed |
| **Function calling / tool calling** | First-class API for structured output | Limited to defined schemas |
| **LLM + executor feedback (self-correction)** | LLM tries, runs, fixes | Latency + cost multiply |
| **Retrieval over schema / few-shot examples** | Grounding for unfamiliar schemas | Retrieval quality matters |
| **Specialized small model (T5-SQL, CodeT5)** | Fast, cheap | Bounded domain |
| **Sketch + fill-in (sketch-then-realize)** | Decomposes hard tasks | Two-stage complexity |

**Text-to-SQL specifically (the most-studied of these):**

Standard production stack:
1. Schema linking — find which tables/columns matter for the question.
2. Sketch — produce a SQL skeleton.
3. Fill — actual SQL.
4. Execute — run against DB; catch errors.
5. Self-correct on error — feed error back to LLM.

Benchmarks: Spider, Bird, KaggleDBQA. **All saturated; none predict production performance.** Production text-to-SQL is dominated by schema unfamiliarity, ambiguous queries, and dialect-specific SQL features that benchmarks don't cover.

**Failure modes:**
- **Ambiguity** — "show me top customers" — by revenue? count? region?
- **Schema unfamiliarity** — model invents column names.
- **Joins on non-FK columns** — model doesn't realize the natural-key.
- **Dialect mismatch** — generates Postgres on a SQL Server DB.
- **Silent wrong answers** — query runs, returns wrong rows.
- **Validation gaps** — JSON output passes schema, semantically wrong.

---

### 3.12 Code Intelligence

> NLP for code: code completion, code review, code search, code generation, code translation.

**Sub-products:**

| Product | Examples | Notes |
|---|---|---|
| **Inline completion** | Copilot, Cursor Tab, Codeium | Latency-critical: <300 ms TTFT |
| **Code search** | Sourcegraph, GitHub semantic search | Embedding + AST hybrid |
| **Code review / PR comments** | GitHub Copilot for PRs, Diamond | Quality matters more than latency |
| **Agentic coding** | Devin, Cursor Composer, Claude Code | Multi-step; tool use |
| **Code translation** | "Port this from Java to Kotlin" | Long-tail languages tough |
| **Bug fixing** | SWE-bench-style agents | Verifier loop required |
| **Test generation** | EvoSuite-NL, LLM-driven test gen | Coverage + non-trivial tests |

**Sub-techniques specific to code:**

| Technique | Resolves |
|---|---|
| **Fill-in-the-middle (FIM) training** | Inline-completion needs middle-fill, not just suffix |
| **Repository-level retrieval** | Code makes sense only in context |
| **AST-aware embeddings** | Syntactic neighbors; refactor-aware |
| **Tree-sitter parsing** | Language-aware chunking |
| **Type-aware decoding** | Constrained to type-correct outputs |
| **Test execution feedback** | Verifier-in-the-loop |
| **Static-analysis grounding** | Lint + LSP signals as features |

**Failure modes:**
- **Hallucinated APIs** — function doesn't exist in the version installed.
- **Repository context blindness** — completion ignores conventions used elsewhere in repo.
- **Long-context regressions** — bigger context window doesn't always help with code.
- **Test-passing but wrong** — LLM games the test, doesn't fix the bug.
- **Cross-language confusion** — Python idioms in TypeScript code.

---

### 3.13 Knowledge Graphs & Structured Knowledge

> The complement to unstructured NLP. Many products combine both.

**Sub-problems:**

| Task | Notes |
|---|---|
| **KG construction (from text)** | Use IE (§3.3) at scale |
| **Entity linking / disambiguation** | "Apple" — the company or the fruit? |
| **KG completion (link prediction)** | Embedding-based: TransE, RotatE, ComplEx |
| **KG embedding for downstream** | Recommend, search, RAG-grounding |
| **KG-augmented retrieval / Graph RAG** | Multi-hop reasoning |
| **Querying with NL → SPARQL** | See §3.11 |
| **Updating KG with deltas** | Streaming change ingestion |

**When KGs help:**
- Multi-hop queries the LLM gets wrong on flat text.
- Auditability and explainability.
- Stable ID-based references where text varies.
- Cross-document deduplication.

**When they don't:**
- Schema rigidity vs evolving real-world entities.
- Maintenance cost dominates the value at small scale.
- Pure text RAG often sufficient up to a point.

---

## Part 4 — System Architecture: The Production NLP Stack

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       USER INPUT (text / audio / doc)                     │
└──────────────────────────────────────────────────────────────────────────┘
              │
              ▼
   ┌─────────────────────┐    ┌─────────────────────┐
   │  Frontend processing │    │  Audio frontend     │
   │  - normalization     │    │  - resample 16 kHz  │
   │  - language detect   │    │  - VAD              │
   │  - tokenization      │    │  - feature extract  │
   └─────────────────────┘    └─────────────────────┘
              │                          │
              ▼                          ▼
   ┌──────────────────────────────────────────────────┐
   │  Cheap pre-filters (rules, regex, classifiers)   │
   │  Trash 90% of inputs without ever calling a model│
   └──────────────────────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────────┐
   │  Light models (DistilBERT-class, FastText, CTC)  │
   │  CPU-served, <50 ms                              │
   │  Handle the bulk of routine traffic              │
   └──────────────────────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────────┐
   │  Heavy models (BERT-large, Conformer, LLM)       │
   │  GPU-served, 50–500 ms                           │
   │  Only the tail that the light model can't decide │
   └──────────────────────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────────┐
   │  Frontier LLM / VLM (GPT-4o, Claude, Gemini)     │
   │  500 ms–5 s; expensive                           │
   │  For hardest queries / open-domain reasoning     │
   └──────────────────────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────────┐
   │  Postprocessing (ITN, punctuation, redaction,    │
   │  detokenization, formatting)                     │
   └──────────────────────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────────┐
   │  Logging / observability / online metrics        │
   │  Per-language slices; quality dashboards;        │
   │  drift detection                                 │
   └──────────────────────────────────────────────────┘
```

**Sub-systems worth knowing:**

| Sub-system | Tools |
|---|---|
| Tokenizer infra | HF `tokenizers`, SentencePiece, tiktoken |
| Embedding service | sentence-transformers, ONNX, TEI (Text Embeddings Inference) |
| Vector store | Faiss, Milvus, Qdrant, pgvector, Pinecone |
| Inverted index | Elasticsearch, OpenSearch, Tantivy, Vespa |
| Inference serving | Triton, TGI, vLLM, SGLang, TF Serving |
| ASR runtime | NVIDIA Riva, Whisper.cpp, faster-whisper, k2/icefall |
| TTS runtime | Coqui, Bark, ElevenLabs API, OpenAI TTS API |
| Streaming | gRPC bidi, WebRTC, WebSockets |
| Pipelines | NVIDIA NeMo, FastAPI / serve, Ray Serve |
| Annotation tooling | Label Studio, Prodigy, Argilla, Doccano |
| Eval frameworks | LM-Eval Harness, Helm, internal custom |

---

## Part 5 — Foundations to Backfill (Just-in-Time)

> Don't read these front-to-back. Pull each in when a problem in Part 3 forces you to.

### 5.1 Linguistic foundations

- Phonetics, phonology, morphology, syntax, semantics, pragmatics — at a working level, not a PhD level.
- IPA basics for ASR/TTS work.
- Morphologically rich languages (Turkish, Finnish, Arabic) — agglutination, inflection.
- Syntax: dependency parsing, constituency parsing, POS.
- Why CJK / Thai / Khmer break "split on whitespace" assumptions.

### 5.2 Statistical NLP (the parts still relevant)

- N-gram language models, smoothing (Kneser-Ney).
- HMMs, CRFs (still used for some IE tasks).
- TF-IDF, BM25 (the latter is *the* classical retrieval baseline).
- Information theory: entropy, perplexity, KL.
- Feature hashing.

### 5.3 Sequence-model fundamentals

- RNN, LSTM, GRU — and *why* they failed at long range.
- CTC loss — the alignment trick that made E2E ASR work.
- Attention mechanism: Bahdanau (additive), dot-product, scaled dot-product.
- Encoder–decoder with attention.
- Pointer networks, copy mechanisms.

### 5.4 Modern NLP architectures

- **Transformer** (encoder-decoder, encoder-only BERT-style, decoder-only GPT-style). See [`../../ML-Implementations/transformers/`](../../ML-Implementations/transformers/).
  - [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  - [Improving Language Understanding by Generative Pre-Training](https://mng.bz/x2qg)
  - [InstructGPT](https://arxiv.org/abs/2203.02155)
- BERT pretraining (MLM, NSP); RoBERTa improvements; DeBERTa; ELECTRA.
- T5 / UL2 unified text-to-text.
- Decoder-only LLMs and their training stack — see DL-Engineer roadmap §3.5.
- Sequence-to-sequence with attention for MT.
- RNN-T and Conformer for ASR.

### 5.5 Evaluation literacy

| Task | Metric |
|---|---|
| Classification | Precision, recall, F1, PR-AUC, ROC-AUC, calibration |
| NER | Span-level F1, exact vs partial match |
| MT | BLEU, chrF, COMET, BLEURT, MQM |
| ASR | WER, CER, real-time factor |
| Summarization | ROUGE, BERTScore, faithfulness scores |
| Generation | Human MOS, win-rate, LLM-judge |
| Retrieval | Recall@k, MRR, nDCG |

### 5.6 Tokenization deep-dive

- **BPE, BPE-dropout, byte-level BPE**
  - [Implementing A Byte Pair Encoding (BPE) Tokenizer From Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html)
  - BPE Implementations by OpenAI
    - [gpt-2](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
    - [tiktoken](https://github.com/openai/tiktoken)
- WordPiece (BERT-family).
- SentencePiece (T5, Llama, NLLB).
- Tokenizer training: vocab size choice, character coverage, byte fallback.
- Unicode normalization (NFC/NFKC), zero-width chars, homoglyph attacks.

### 5.7 Programming & tooling

- Python idiomatic; NumPy / PyTorch / JAX; HF Transformers / Datasets / Tokenizers.
- spaCy / Stanza for classical pipelines.
- Audio: librosa, torchaudio, ffmpeg.
- Document/PDF: pdfplumber, pypdfium2, Surya, PaddleOCR.
- Search/IR: pyserini, Faiss, Elasticsearch.
- ONNX export, quantization (INT8, INT4, GGUF).

### 5.8 Adjacent foundations

- Linear algebra, probability, calculus — at the level of [DL Engineer roadmap §5.1](../DL-Engineer-Roadmap/README.md#51-math-the-parts-that-actually-bite).
- Classical ML basics — [`../../ML-Implementations/basics/`](../../ML-Implementations/basics/).
- Optimizers — [`../../ML-Implementations/optimizers/`](../../ML-Implementations/optimizers/).

---

## Part 6 — Interview Signal: What NLP Teams Actually Probe

### Coding (universal first round)

- Tokenize a string with BPE — implement merges from scratch.
- Implement attention forward.
- Implement beam search.
- BM25 from scratch (no library).
- Write a streaming-aware sliding-window classifier.
- Edit distance / Levenshtein, LCS.
- Trie for autocomplete.

### NLP system design

- "Design a content moderation system for 1B messages/day across 50 languages."
- "Design a real-time meeting transcription + summarization product."
- "Design a multilingual semantic search over enterprise docs."
- "Design a voice assistant under 1 s end-to-end latency."
- "Design a translation memory + neural MT hybrid for a CAT tool."
- "Design a NER pipeline for medical records."

What they're checking: do you reason about *latency, cost, multilingual coverage, evaluation, drift, cascades, fallbacks* — together.

### Domain depth

- "Why is BLEU not a great metric? What do you use instead?"
- "Walk me through CTC vs RNN-T vs attention encoder-decoder for ASR."
- "What's the difference between BPE and WordPiece?"
- "How does FlashAttention help BERT inference?"
- "Why is NER harder in Chinese than English?"
- "How does Whisper handle silence/music robustly?"
- "What's the curse of multilinguality?"

### Statistical / linguistic reasoning

- "Why does perplexity go up when vocabulary grows? Should you care?"
- "When would you choose CRF over a Transformer for NER?"
- "How do you detect concept drift in a deployed classifier?"
- "Two NER models have the same F1 — how do you decide which to ship?"

### "Should I just use an LLM?" judgment

- "Your team built a fine-tuned BERT classifier. PM says 'just use GPT-4 instead.' What do you say?"
- "When would you choose retrieval over fine-tuning?"
- "Customer wants real-time translation. LLM API is 800 ms. What do you do?"

### Failure-mode questions

- "Your spam classifier had 99% F1 in dev, 60% in prod. Why?"
- "Whisper is hallucinating on long silent inputs. Diagnose and fix."
- "Translation system outputs the wrong target language sometimes. Why?"
- "Your search reranker improved offline but online CTR dropped. Walk me through investigation."

---

## Part 7 — Suggested Project Track

> Pick three. Real shipped projects beat any reading list.

### Project 1 — Hybrid search over a real corpus

Pick a corpus you care about (Wikipedia, GitHub, ArXiv, your company docs). Build:
- BM25 baseline (no library — implement scoring).
- Dense embedding retrieval (BGE / E5 / GTE) with Faiss HNSW.
- Hybrid via Reciprocal Rank Fusion.
- Cross-encoder reranker on top-50.
- Eval suite: 200 hand-labeled queries, recall@k, nDCG, MRR.
- Latency / cost dashboard.

### Project 2 — Multilingual classifier with cascade

Pick a real classification task (toxicity, intent, news topic) across 5+ languages. Build:
- fastText baseline.
- Fine-tuned XLM-R or DistilBERT-multilingual.
- LLM zero-shot baseline (frontier API).
- LLM-as-labeler → distill into small model.
- A cascade routing inputs by confidence.
- Per-language quality dashboard.

### Project 3 — End-to-end ASR fine-tune

Pick Whisper-small or wav2vec2-base. Build:
- Domain fine-tune on a real audio dataset (e.g. Common Voice for a specific language, or a podcast).
- Streaming inference with VAD.
- WER eval on a held-out test set; per-condition slicing (clean, noisy, accented).
- INT8 quantization; throughput benchmark.

### Project 4 — Document AI extraction pipeline

Pick a document type (invoices, papers, contracts). Build:
- Layout detection (Surya or DocLayout-YOLO).
- OCR (Surya / PaddleOCR).
- Field extraction with LayoutLMv3 fine-tune *and* a VLM (GPT-4o or Qwen-VL) version.
- Hybrid: rule + small model + VLM-fallback.
- Per-field F1; cost-per-doc; latency p99.

### Project 5 — Faithful summarization with eval

Pick a long-doc dataset (CNN/DM at the easy end, GovReport / ScrollsQA / arxiv at the hard end). Build:
- Map-reduce summarization with a fine-tuned BART / Pegasus.
- Long-context LLM version.
- A faithfulness scorer (FactScore-style).
- An LLM-judge calibrated against 100 human-rated examples.
- Pareto plot: cost vs ROUGE vs faithfulness.

### Optional Project 6 — Voice agent under a budget

Build a voice agent that:
- Streams ASR (Whisper.cpp or Conformer-CTC).
- Calls a small LLM with RAG.
- Streams TTS (Bark / Piper).
- Endpointing and barge-in.
- Total budget <1.5 s end-to-end p95.

---

## References

### Books worth reading
- *Speech and Language Processing* — Jurafsky & Martin (the field's textbook; free online; updated for transformer era). Already in [`../../Books/`](../../Books/).
- *Natural Language Processing with Transformers* — Tunstall, von Werra, Wolf.
- *Foundations of Statistical Natural Language Processing* — Manning & Schütze (the classical bible).
- *Speech and Language Processing* chapters 16–18 for the ASR/TTS section specifically.
- *Build a Large Language Model from Scratch*

### Courses
- [**Stanford CS224N**](../../ML-Courses/Stanford/CS224N_NLP_with_Deep_Learning/) — NLP with Deep Learning (the canonical modern course).
- [**Stanford CS336**](../../ML-Courses/Stanford/CS336_Language_Modeling_from_Scratch/) — Language Modeling from Scratch.
- **Stanford CS224S** — Spoken Language Processing.
- **CMU 11-411 / 11-611 / 11-737** — NLP and multilingual NLP.
- **Hugging Face NLP Course** — practical, free, frequently updated.
- **fast.ai** — code-first NLP.

### Foundational papers (read when relevant)
- *Attention Is All You Need* (2017).
- BERT (2018), RoBERTa, DeBERTa, ELECTRA.
- T5 (2019), UL2.
- BPE (Sennrich 2016), SentencePiece (Kudo 2018).
- BART, Pegasus.
- mBERT, XLM, XLM-R, NLLB, M2M-100.
- wav2vec2, HuBERT, Whisper, Conformer, RNN-T (Graves).
- VITS, FastSpeech 2, Tacotron 2, NaturalSpeech.
- LayoutLMv3, Donut, Pix2Struct, Surya.
- DPR (Karpukhin), ColBERT, SPLADE, BM25 (Robertson).
- ESM (protein LM as adjacent example).

### Practical / industrial
- Hugging Face blog (model releases, fine-tune recipes).
- Sebastian Ruder's blog and *NLP-Progress*.
- Lilian Weng's blog (`lilianweng.github.io`).
- spaCy and Explosion's blog (production NLP).
- Anthropic / OpenAI / Google / Meta tech reports (multilingual, alignment, audio).

### Tools to actually know
- Hugging Face: `transformers`, `datasets`, `tokenizers`, `accelerate`, `peft`, `trl`.
- spaCy + Stanza for classical pipelines.
- Sentence-Transformers, FlagEmbedding (BGE), LLaMA-Index, LangChain.
- OpenSearch / Elasticsearch / Vespa / Pyserini.
- Faiss, Milvus, Qdrant, pgvector.
- NVIDIA NeMo (ASR / TTS / NLP).
- faster-whisper, whisper.cpp, k2/icefall.
- Coqui-TTS, Piper, Bark, OpenVoice.
- Surya, PaddleOCR, LayoutLMv3, Donut.
- Outlines / XGrammar / SGLang for structured output.

### Reference implementations in this repo
- [`../../ML-Implementations/transformers/`](../../ML-Implementations/transformers/) — self-attention, multi-head attention, transformer block.
- [`../../ML-Implementations/basics/`](../../ML-Implementations/basics/) — classical ML (logistic regression, softmax classifier).
- [`../../ML-Implementations/optimizers/`](../../ML-Implementations/optimizers/) — SGD, Adam, Adagrad.
- [`../../ML-Implementations/losses/`](../../ML-Implementations/losses/), [`../../ML-Implementations/metrics/`](../../ML-Implementations/metrics/) — building blocks for any NLP training loop.
