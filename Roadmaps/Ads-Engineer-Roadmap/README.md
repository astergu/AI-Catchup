# Ads Engineer / Scientist Roadmap — Top-Down

Most ads roadmaps are encyclopedias of skills: "learn SQL, learn Spark, learn XGBoost, learn DCN..." This one is the inverse. **Start from what ads systems actually look like in production at Google / Meta / Amazon / TikTok / ByteDance / Pinterest / Snap. Decompose each system into the sub-problems it solves. Surface the latency, calibration, and privacy constraints that drive every model architecture decision. Pull in theory only when a problem demands it.**

---

## Table of Contents

- [Part 0 — How to Read This Roadmap](#part-0--how-to-read-this-roadmap)
- [Part 1 — What Ads Engineers Are Actually Hired To Do](#part-1--what-ads-engineers-are-actually-hired-to-do)
- [Part 2 — The Constraints That Drive Every Decision](#part-2--the-constraints-that-drive-every-decision)
- [Part 3 — Industrial Solutions, Decomposed](#part-3--industrial-solutions-decomposed)
  - [3.1 The Ad Serving Funnel — the organizing principle](#31-the-ad-serving-funnel--the-organizing-principle)
  - [3.2 Candidate Retrieval (Recall)](#32-candidate-retrieval-recall)
  - [3.3 CTR / CVR Prediction (Ranking)](#33-ctr--cvr-prediction-ranking)
  - [3.4 Multi-Task Learning (CTR + CVR + …)](#34-multi-task-learning-ctr--cvr--)
  - [3.5 Calibration — why AUC alone doesn't ship](#35-calibration--why-auc-alone-doesnt-ship)
  - [3.6 Auctions & Pricing](#36-auctions--pricing)
  - [3.7 Bid Optimization & Budget Pacing](#37-bid-optimization--budget-pacing)
  - [3.8 Deep-Funnel Events & Value Optimization](#38-deep-funnel-events--value-optimization)
  - [3.9 Causal Measurement & Incrementality](#39-causal-measurement--incrementality)
  - [3.10 The Privacy-First Era (Post-Cookie, Post-ATT)](#310-the-privacy-first-era-post-cookie-post-att)
  - [3.11 Cold Start & Exploration](#311-cold-start--exploration)
  - [3.12 Ad Fraud, Quality & Brand Safety](#312-ad-fraud-quality--brand-safety)
  - [3.13 GenAI in Ads](#313-genai-in-ads)
- [Part 4 — System Architecture: The Request Flow](#part-4--system-architecture-the-request-flow)
- [Part 5 — Foundations to Backfill (Just-in-Time)](#part-5--foundations-to-backfill-just-in-time)
- [Part 6 — Interview Signal: What Ads Teams Actually Probe](#part-6--interview-signal-what-ads-teams-actually-probe)
- [Part 7 — Suggested Project Track](#part-7--suggested-project-track)
- [Career Progression](#career-progression)
- [References](#references)

---

## Part 0 — How to Read This Roadmap

For each industrial solution:

1. **The real problem** — what business KPI is being moved, and why naive approaches fail.
2. **Sub-solutions** — the components a production system decomposes into.
3. **What each resolves** — and the new problem it introduces.
4. **Pros / cons / when it's the wrong tool**.
5. **Hardware / latency footprint**.
6. **Failure modes you'll see in production**.

The order is the order industry actually thinks: *funnel → ranking → auction → measurement → privacy*. Theory (FTRL math, attention internals, GBDT splits) is in Part 5 — pull it in only when something in Part 3 forces you to.

---

## Part 1 — What Ads Engineers Are Actually Hired To Do

The job title varies (Ads ML Engineer, Ranking Scientist, Applied Scientist, ML Engineer–Monetization), but the **product surface is small**:

| Surface | Concrete examples | Primary business KPI | Primary technical KPI |
|---|---|---|---|
| **Search ads** | Google Search Ads, Amazon Sponsored Products, Bing | Revenue per search, advertiser ROAS | pCTR calibration, P99 ranking latency |
| **Feed / display ads** | Meta Newsfeed, TikTok in-feed, Snap, Pinterest, X | Revenue per session, ad load efficiency, retention | Calibrated pCTR × pCVR, user-experience guardrails |
| **Video ads** | YouTube, TikTok, Reels, IG Stories | Revenue per impression, completion rate | View-through rate, retention guardrails |
| **Shopping / product ads** | Amazon, Google Shopping, Instacart | GMV-attributable revenue, advertiser ROAS | pCVR, bid efficiency |
| **App-install ads** | Meta AAA, Google App Campaigns, Unity, AppLovin | Cost per install, ROAS d7/d30 | LTV prediction, attribution accuracy |
| **Deep-funnel / value optimization** | Meta AEO + VO, Google ACi/ACe/ACa, TikTok AEO/VBO, App-of-the-day re-engagement | tROAS, post-install LTV, repeat-purchase rate | Deep-event prediction, value regression, sparse-event calibration |
| **DSP / programmatic** | The Trade Desk, DV360, Criteo | Win rate × ROI | Bid shading accuracy, latency budget |
| **SSP / exchange** | Google AdX, OpenX, Magnite | Take rate × fill rate | Auction throughput, floor price optimization |
| **Brand & measurement** | Nielsen, brand-lift studies | Incremental brand metrics | Causal lift estimation |

### What you're actually graded on

Three numbers control almost every decision:

1. **Revenue** — eCPM, ARPU, total ad revenue. The line on the P&L.
2. **Advertiser ROAS sustainability** — if advertisers don't make their money back, they leave. This is the #1 underrated metric. A short-term revenue lift that destroys ROAS kills the marketplace.
3. **User experience** — guardrails on ad load, latency, irrelevance, fatigue. Sustained ad revenue requires a healthy non-ad product.

A senior ads engineer can **trade these off explicitly**: "this change is +1.2% revenue but -0.4% advertiser ROAS, here's the marketplace simulation that says it's still net-positive at 12 months." Junior engineers optimize AUC. Senior engineers optimize the ecosystem.

### The eCPM identity (the most important equation in ads)

```
eCPM = pCTR × pCVR × bid × 1000                                    # action-based (CPA)
eCPM = pCTR × pCVR × pDeep × bid × 1000                            # deep-funnel optimization
eCPM = pCTR × pCVR × E[value | conversion] × tROAS × 1000          # value-based bidding (VO/VBB)
eCPM = bid × 1000                                                  # brand / CPM buys
ranking_score = eCPM × quality_multiplier                          # auction input
```

Almost every model an ads engineer builds feeds one of `pCTR`, `pCVR`, `pDeep` (deep-funnel event probability — purchase, signup, D7 retention, repeat purchase…), `E[value]` (expected purchase value), `bid`, or `quality`. **If `pCTR` is miscalibrated by 10%, every auction outcome is wrong by 10% in expectation** — even if AUC didn't move. The same is true for `pDeep` and `E[value]`, only worse, because deep-funnel labels are 10–100× sparser. That is why calibration ([§3.5](#35-calibration--why-auc-alone-doesnt-ship)) is not an afterthought, and why deep-funnel optimization ([§3.8](#38-deep-funnel-events--value-optimization)) is its own first-class problem.

---

## Part 2 — The Constraints That Drive Every Decision

Ads ML is shaped by a brutal set of constraints. Every architectural choice is downstream of these.

### 2.1 Latency budgets

| Surface | End-to-end p99 | Ranking model budget |
|---|---|---|
| Search ads | 30–80 ms | 5–20 ms |
| Feed ads (mobile) | 50–150 ms | 10–30 ms |
| Display / RTB exchange | **<100 ms hard cap (OpenRTB)** | 10–50 ms |
| Programmatic DSP bid | <80 ms | <30 ms (or you lose the auction) |
| Video pre-roll | 200–500 ms | 50–200 ms |

**Implication:** GPU inference is rare in the hot path. Most ranking models run on CPU with vectorized ops. When a model is too big for CPU, you cascade: a small model in the hot path, a heavier model offline pre-computing scores. **"Just use a bigger neural net"** is not an answer if it busts the latency budget.

### 2.2 Throughput & embedding-table reality

Aggregate platform QPS at hyperscaler ads (rough):
- Google Search Ads / Meta: peak millions of ad-rank QPS.
- Amazon Sponsored Products: hundreds of thousands of QPS.
- Mid-size DSP / SSP: tens of thousands of QPS.

The dominant memory cost in ads ML is not the dense network — **it's the embedding tables**. Production-scale parameters:

| Item | Cardinality | Embedding dim | Memory at FP16 |
|---|---|---|---|
| User IDs | 100M–1B | 32–64 | 6–128 GB |
| Ad/creative IDs | 10M–100M | 32–128 | 0.6–25 GB |
| Publisher / placement IDs | 1M–10M | 16–64 | small |
| Cross IDs (user × category) | 1B–10B | 16 | tens to hundreds of GB |

Total embedding-table size at scale: **multiple TB**. This is sharded across parameter servers (TF PS, ByteDance Monolith, Meta DLRM hash tables) or across in-memory key-value stores. **Almost no production ads model fits on a single machine** — distributed embedding lookup is the hot path.

### 2.3 Label sparsity and delay

| Signal | Rate | Delay |
|---|---|---|
| Impression | 100% | <1 s |
| Click | 0.05–5% (display) / 3–10% (search) | seconds |
| Add-to-cart | 0.1–2% | minutes to hours |
| Purchase / install | 0.05–1% | hours to **days** |
| App d7 / d30 LTV | even sparser | 7–30 days |

Two consequences:
- **Class imbalance**: 1:1000 positives is normal. Loss reweighting, negative sampling, focal loss exist for this reason.
- **Delayed feedback**: when you train at hour `t`, conversions for impressions at `t-2h` haven't happened yet. The training label is wrong. This is its own subfield ([§3.7](#37-bid-optimization--budget-pacing) / delayed feedback modeling).

### 2.4 Sample selection bias (the classic ads-ML problem)

```
                impression → click → conversion
       observe everything    only-if-clicked    only-if-clicked
```

If you train CVR on clicked impressions only, the model doesn't see the population it'll be applied to (all impressions). Predictions are biased on impressions where users would never click. **ESMM (Alibaba 2018)** addresses this directly by training over the entire impression space — the canonical answer to the canonical interview question.

### 2.5 Position / presentation bias

Top-position ads get ~3–5× the CTR of bottom-position ads, *for the same ad on the same query*. Train naively on click logs and the model "learns" that ads near the top are good — confounding presentation with relevance. Treatments: position features, IPS (inverse propensity scoring), counterfactual learning, randomization slots.

### 2.6 Marketplace and feedback loops

Your ranking model is a participant in an auction. If your model predicts a higher pCTR for ad A, advertiser A's spend grows, their ROAS shifts, their bidder reacts, the equilibrium moves. **Static offline AUC does not capture this.** Marketplace simulators, switchback experiments, and long-horizon A/B tests exist because of it.

### 2.7 Privacy and signal loss

The 2021–2025 transition has reshaped the field:
- **Apple ATT (App Tracking Transparency, 2021)** — IDFA opt-in collapsed mobile attribution; SKAdNetwork is now the dominant iOS attribution channel and provides aggregate, delayed, noisy postbacks.
- **Chrome Privacy Sandbox** — Topics, Protected Audience (FLEDGE), Attribution Reporting API. Cookie deprecation has slipped multiple times but the direction is fixed.
- **GDPR / CCPA** — consent gates, right-to-be-forgotten, age-of-consent, data-minimization.

Implication: **the rich user-level features and per-impression labels that ads ML was built on are eroding**. New models must operate on aggregated, noised, on-device, or federated signals. This is the single biggest *technical* shift in ads ML this decade.

### 2.8 What it costs to lose

Ads engineers think about cost in three currencies:
- **Revenue / day** — a 0.5% revenue regression on a $100B/yr business is $1.4M/day. Releases are gated on metrics for a reason.
- **Advertiser ROAS** — degraded ROAS today → advertiser churn next quarter.
- **Compute** — training clusters of hundreds of GPUs running for days; serving fleets in the tens of thousands of cores. Single-digit-percent efficiency wins are real money.

---

## Part 3 — Industrial Solutions, Decomposed

### 3.1 The Ad Serving Funnel — the organizing principle

> **Every ads ML system is a funnel.** Internalize the funnel and the rest of the roadmap snaps into place.

```
billions of ads
       │
       ▼  Targeting / eligibility (rule-based, hard filter)
millions
       │
       ▼  Retrieval / candidate generation (embedding ANN, two-tower)
1K–10K
       │
       ▼  Pre-ranking (small model: lightweight DNN, distilled ranker)
100–1K
       │
       ▼  Ranking (full DNN: CTR × CVR × … with rich features)
10–100
       │
       ▼  Auction (eCPM, GSP / VCG / first-price, reserve prices)
1–10
       │
       ▼  Re-ranking / blending (diversity, business rules, slate-level)
final slate
       │
       ▼  Pacing & delivery control
       │
       ▼  Render → log → train next model
```

**Why a funnel and not one big model?** Latency. A 50-ms budget can't run a heavy DNN over a million candidates. Each stage is a tradeoff between *recall* (don't drop the right ad) and *cost* (compute per request).

**Stage-by-stage industry choices:**

| Stage | Common production choice | Why |
|---|---|---|
| Targeting | Rule engine + bloom filters / inverted index | Hard constraints (geo, language, time) must be exact and fast |
| Retrieval | Two-tower DNN + ANN (HNSW, ScaNN, Faiss) | Embedding similarity at scale |
| Pre-rank | Small DNN, distilled from ranker | Score 1K with 1-2 ms budget |
| Ranking | Multi-task DNN (CTR + CVR + … via MMoE / PLE), DCN-V2, DIN/DIEN | Highest-cost per candidate, justified by funnel narrowing |
| Auction | GSP (search), first-price + bid shading (display), VCG (rare in practice) | Mechanism-design tradeoffs |
| Blending | Slate-level optimizer, MMR / determinantal point process for diversity | Single-item ranking misses slate effects |
| Pacing | PID controller / model-predictive control over budget remaining | Smooth spend across day; avoid early budget exhaustion |

### 3.2 Candidate Retrieval (Recall)

> **You can't rank what you don't retrieve.** Retrieval miss caps everything downstream.

**The real problem.** Of billions of eligible ads, fetch the few thousand most likely to be relevant in <10 ms.

**Sub-solutions:**

| Method | What it resolves | Pros | Cons |
|---|---|---|---|
| **Inverted index (term match, tag match)** | Hard targeting + keyword recall | Sub-ms latency; deterministic | Misses semantic matches; rule explosion |
| **Collaborative filtering (item-item, user-item co-counts)** | Behavioral co-occurrence | Strong baseline, cheap | Cold start; popularity bias |
| **Two-tower DNN + ANN** | Semantic recall at scale | Generalizes; embeddings reusable | Training/serving consistency hard; ANN tuning is real work |
| **YouTubeDNN / DSSM-style** | First production "deep retrieval" pattern | Reference architecture | Now superseded but still common |
| **Multi-interest (MIND, ComiRec)** | One user has many interests; a single embedding loses signal | Better recall on diverse-interest users | Higher serving cost (k vectors per user) |
| **Tree-based deep retrieval (TDM, JTM)** | Beam-search through a learned tree | Logarithmic scoring; tighter than ANN on some metrics | Engineering complexity; only Alibaba runs this at scale |
| **Generative retrieval (semantic IDs, TIGER)** | Generate item IDs autoregressively | Hot research direction; integrates with LLMs | Production case studies still emerging |
| **Graph-based retrieval (PinSage, GraphSAGE)** | Items connected via user/content graph | Captures structural similarity | Graph-build pipeline cost |

**ANN index choice (the operational decision):**

| Index | Memory | Recall@k | Build | Serving |
|---|---|---|---|---|
| HNSW (Faiss / hnswlib) | high | very high | slow | fast queries |
| IVF + PQ | low | medium | medium | fast, approximate |
| ScaNN (Google) | medium | very high | medium | fast on TPU/CPU |
| Flat | low | exact | none | only viable <100K items |

**Typical production ratio:** retrieval recall@1000 in the 70–90% range against a ground-truth oracle is the bar. Below that, ranking can't recover; above that, more retrieval recall doesn't help (ranking is the bottleneck).

**Hardware footprint.** Embedding tables for users + items often dominate retrieval memory. ANN indexes for 100M+ items live across many machines — sharded by hash, with fallback shards for redundancy. Latency target: <10 ms for embedding lookup + ANN search combined.

**Failure modes:**
- **Train/serve mismatch** — embedding produced at train time differs from serving (vocab drift, hashing mismatch, normalization).
- **Cold-start ad** — new ad has no behavioral signal; pure-content tower must carry weight.
- **Recall misses unfiltered category** — a hard rule applied post-retrieval kills 90% of retrieved candidates and the funnel starves downstream.
- **Position-bias in retrieval logs** — the labels you train on are themselves biased.

### 3.3 CTR / CVR Prediction (Ranking)

> The single most-studied area in ads ML. Almost every paper in this space is *one architectural variation* on the basic embedding → interaction → DNN pattern.

**The real problem.** Predict, for a given (user, context, ad) tuple, the calibrated probability of click and (separately) of conversion. Used directly in `eCPM = pCTR × pCVR × bid`.

**Sub-architecture decomposition:**

| Block | What it does | Common variants |
|---|---|---|
| **Embedding layer** | Map sparse IDs → dense vectors | Standard learned, hash-based (HashEmbed), product-quantized |
| **Feature interaction** | Capture useful crosses without hand-engineering all of them | FM, DCN, DCN-V2, AutoInt, xDeepFM, NFM |
| **Sequence / behavior modeling** | Encode user history | DIN (target attention), DIEN (GRU + AUGRU), BST, SIM (long-sequence retrieval over history) |
| **Multi-task heads** | Joint CTR + CVR + dwell + … | MMoE, PLE, ESMM |
| **DNN trunk** | Generalize | MLP [256, 128, 64] is the unironic industry default |
| **Calibration head** | Make outputs match observed rates | Platt scaling, isotonic, temperature scaling |

**Architecture cheat-sheet** (each links to a working implementation in this repo):

| Model | Year · Lab | Core idea | When to reach for it |
|---|---|---|---|
| **Logistic Regression + manual crosses** | classical | linear in (one-hot ⊗ one-hot) | The honest baseline. Required at all interviews. |
| **GBDT (XGBoost / LightGBM)** | 2014–17 | Tree boosting on dense + low-cardinality features | Strong on tabular; weak on raw high-cardinality IDs |
| **GBDT + LR / DeepFM-style hybrid** | Facebook 2014 | Tree leaves as features into LR | Bridges classical → deep; still in production at smaller shops |
| **[Wide & Deep](../../ML-Implementations/ads/wdl.py)** | Google 2016 | Memorize (wide) + generalize (deep) | The starting point for any new ads-ranking project |
| **[DCN](../../ML-Implementations/ads/dcn.py) / DCN-V2** | Google 2017 / 2021 | Explicit polynomial crosses with O(d) params | Replaces hand-crafted crosses; V2 is the production default |
| **[DeepFM](../../ML-Implementations/ads/deepfm.py)** | Huawei 2017 | FM (2nd-order) + DNN, shared embeddings | DeepFM ≈ Wide & Deep with auto-crosses; the most popular open-source baseline |
| **[NFM](../../ML-Implementations/ads/nfm.py)** | NUS 2017 | Bi-Interaction pooling → DNN | Parameter-efficient when DNN must be deep |
| **[xDeepFM](../../ML-Implementations/ads/xdeepfm.py)** | Microsoft 2018 | Vector-wise CIN; bounded-degree explicit interactions | When you want explicit + implicit crosses; CIN is heavy |
| **[AutoInt](../../ML-Implementations/ads/autoint.py)** | BIT 2019 | Multi-head self-attention over field embeddings | Interpretable interaction weights; competitive with DeepFM |
| **[DIN](../../ML-Implementations/ads/din.py)** | Alibaba 2018 | Target-attention over user behavior | Anywhere a long behavior sequence is available |
| **[DIEN](../../ML-Implementations/ads/dien.py)** | Alibaba 2019 | GRU + AUGRU; interest evolution | DIN + temporal dynamics; non-trivial to train stably |
| **SIM** | Alibaba 2020 | Long-sequence (1000s of events) two-stage retrieval | When you need to reason over a year of user history |
| **BST (Behavior Sequence Transformer)** | Alibaba 2019 | Transformer over user behavior | Direct competitor to DIEN; usually simpler |
| **[ESMM](../../ML-Implementations/ads/esmm.py)** | Alibaba 2018 | CTCVR = CTR × CVR; train on full impression space | The canonical fix for CVR sample-selection bias |
| **[MMoE](../../ML-Implementations/ads/mmoe.py)** | Google 2018 | Shared experts + per-task gates | First-line multi-task ranking |
| **[PLE](../../ML-Implementations/ads/ple.py)** | Tencent 2020 | Task-specific + shared experts; progressive extraction | When MMoE shows negative transfer |
| **DCN-V2 + MMoE / PLE hybrid** | folklore | Cross network + multi-task heads | The production architecture at most shops in 2024 |

**Production architecture in 2024–2025 (rough composite):**
```
Sparse IDs ─┐
            ├─► shared embedding ─► DCN-V2 cross + DNN trunk ─► PLE/MMoE multi-task heads ─► [pCTR, pCVR, pView, pDwell] ─► calibration ─► auction
Dense  ─────┘                                          ▲
                                                       │
User behavior seq ─► DIN/DIEN/BST attention ──────────┘
```

**Pros / cons summary:**

| Family | Pros | Cons |
|---|---|---|
| Tree-based (XGBoost/LightGBM) | Robust on tabular, fast to iterate, calibrated by default | Weak on raw high-cardinality IDs; no GPU; no embeddings |
| Wide & Deep / DeepFM | Memorization + generalization | Wide side is hand-crafted; DeepFM auto-crosses but only 2nd-order |
| DCN-V2 | Explicit higher-order crosses with bounded params | Needs careful regularization; structure choice (parallel vs stacked) matters |
| Sequence models (DIN/DIEN/BST/SIM) | Capture user dynamics | Heavier; more sensitive to behavior log quality |
| Multi-task (MMoE/PLE) | Share signal across CTR/CVR/dwell | Negative transfer when tasks conflict; gating tuning |
| ESMM | Solves SSB elegantly | Tightly couples CTR and CVR; doesn't help if CTR head is weak |

**Hardware footprint.** Production ranking model: 10s of GBs of embeddings + a small dense net (~10s of MB). Trained on parameter-server clusters or with sharded embedding (TorchRec, DLRM, Monolith). Served on CPU with vectorized embedding lookup; GPU only for heavier sequence components or when batch sizes amortize transfer cost.

**Failure modes:**
- AUC up, calibration broken — ranking unchanged but auction outcomes worse.
- Negative transfer in multi-task — CVR head pulls CTR down.
- Long-tail collapse — model serves popular ads well, ignores tail (fixes: frequency-aware sampling, tail-aware loss).
- Hashing collisions in ID embeddings — silent quality loss as cardinality grows.

### 3.4 Multi-Task Learning (CTR + CVR + …)

> One ad request → many things to predict. Train them jointly or you waste signal.

**The real problem.** Search/feed ads care about CTR (will you click), CVR (will you convert), dwell time, hide/skip rate, video completion, store visit, purchase value. Each has its own loss, label sparsity, and bias. Training one model per task duplicates infrastructure and ignores shared signal.

**Sub-solutions:**

| Pattern | What it resolves | Pros | Cons |
|---|---|---|---|
| **Hard parameter sharing (shared trunk → task heads)** | Cheap, simple multi-task | Strong baseline; minimal infra | Negative transfer when tasks disagree |
| **MMoE** | Tasks pick different experts via gates | Explicitly models task relationships | Gates can collapse; training instability |
| **PLE** | Adds task-specific experts to MMoE | Reduces negative transfer | More parameters; more complex |
| **ESMM** | Trains CVR over full impression space via CTCVR factorization | Solves sample-selection bias canonically | CVR is "leaked" through CTR; tightly coupled |
| **ESM2 / multi-step ESMM** | Generalize ESMM beyond click→convert | Models full funnel | More heads; data sparsity at deep funnel stages |
| **Auxiliary losses (predict next click, etc.)** | Regularize main task | Free signal | Can interfere with primary task |
| **Uncertainty-weighted multi-task (Kendall et al.)** | Per-task loss-weighting auto-learned | Reduces hyperparameter sweep | Numerically tricky |

**ESMM in plain English** — write it on the whiteboard:

```
y_ctr     ∈ {0,1}     — clicked (observed everywhere)
y_ctcvr   ∈ {0,1}     — clicked AND converted (observed everywhere)
y_cvr_obs ∈ {0,1}     — converted | clicked (observed only when clicked)

Naive: train CVR model on clicked subset → biased on impressions where p(click)=0
ESMM:  p_ctcvr(x) = p_ctr(x) × p_cvr(x), train both heads on the full impression space
       Loss = BCE(y_ctr, p_ctr) + BCE(y_ctcvr, p_ctr × p_cvr)
       p_cvr is learned implicitly without observing CVR labels on unclicked impressions.
```

This is one of those patterns that, once internalized, changes how you read every ads paper.

### 3.5 Calibration — why AUC alone doesn't ship

> A model can be a great ranker and a terrible probability estimator. The auction needs the latter.

**Why this matters.** `eCPM = pCTR × pCVR × bid × 1000`. If `pCTR` is uniformly 2× too high, `eCPM` is 2× too high, the auction picks the wrong winner, advertisers overpay, ROAS collapses. AUC measures *order*; calibration measures *level*. **Both must be right.**

**Sub-solutions:**

| Method | What it resolves | Pros | Cons |
|---|---|---|---|
| **Platt scaling** | Logistic recalibration on a held-out set | Simple, parametric | Assumes a specific shape |
| **Isotonic regression** | Non-parametric monotone recalibration | Flexible | Needs more data; can overfit |
| **Temperature scaling** | One-parameter softening | Trivial; works when miscalibration is "too peaky" | Can't fix shape mismatch |
| **Bias correction for negative sampling** | Negative downsampling biases logits | Closed-form correction `p' = p / (p + (1-p)/w)` | Required if you downsampled negatives |
| **Calibration-aware training (focal loss, label smoothing)** | Train calibrated from the start | No post-hoc step | Indirect; harder to tune |
| **Field-aware calibration** | Calibration drift varies by segment (geo, device, advertiser) | Catches segment miscalibration | Many small calibrators; data-hungry per segment |
| **Online recalibration** | Distribution drift across days | Tracks reality | Risk of feedback loops |

**The "negative sampling correction" is an interview classic** — if you trained with downsampling factor `w` (kept all positives, kept `1/w` of negatives), the model's predicted probabilities are biased high. Closed-form correction: `p_true = p / (p + (1-p) × w)`.

**Field-aware miscalibration** — globally calibrated, locally broken. A model can have great ECE overall but be 30% off for "new advertisers in Brazil at 3am." Production teams build calibration dashboards sliced by 5–10 axes.

**Failure modes:**
- AUC unchanged, calibration drifted → revenue tanks silently.
- Calibration measured on click logs only → CVR calibration on full impression space is unmeasured.
- Calibration "fixed" with isotonic on a small sample → overfit, worse in production than no calibration.

### 3.6 Auctions & Pricing

> The mechanism converts ML predictions into outcomes. Get the ML perfect and the auction wrong, and you still lose money.

**The real problem.** Multiple advertisers want the same impression. Choose a winner and a price in a way that is (ideally) truthful, revenue-good, and stable.

**Mechanisms (and where they live):**

| Mechanism | Used by | What it resolves | Tradeoffs |
|---|---|---|---|
| **First-price** | Display, RTB exchanges (post-2018 industry shift), Google Ads (post-2019) | Revenue maximization in single-shot context | Bidders must bid-shade; not truthful |
| **Second-price (Vickrey)** | Theoretical baseline; rare in pure form | Truthful in single-item case | Multi-slot extension is non-truthful (GSP) |
| **GSP (Generalized Second-Price)** | Historic search ads | k-slot version of 2nd-price | Non-truthful but simple; was the industry default for 2 decades |
| **VCG (Vickrey-Clarke-Groves)** | Some Facebook ad surfaces | Truthful in multi-item | Externalities are hard to compute; revenue often lower than GSP |
| **First-price + bid shading** | Display / RTB now | Recover truthful behavior in first-price | Bid shading is its own ML problem |
| **Reserve prices (static, dynamic)** | Everywhere | Floor on revenue per impression | Set too high → lose auctions; too low → leave money on table |
| **Header bidding** | Web display | Multi-SSP parallel auction | Latency complexity; waterfall vs unified auction |
| **Auction with quality multiplier** | Google Ads | Penalize low-quality ads | Quality definition is subjective and political |

**The three things every senior ads engineer should be able to derive on a whiteboard:**
1. **Why second-price is truthful** (single item): proof by case analysis on what happens if you over- or under-bid.
2. **Why GSP is not truthful** (multi-slot): a counterexample where bidder benefits from misreporting.
3. **Why first-price needs bid shading**: in a first-price auction, you should bid below your value; the optimal shading depends on your belief about competitor distribution.

**Bid shading as an ML problem (DSP-side).** Given features about the auction (publisher, time, audience, recent win-loss history), predict the minimum bid that wins. Trade off win-rate vs surplus. Models: gradient-boosted trees, lightweight DNNs, contextual bandits.

**Reserve-price optimization.** Per-query (or per-segment) optimal floor that maximizes expected revenue given the bid distribution. Classical paper: Myerson 1981. Production: estimate bid distribution per segment, derive reserve, recompute periodically. Too aggressive → empty auctions; too soft → lost revenue on inelastic demand.

**Failure modes:**
- Reserve set globally, not per-segment → leaves money on inelastic queries.
- First-price launched without shading → advertiser overspend, ROAS death spiral.
- Quality multiplier gamed by advertisers → low-quality ads with high CTR dominate.

### 3.7 Bid Optimization & Budget Pacing

> The advertiser tells the platform "spend $1000 today, get me as many conversions as possible." The platform must translate that into per-impression bids.

**Sub-problems:**

| Problem | Industry solution | Pros | Cons |
|---|---|---|---|
| **Autobidding (tCPA, tROAS, max conversions, max conversion value)** | The platform sets bids on advertiser's behalf, given a goal — including value-based goals (see [§3.8](#38-deep-funnel-events--value-optimization)) | Captures advertiser intent; lifts marketplace performance | Loss of advertiser control; "black box" complaints |
| **Budget pacing** | PID / model-predictive controller adjusts bid multiplier as day progresses | Smooth spend; avoid 9am budget exhaustion | Pacing can fight bidding; oscillations |
| **Delayed-feedback modeling** | Some conversions take days; train-time labels are wrong | Importance-weighted loss / survival models / DELAYED loss | Adds variance; eval is harder |
| **LTV (lifetime value) prediction** | Bid on long-term value, not just immediate conversion | Aligns advertiser with quality users | Sparse, noisy, long-horizon labels |
| **Cross-publisher / cross-device attribution** | A user touched 4 ads before converting — credit assignment | Multi-touch attribution; data-driven (Shapley, Markov) | Attribution = causal inference, see §3.9 |
| **Bid landscape modeling** | Distribution of competing bids per segment | Enables shading, reserves, win-rate prediction | Distribution is non-stationary |

**The autobidder as an ML system** (this is increasingly the most valuable thing on the platform side):

```
goal (tCPA) + budget + remaining time + bid landscape ─► policy ─► per-impression bid
```

Modern autobidders are reinforcement-learning or contextual-bandit systems. The reward is delayed (conversions arrive hours later). The action space is continuous (the bid). Off-policy evaluation is mandatory because you can't A/B test millions of bidding strategies in production.

**Pacing controllers** (control theory shows up here):
- **Throttling** — probabilistically skip eligible auctions to slow spend.
- **Bid multipliers** — scale bids by `min(1, budget_remaining / expected_remaining_spend)`.
- **Model-predictive control** — short-horizon optimization re-run every minute.

**Failure modes:**
- Budget exhausted at 11am on Black Friday → advertiser furious, missed peak demand.
- Autobidder converges to a degenerate strategy (bid 0.01 everywhere) on cold-start campaigns.
- Pacing controller oscillates → "lumpy" spend rather than smooth.
- Delayed feedback ignored → autobidder optimizes for immediate conversions, undervalues high-LTV users.

### 3.8 Deep-Funnel Events & Value Optimization

> An install is not a customer. A click is not a purchase. The deepest events in the funnel — the ones that actually move advertiser P&L — are also the rarest, the most delayed, and the hardest to attribute. This is where modern performance advertising lives.

**The funnel below the click:**

```
impression → click → install → open (D0) → tutorial / signup → first action
                                              │
                                              ▼
                                     add-to-cart / first purchase
                                              │
                                              ▼
                                  D1 / D7 / D30 retention
                                              │
                                              ▼
                                  repeat purchase / subscription / LTV
```

**Why each new event downstream is roughly 10× harder than the one before it:**

| Event | Typical rate (post-prior-event) | Delay | Why hard |
|---|---|---|---|
| Click | 1–10% | seconds | — |
| Install | 5–30% of clicks | minutes | Attribution windows, click vs view |
| D0 open | 60–90% of installs | minutes–hours | Mostly a quality check |
| Tutorial complete / signup | 30–70% of D0 | minutes | First behavioral signal of intent |
| First purchase / subscription | 1–10% of installs | hours–days | Sparse, delayed, label noise |
| D7 retention | 10–30% of installs | **7 days** | The label literally doesn't exist for a week |
| D30 retention | 5–15% of installs | **30 days** | Severe delayed feedback |
| Repeat purchase / LTV | 0.5–5% of installs | weeks–months | Even sparser; long-tail-dominated |

**Compounded sample-selection bias.** ESMM solved one stage (CVR | click). Deep-funnel modeling has *four to six* such stages, each filtering the population further. A naive "predict purchase given install" model trained on installed users is biased on the impression population. Same problem as CVR, only worse.

**Industry product names to recognize (and what they actually are):**

| Product | Platform | What it optimizes for | Notes |
|---|---|---|---|
| **AEO (App Event Optimization)** | Meta | A specific post-install event (level reached, purchase, signup) | Standard in 2018+ |
| **VO (Value Optimization)** | Meta | Predicted value (purchase amount), not count | Bid scaled to predicted value × tROAS |
| **AAA (Advantage+ App Campaigns)** | Meta | AEO + VO + automation | Subsumes manual configuration |
| **App Campaigns for Installs (ACi)** | Google | Install volume at tCPA | Entry-level app campaign |
| **App Campaigns for Engagement (ACe)** | Google | Re-engaging existing installs for in-app actions | Reengagement product |
| **App Campaigns for Actions (ACa)** | Google | Specific in-app conversion at tCPA / tROAS | Deep-funnel + value |
| **tROAS bidding** | Meta / Google / TikTok | Bid s.t. expected value × tROAS = bid | The bid itself is value-aware |
| **VBB (Value-Based Bidding)** | Generic term | Bid proportional to predicted value | Industry standard in 2024 |
| **Reengagement / retargeting** | Everywhere | Existing users → repeat purchase | Different population from acquisition |

**Sub-solutions:**

| Sub-solution | What it resolves | Pros | Cons |
|---|---|---|---|
| **Multi-stage funnel model (extends ESMM)** | Compound sample-selection bias across click → install → deep event | Single coherent model; shared embeddings | Negative transfer between stages; deep-stage signal is drowned out |
| **ESM2 / multi-step ESMM** | Generalize ESMM to N-step funnels | Probabilities multiply; trained on full impression space | Conditional probabilities at deep stages are tiny → numerical issues |
| **Two-tower: P(event) × E[value \| event]** | Splits classification from regression | Each head is simpler; calibrate independently | Joint distribution may be lost |
| **Direct value regression head** | Predict expected purchase value | Optimizes the dollar, not the count | Long-tail distribution; whales dominate; needs log-transform / quantile loss |
| **Quantile / log-normal value heads** | Real purchase distributions are fat-tailed | More robust than MSE on raw value | Choosing the right transform matters |
| **Curriculum learning (click → install → purchase)** | Deep events have ~zero gradient if trained from scratch | Stable training; gradual unlocking | More pipeline complexity |
| **Auxiliary deep-funnel losses** | Borrow strength from sparser tasks | Regularizes and shares signal | Can drag down primary task |
| **Hazard / survival models for retention** | "D7 retention" is censored time-to-event data | Statistically principled | Engineering effort; less common in production |
| **Sample reweighting (importance / focal / class-balanced)** | Positives are 0.1% of impressions | Recovers gradient signal | Distorts calibration unless corrected |
| **Negative downsampling + logit correction** | Compute and storage savings | Standard ads technique | Required correction at serving |
| **Delayed-feedback modeling (DFM)** | Conversion happens *after* training cutoff | Bias-corrected loss | Adds variance; eval is harder |
| **Importance-weighted loss for late conversions** | "This impression's label may still flip" | Reduces stale-label bias | Hyperparameter-heavy |
| **Conversion-window-aware training** | Different events have different windows (24h, 7d, 30d) | Aligns label with reality | Per-window pipelines |
| **LTV regression (continuous, long-horizon)** | Optimize for high-value users not just any user | Aligns with advertiser P&L | Labels arrive over weeks; high variance |
| **Staged LTV models (pLTV-d7, pLTV-d30, pLTV-d180)** | Long-horizon labels arrive too late to train on | Predict early surrogate; refine later | Bias if surrogate ≠ true LTV |
| **Reengagement-specific models** | Existing-user population is distinct from acquisition | Better targeting for repeat actions | Two model stacks to maintain |
| **Modeled conversions / modeled value** | SKAN reports are aggregated and noisy on iOS | Recovers signal lost to privacy | "Modeled" = guessed; advertisers skeptical |

**The canonical multi-stage funnel model (interview-ready derivation):**

```
y_click   ∈ {0,1}   — clicked? (observed everywhere)
y_install ∈ {0,1}   — installed? (observed everywhere; via SDK / postback)
y_purch   ∈ {0,1}   — purchased? (observed everywhere if attributed)
v         ∈ R+      — purchase value (observed iff y_purch = 1)

Train heads jointly on the full impression space:

  p_click_install_purch(x) = p_click(x) × p_install|click(x) × p_purch|install(x)

Loss = BCE(y_click,                    p_click)
     + BCE(y_click·y_install,          p_click × p_install|click)
     + BCE(y_click·y_install·y_purch,  p_click × p_install|click × p_purch|install)
     + λ · MSE_or_quantile(v,           E[v|purch], mask=y_purch)

eCPM (VO) = p_click × p_install|click × p_purch|install × E[v|purch] × tROAS × 1000
```

This is roughly the shape of Meta-AEO / Google-ACa-style multi-stage models. Variants differ in how they share embeddings, where they apply MMoE/PLE-style multi-task heads, and how they handle the value-regression tail.

**Value optimization specifically:**

Value optimization (VO / VBB) is the move from "maximize conversions" to "maximize purchase value." Two production variants:

1. **Bid = expected value** — `bid = E[v | impression] × tROAS = p_conv × E[v | conv] × tROAS`. Auction picks highest expected-value impression.
2. **Bid scaled by value bucket** — common when value is predicted into discrete buckets (notably for SKAN, where iOS conversion values are 0–63).

Why VO is harder than count-based optimization:
- **Tail-dominated value distribution.** Top 1% of purchasers can be 30%+ of GMV. MSE on raw value is dominated by whales; the model overfits a few outliers.
- **Calibration on value, not just probability.** Underpredicting value by 20% means underbidding by 20% on every high-value user, who go to competitors.
- **Time-aggregated value vs single-event value.** Is "value" first-purchase basket size, or D30 cumulative spend, or D180 LTV? Each is a different model with different label-arrival latency.
- **Whale segmentation.** Many production systems separately model `P(whale)` and `E[v | non-whale]`; unifying them is fragile.

**Reengagement campaigns** are a separate product line:
- Population is *existing installs / users*, not new acquisition.
- Features that work for cold acquisition (lookalike, demographic) lose value; in-app behavior (recency of last open, in-app event history, prior purchase value) dominates.
- Attribution is harder — the user already had the app; what counts as ad-driven re-engagement?
- Holdout experiments are particularly important (high incrementality risk: ad shown to user who was about to come back anyway).

**Hardware and label-pipeline footprint.**
- **Label backfill pipelines are days deep.** Training data for a D30 model uses impressions from ≥30 days ago. Pipelines must support late-arriving labels and re-stitching.
- **Storage:** every impression must be retained long enough to join with deep-event labels (often ≥90 days at scale, hundreds of TB to PB).
- **Streaming joins:** Kafka topics for impression / install / event have to be co-partitioned and time-aligned for online feature computation.
- **Postback handling on iOS:** SKAN postbacks arrive on a randomized delay (24–72h) with a single 6-bit conversion value per install. Pipelines must decode operator-defined conversion-value schemas back into events.

**Failure modes (the ones that actually ship):**
- Trained the deep-event head on only-installed users → biased on the impression population (compounded SSB).
- Optimized for `pPurch` and call it done, while VO competitors are bidding on `E[value]` → lose all the high-value users to them, ROAS looks fine on count but advertiser GMV craters.
- Value head fits the long-tail whales perfectly, ignores the 95% of users who buy normally → average user underbid, marketplace share drops.
- Delayed-feedback ignored → model converges to a stale equilibrium; new advertisers' campaigns "die" before their conversions arrive.
- Reengagement campaign bids equally on users who would have come back anyway → high attributed conversions, near-zero incremental conversions, advertiser cancels.
- iOS deep-funnel modeling treats SKAN postbacks as unbiased ground truth → calibration drift as conversion-value schema, redownload behavior, or null-CV rates change.
- Conversion-window mismatch — model trained on 7-day attribution serving impressions tracked under 1-day attribution → silent ~30% miscalibration.

### 3.9 Causal Measurement & Incrementality

> Did the ad cause the conversion, or would the user have converted anyway? **This is the question that distinguishes senior ads engineers.**

**The real problem.** Click-based attribution gives credit to every ad in the path, including ads that didn't move the needle. Advertisers are increasingly skeptical and demand *incremental* lift measurement.

**Sub-solutions:**

| Method | What it resolves | Pros | Cons |
|---|---|---|---|
| **A/B testing (user-level holdout)** | Gold-standard causal measurement | Unbiased if randomization is clean | Requires opportunity cost (un-served users); slow |
| **Geo experiments** | When user-level holdout impossible (TV, OOH, brand) | Works at market level | Less power; geo confounders |
| **Switchback experiments** | Marketplaces where treatment leaks across users | Captures equilibrium effects | Complex analysis; switching costs |
| **Synthetic control** | Pre/post comparison with constructed counterfactual | Works when randomization isn't possible | Strong identifying assumptions |
| **PSA (placebo / public-service ad)** | Show an unrelated ad to control group | Cleaner than no-ad holdout | Costs impressions; advertisers hate it |
| **Ghost ads** | Log "would have shown" rather than show | Cheap; no opportunity cost | Requires same ranking model in counterfactual |
| **Conversion lift studies (Meta CLS, Google Conversion Lift)** | Productized incrementality | Trustworthy when run right | Needs scale to detect typical lifts |
| **MMM (Marketing Mix Modeling)** | Cross-channel, cross-platform attribution | Survives privacy changes; aggregate data | Slow (weekly), low-resolution |
| **MTA (Multi-Touch Attribution)** | Per-user path credit assignment | Granular | Identifier-dependent; broken by ATT/cookie loss |
| **IPS / counterfactual learning to rank** | Train ranker to optimize causal lift directly | Handles position bias | Propensities are noisy; high variance |
| **Doubly-robust estimators** | Off-policy evaluation | More efficient than IPS alone | Complex; nuisance models can fail |

**Why this is hard.** Naive lift = `treatment_conv_rate - control_conv_rate`. But:
- Selection bias: did the platform pick the right users to show ads to?
- Network effects: control users are exposed to treatment-influenced peers.
- Multi-touch paths: a user sees ad A, then ad B, then converts.
- Cannibalization: a paid ad got the click an organic result would have gotten.

**Industry signal:** Meta's "Conversion Lift" team, Google's "Geo Experiments / Brand Lift", Amazon's "Incrementality" all are senior-IC hubs. Causal inference fluency is a fast-track skill.

### 3.10 The Privacy-First Era (Post-Cookie, Post-ATT)

> The ground under ads ML is shifting. Models built on user-level signals are being rebuilt on aggregated, noised, on-device, or federated signals.

**The transition:**

| Era | What was true | What's changing |
|---|---|---|
| **Pre-2021** | 3rd-party cookies + IDFA → cross-device, cross-publisher user graph | — |
| **iOS 14.5+ (Apr 2021)** | ATT prompt; <30% IDFA opt-in | Mobile attribution collapsed; SKAN became primary iOS channel |
| **Chrome 2024+** | Privacy Sandbox phase-in (Topics, Protected Audience / FLEDGE, Attribution Reporting API) | Cookie deprecation slipped multiple times but trajectory is fixed |
| **GDPR / CCPA / CPRA / PIPEDA / etc.** | Consent gates, DSARs, age gating | Tightening, not loosening |

**Privacy-preserving ML toolkit:**

| Technology | What it resolves | Pros | Cons |
|---|---|---|---|
| **SKAdNetwork (SKAN 4)** | iOS attribution without IDFA | Apple-blessed | Aggregated, delayed, noised; conversion values quantized |
| **Privacy Sandbox: Protected Audience (FLEDGE)** | On-device retargeting | No cross-site identifier | Limited cohort size; auction logic constrained |
| **Privacy Sandbox: Topics API** | Coarse interest signals from browser | No cross-site tracking | Far less granular than 3p cookies |
| **Privacy Sandbox: Attribution Reporting API** | Aggregate / event-level reports | Replaces cookie-based attribution | Reports are noised, capped, delayed |
| **Differential privacy** | Train models / publish stats with formal privacy bounds | Mathematically rigorous | Adds noise; epsilon budget management |
| **Federated learning (cross-device)** | Train on devices; aggregate gradients | Raw data never leaves device | Slow; system complexity; limited model size |
| **On-device inference** | Score on device with only public signals | No data exfiltration | Model size constrained; update cadence slow |
| **Server-side tagging (CAPI)** | Replace browser pixels with server-to-server | More reliable; first-party data | Lower-fidelity than cookie-based |
| **Secure aggregation / SMPC / TEE** | Cross-org joint training without data sharing | Enables data partnerships | High infra cost; small ecosystem |
| **Modeled / inferred conversions** | Conversions API + modeling for the gap | Recovers some signal | Modeled = guessed; advertiser pushback |
| **Conversion modeling (Meta, Google)** | Aggregate-data ML models predict missing conversions | Salvages measurement | "Trust us" problem |

**Deep-funnel ([§3.8](#38-deep-funnel-events--value-optimization)) under privacy constraints — the brutal table:**

| Constraint (iOS / SKAN 4) | Effect on deep-funnel modeling |
|---|---|
| Single 6-bit `conversionValue` per install (0–63) | Operator-defined schema must encode count, value bucket, and event type into 64 slots — every choice trades off |
| Three postback windows (0–2 days, 3–7 days, 8–35 days) with hierarchical CV | Deep events outside window are invisible; D30 LTV is unrecoverable from SKAN alone |
| Random delay (24–72h) on postbacks | Training-data labels are even later; pipelines must wait or modeled-fill |
| `null` CV when `crowdAnonymityTier` too low | Small advertisers lose deep-event signal entirely |
| No user-level join across postbacks | LTV / repeat-purchase modeling on iOS is structurally aggregated |

**Implications for ads ML in 2024–2025:**
- User-level features become weaker → contextual features (page content, query, device class, geo) get re-prioritized.
- Cohort and segment models replace individual user models.
- Aggregate / population-level training becomes more common.
- Causal measurement (§3.9) is harder *and* more valuable.
- Deep-funnel and value optimization on iOS become *modeling problems* (Meta's "Modeled Conversions", Google's "Modeled Conversions" / "consent mode" gap-filling), not measurement problems.
- On-device / federated models are no longer research curiosities — they are roadmap items.

### 3.11 Cold Start & Exploration

> The model has never seen this advertiser / user / item. What do you do for the first 1000 impressions?

**Sub-solutions:**

| Method | What it resolves | Pros | Cons |
|---|---|---|---|
| **Content-based features** | Use ad creative / item metadata when ID has no history | Always works; good prior | Generally weaker than learned ID embeddings |
| **Hierarchical embeddings** | Roll up to category/brand/segment when ID is cold | Smooth backoff | Hierarchy must exist and be clean |
| **ε-greedy / random exploration** | Force exposure to explore | Trivial | Wasteful; bad UX |
| **Upper Confidence Bound (UCB)** | Explore items with high uncertainty | Principled; well-studied | Variance estimation in deep models is hard |
| **Thompson sampling** | Sample from posterior, act greedily | Strong empirically | Posterior is hard for big models |
| **Contextual bandits (LinUCB, Neural Bandits)** | Personalized exploration | Captures context | Off-policy evaluation is its own subfield |
| **Counterfactual learning to rank with logged data** | Reuse logged data with propensity weights | No fresh exploration cost | Needs propensities; high variance |
| **Bootstrap / dropout uncertainty** | Cheap proxy for posterior | Easy to plug in | Calibration of uncertainty is shaky |

**The exploration vs exploitation dilemma — and why production teams under-explore.** Exploration costs money *now* for information gains *later*. Most teams under-invest in exploration because the tradeoff is hard to quantify and easy to lose A/B tests on. Senior ICs make the case for exploration budgets explicitly.

### 3.12 Ad Fraud, Quality & Brand Safety

> The flip side of growth: clicks that aren't real, ads that hurt the brand, content next to the wrong context.

**Sub-areas:**

| Area | Industrial solution | Notes |
|---|---|---|
| **Invalid Traffic (IVT)** | Anomaly detection on click patterns; bot fingerprinting; co-visitation graphs | IAB / MRC-defined; advertisers won't pay for IVT |
| **Click farms / incentivized installs** | Behavioral consistency models; install-to-action funnel checks | App-install is the worst-affected vertical |
| **Ad creative review** | ML classifiers (image, text, video) + human review queue | Two-stage: cheap filter → expensive judge |
| **Brand safety / suitability** | Page / video content classification; GARM categories | Adjacency to violence, hate speech, etc. |
| **Policy enforcement** | Hard rules + ML for grey areas | Compliance + appeals process |
| **Ad fatigue / frequency capping** | Per-user impression counting; recency features | Trades short-term revenue for long-term retention |

**Why this matters even if you're not on a trust team.** Advertiser ROAS evaluation includes IVT-discounted conversions. A revenue lift that increases IVT is fake.

### 3.13 GenAI in Ads

> The emerging frontier. Most of these are early; expect rapid change.

**Active areas (2024–2025):**

| Area | Status | Notes |
|---|---|---|
| **Generative ad copy / headline / image / video** | Production at Google PMax, Meta Advantage+, Amazon, TikTok | Quality varies; brand-safety review pipelines required |
| **Audience expansion via LLM-derived embeddings** | Production | LLM embeddings of ad copy + landing page replace hand-tuned audience signals |
| **Query understanding / intent classification with LLMs** | Production | Replacing or augmenting older NLU stacks |
| **LLMs as feature extractors** | Production | Off-the-shelf or fine-tuned LLMs produce dense features for downstream ranking |
| **LLMs as rankers** | Research / early production | Cost and latency are the blocker; distillation makes it tractable |
| **Generative retrieval (semantic IDs)** | Research → early production | TIGER-style models; potentially replaces ANN |
| **Conversational shopping / advertising** | Early product | Perplexity-style answers + sponsored results; mechanism design unsettled |
| **Auto-creative-optimization (DCO + GenAI)** | Production | Ad variants generated and tested at scale |
| **LLM judges for creative quality / brand safety** | Production | Replacing or supplementing classifier sandwiches |

**The honest summary:** GenAI is currently augmenting the ads stack at the edges (copy gen, query understanding, embeddings) but has *not yet* replaced the dense-DNN ranker in the hot path at any large platform. Watch this space — it will move.

---

## Part 4 — System Architecture: The Request Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                            USER LOADS PAGE                                 │
└───────────────────────────────────────────────────────────────────────────┘
              │
              ▼
   Ad request (with context: user, query, page, device, geo, time)
              │
              ▼
   ┌──────────────────┐
   │  Targeting /     │  rule engine, eligibility, frequency caps
   │  eligibility     │
   └──────────────────┘
              │
              ▼
   ┌──────────────────┐    Feature Store (online):
   │  Retrieval       │◄── user embedding (Redis / RocksDB)
   │  (ANN, k=1K-10K) │    item embedding (sharded KV)
   └──────────────────┘    realtime counters (Flink)
              │
              ▼
   ┌──────────────────┐
   │  Pre-ranking     │  small distilled model, k=100-1K
   └──────────────────┘
              │
              ▼
   ┌──────────────────┐    Feature Store (online):
   │  Ranking         │◄── dense user features
   │  CTR×CVR×…       │    contextual features
   │  k=10-100        │    creative features
   └──────────────────┘
              │
              ▼
   ┌──────────────────┐
   │  Auction         │  eCPM = pCTR × pCVR × bid × quality
   │  (GSP/1P/VCG)    │  reserve prices, bid shading
   └──────────────────┘
              │
              ▼
   ┌──────────────────┐
   │  Re-ranking /    │  diversity (MMR/DPP), business rules
   │  Slate blending  │  ad-load policy
   └──────────────────┘
              │
              ▼
   ┌──────────────────┐
   │  Pacing          │  PID controller, throttling
   └──────────────────┘
              │
              ▼
   ┌──────────────────┐
   │  Render          │  creative server, viewability tracking
   └──────────────────┘
              │
              ▼  Impression / Click / Conversion log
   ┌──────────────────────────────────────────────────────────────────────┐
   │  Streaming (Kafka)  ─►  Flink (feature aggregation)  ─►  Feature    │
   │                                                          Store      │
   │                     ─►  Storage (Parquet / lake)                    │
   │                     ─►  Training pipeline (Spark / SQL → Trainer)   │
   │                     ─►  Causal eval (A/B platform)                  │
   └──────────────────────────────────────────────────────────────────────┘
              │
              ▼
   ┌──────────────────┐  shadow → canary 1% → 10% → 50% → 100%
   │  Model rollout   │  guardrail metrics gate every step
   └──────────────────┘
```

**Sub-systems worth knowing in depth:**

| Sub-system | Industry tools |
|---|---|
| **Feature store** | Feast (OSS), Tecton, Meta FBLearner FS, Uber Michelangelo Palette, Airbnb Zipline |
| **Streaming features** | Kafka + Flink (most common), Kinesis + KCL, Pulsar |
| **Training data lake** | Parquet/ORC on S3/GCS/HDFS, partitioned by hour/day/surface |
| **Distributed training** | TF Distributed (PS), PyTorch DDP/FSDP, TorchRec, Horovod, Ray Train |
| **Embedding sharding** | TorchRec, TF Embedding Distribution, ByteDance Monolith, Meta DLRM |
| **Model serving** | TF Serving, TorchServe, ONNX Runtime, NVIDIA Triton, custom C++ |
| **Experimentation platform** | Internal (every big shop has one); OSS: GrowthBook, Optimizely |
| **Workflow orchestration** | Airflow, Dagster, Kubeflow Pipelines |

**Latency-critical contracts you'll be asked about:**
- Feature store online read: <5 ms p99.
- Embedding lookup (ID → vector): <2 ms per batch.
- Ranking model inference for k=100 candidates: <20 ms p99 on CPU.
- Auction logic: <5 ms.
- End-to-end ad-rank: <50 ms p99 on most surfaces.

---

## Part 5 — Foundations to Backfill (Just-in-Time)

> Don't read these front-to-back. Pull each in when a problem in Part 3 forces you to.

### 5.1 ML / DL fundamentals

- Bias-variance, regularization, cross-validation.
- Logistic regression and its loss derivation. (Required at every ads interview.)
- Gradient boosted trees (XGBoost / LightGBM / CatBoost): how splits work, what makes them fast.
- Optimizers: SGD, Momentum, Adam, [Adagrad](../../ML-Implementations/optimizers/adagrad.py), FTRL — and **why FTRL was the dominant ads optimizer for a decade** (online updates with L1 sparsity → tiny serving footprint).
- Embedding layers, hashing, hash collision tradeoffs.
- DNNs: feedforward, attention, residual / normalization.
- Sequence models for behavior: RNN/GRU/LSTM → Transformer.
- See [`ML-Implementations/ads/`](../../ML-Implementations/ads/) in this repo for runnable references of the canonical architectures.

### 5.2 Statistics & causal inference

- Hypothesis testing (t, χ², Mann-Whitney), sample-size calculation, power.
- Multiple-testing correction (Bonferroni, BH-FDR).
- Sequential testing / always-valid p-values (mSPRT, Howard et al.).
- Variance-reduction (CUPED) for A/B tests.
- Potential outcomes framework, ATT, ATE.
- IV, RDD, diff-in-diff, synthetic control.
- Off-policy evaluation: IPS, doubly robust, FQE.

### 5.3 Auction theory & mechanism design

- Single-item auctions: 1st-price, 2nd-price, sealed-bid.
- Revenue equivalence theorem (Myerson).
- Multi-slot generalizations: GSP, VCG.
- Optimal reserve prices.
- Bidder strategy in repeated auctions.
- Reading: Edelman, Ostrovsky, Schwarz (2007) on GSP; Varian on position auctions.

### 5.4 Distributed systems for ML

- Parameter servers vs all-reduce.
- Embedding sharding: hash, range, hybrid.
- Online vs offline training; lambda architecture; the "feature consistency" problem.
- Ray, Spark MLlib, Dask for parallelism.

### 5.5 Data engineering

- SQL fluency (window functions, query optimization). Non-negotiable.
- Spark/PySpark; partitioning, broadcast joins, skew handling.
- Streaming: Kafka, Flink (event-time vs processing-time, watermarks, exactly-once).
- Columnar storage: Parquet/ORC, partitioning strategies.

### 5.6 Programming

- Python (idiomatic), NumPy, Pandas, PyTorch, TensorFlow.
- C++ (basic) — many serving paths are C++; reading inference code is a real skill.
- Go (optional, common in serving microservices).
- Algorithms / DS at FAANG-coding-interview depth.

---

## Part 6 — Interview Signal: What Ads Teams Actually Probe

### ML system design (the highest-weighted round at most ads orgs)

- "Design a CTR prediction system for a feed surface with 100K QPS and <50ms p99."
- "Design an ad retrieval system over 100M ads with 10ms latency."
- "Design a real-time bidder for a DSP."
- "Build a multi-task ranking model for CTR + CVR + dwell — explain MMoE vs PLE choice."
- "How would you measure ad incrementality?"

What they're checking: do you reason about *funnel, latency, calibration, sample-selection bias, evaluation, marketplace effects* — together, not in isolation.

### Domain depth

- "What's sample selection bias and how does ESMM solve it?"
- "Why is calibration important and how do you correct for negative downsampling?"
- "Explain why GSP is non-truthful with a 2-bidder example."
- "What changed for ads ML after iOS ATT?"
- "How do you handle delayed feedback in CVR training?"
- "What's position bias and how do you correct for it?"

### ML algorithms

- "Derive the logistic regression loss."
- "Explain FTRL and why it was the ads-ranking default for a decade."
- "Compare GBDT with deep ranking models — when does each win?"
- "Explain attention in DIN. Why target-attention rather than self-attention?"
- "What's the difference between MMoE and PLE?"

### Causal / experimentation

- "How would you measure a 0.3% lift on a 100M-DAU surface?"
- "What's CUPED and when does it help?"
- "When can you not run an A/B test? What do you do instead?"
- "Your A/B has p=0.03 and the metric moved 0.05% — would you ship?"

### Coding (LeetCode-style, medium-hard)

- Standard arrays / strings / hash / trees / graphs / DP.
- Pragmatic: can you implement a streaming top-k? An LRU cache for a feature store? A simple ANN?

### Behavioral

- "Tell me about a time you improved a metric by X% — what was your hypothesis, your experiment, your guardrails?"
- "Describe a model you shipped that regressed in production. What did you learn?"
- "How do you communicate a 0.2% lift to a skeptical PM?"

---

## Part 7 — Suggested Project Track

> Real projects > a long reading list. Pick three.

### Project 1 — End-to-end CTR prediction with calibration

Take a public CTR dataset (Criteo, Avazu, KuaiRand). Build:
- A logistic-regression baseline (with feature crosses).
- A GBDT baseline.
- A [DCN-V2](../../ML-Implementations/ads/dcn.py) / [DeepFM](../../ML-Implementations/ads/deepfm.py) / [DIN](../../ML-Implementations/ads/din.py) deep model.
- **Calibration** (Platt + isotonic) with an ECE plot.
- A negative-downsampling experiment with explicit logit correction.
- A short report: AUC, log-loss, ECE, latency-per-prediction. Don't only report AUC.

### Project 2 — Multi-task ranker (CTR + CVR) with sample-selection-bias fix

Same dataset (or any dual-label dataset). Build:
- A naive CVR model trained on clicked-only (the broken baseline).
- An [ESMM](../../ML-Implementations/ads/esmm.py) implementation trained on full impression space.
- An [MMoE](../../ML-Implementations/ads/mmoe.py) or [PLE](../../ML-Implementations/ads/ple.py) extension.
- A counterfactual-evaluation showing how the naive model is biased on unclicked impressions.

### Project 3 — Two-tower retrieval + ranking funnel

Build the funnel end-to-end (small scale):
- Two-tower model for retrieval (item tower + user tower).
- ANN index (Faiss HNSW).
- A heavier ranker over the top-k.
- Measure: recall@k for retrieval, AUC + calibration for ranking, and the **funnel-level** answer-quality metric (NDCG / MRR / hit-rate@k).

### Project 4 — Multi-stage funnel model with value optimization

A deep-funnel ranker on a public dataset with an event chain (Criteo Sponsored Search, Avazu, KuaiRand, or simulated). Build:
- A naive `P(purchase | install)` baseline trained on installed users only — show its bias on the impression population.
- An ESM2-style multi-stage model: `P(click) × P(install | click) × P(purchase | install)` trained on the full impression space.
- A **value head** with log-transform or quantile loss; report calibration on value, not just ECE on probability.
- A `tROAS` bidding simulation: compare count-based eCPM vs value-based eCPM on simulated marketplace lift.
- A delayed-feedback experiment: train with labels that are 0–7 days late; show the bias and a DFM-style correction.

### Optional Project 5 — Auction simulator + bid shading

Simulate a first-price RTB auction with multiple bidders sampling from realistic distributions.
- Implement a naive truthful bidder (loses revenue).
- Implement a bid-shading model (LGBM regressor predicting "minimum winning bid").
- Plot win-rate × surplus tradeoff.
- Bonus: implement a reserve-price optimizer; show its revenue impact.

### Optional Project 6 — Causal lift estimation on a public dataset

Take any treated/control dataset (Criteo Uplift, MovieLens with simulated treatment).
- Naive lift estimate.
- IPS-corrected lift.
- Doubly-robust estimator.
- Show variance vs bias tradeoff.

---

## Career Progression

| Level | Title | Focus | Typical impact |
|---|---|---|---|
| **L3 / E3** | Ads ML Engineer I | Ship features, learn the system, run tests with guidance | Component-level wins (single feature, single model variant) |
| **L4 / E4** | Ads ML Engineer II | Own end-to-end pipelines, ship measurable wins | 1–5% metric lift on a model you own |
| **L5 / E5** | Senior ML Engineer / Scientist | Lead projects, set technical direction for a model family | 5–10% on a major surface, or new system shipped |
| **L6+** | Staff / Principal | Org-level technical leadership, novel research/systems | Defines architectures used across the org |
| **L7+ / Director** | Engineering / Science Director | Multi-team strategy, hiring, partner with PM/Biz | Org direction; multi-quarter bets |

**The career-defining skills** at L5+ are not "knows more ML papers." They are:
1. **Calibration / measurement rigor** — can you tell whether a win is real?
2. **Marketplace thinking** — can you predict second-order effects?
3. **Cross-system reasoning** — funnel, auction, pacing, measurement coupled.
4. **Comm + writing** — can you make the case to skeptical PM/Biz partners?

---

## References

### Books
- *Internet Advertising and the Generalized Second-Price Auction* — Edelman, Ostrovsky, Schwarz (2007). The mechanism-design starting point.
- *Real-Time Bidding* — Jun Wang et al.
- *Designing Data-Intensive Applications* — Kleppmann. Required reading for the system-design half of the job.
- *Trustworthy Online Controlled Experiments* — Kohavi, Tang, Xu. The A/B-testing bible.
- *Causal Inference: The Mixtape* — Cunningham. Practical causal inference.

### Papers (must-read, in the rough order they'll bite you in production)
- **Wide & Deep Learning** — Cheng et al., Google 2016.
- **DeepFM** — Guo et al., Huawei 2017.
- **DCN / DCN-V2** — Wang et al., Google 2017 / 2021.
- **DIN / DIEN** — Zhou et al., Alibaba 2018 / 2019.
- **ESMM** — Ma et al., Alibaba 2018.
- **MMoE** — Ma et al., Google 2018.
- **PLE** — Tang et al., Tencent 2020.
- **xDeepFM** — Lian et al., Microsoft 2018.
- **AutoInt** — Song et al., 2019.
- **SIM** — Pi et al., Alibaba 2020.
- **YouTube Recommendations: Two-tower / DNN candidate generation** — Covington et al., 2016.
- **Ad Click Prediction: a View from the Trenches** — McMahan et al., Google 2013. The most important production-experience paper in ads ML.
- **Practical Lessons from Predicting Clicks on Ads at Facebook** — He et al., 2014. (GBDT + LR.)
- **Counterfactual Reasoning and Learning Systems** — Bottou et al., 2013.
- **Position Bias Estimation for Unbiased Learning to Rank in Personal Search** — Wang et al.
- **CUPED** — Deng et al., Microsoft 2013.

### Conferences
- **RecSys**, **KDD**, **CIKM**, **WWW**, **AdKDD** workshop.

### Industry blogs / engineering
- Google Ads Research, Meta Engineering (Ads), Amazon Science (Ads), Pinterest Engineering, Uber Engineering, DoorDash Engineering, Airbnb Engineering.
- The Trade Desk / Criteo / AppLovin engineering writeups for DSP-side perspective.

### Privacy & post-cookie
- Apple SKAdNetwork docs (SKAN 4).
- Chrome Privacy Sandbox documentation (Topics, Protected Audience, Attribution Reporting API).
- IAB Tech Lab specs.

### Causal inference & experimentation
- Susan Athey & Guido Imbens (Stanford) — modern econometrics with ML.
- *Trustworthy Online Controlled Experiments* (Kohavi et al.) again — read it twice.

### Reference implementations in this repo
- [`ML-Implementations/ads/`](../../ML-Implementations/ads/) — runnable WDL, DCN, DeepFM, DIN, DIEN, AutoInt, xDeepFM, NFM, ESMM, MMoE, PLE on synthetic data with calibration and benchmark numbers.
