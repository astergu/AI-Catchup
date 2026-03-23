# Deep Dive: Designing a Search Ranking System (Amazon Product Search / Google Maps)

*Deep dive into Question #2. A strong candidate distinguishes between semantic and keyword matching and demonstrates knowledge of Learning to Rank.*

---

## The Standard "Very Strong" Answer

### 1. Problem Clarification

Before designing, ask:
- **Scale:** QPS (queries per second)? Size of the item catalog (millions of products)?
- **Latency:** < 100ms? < 500ms? This determines how many ranking stages we can afford.
- **Success Metric:** Click-Through Rate? Conversion (purchase)? Revenue per session? NDCG (offline)?

### 2. Architecture Overview

Search ranking is a retrieval + ranking problem, similar to recommendations but query-driven.

```
[User Query: "red running shoes size 10"]
          |
  [Query Understanding]   → spell correction, synonym expansion, intent classification
          |
  [Multi-Source Retrieval]
    ├─ Inverted Index (BM25/TF-IDF)   → keyword exact match
    └─ Vector Index (ANN)             → semantic embedding match
          |
  [Fusion & Candidate Pool]           ~1,000 candidates
          |
  [Learning to Rank (LTR) Model]      ~50 candidates
          |
  [Re-ranking: Business Rules]        → sold-out items demoted, promoted listings boosted
          |
  [Final Ranked Results]
```

### 3. Query Understanding

Before retrieval, the raw query needs to be processed:
- **Spell Correction:** Handle typos ("runing shoes" → "running shoes").
- **Synonym Expansion:** "sneakers" and "trainers" should match "running shoes." Requires a curated or learned synonym dictionary.
- **Intent Classification:** Is the user looking to *buy* (transactional) or *browse* (informational)? This can change the ranking objective.

### 4. Retrieval: Semantic vs. Keyword

Do not use only one approach:

| Method | Strength | Weakness |
|---|---|---|
| **BM25 / Keyword** | Exact matches for brand names, SKUs, model numbers | Fails for synonyms, paraphrases |
| **Two-Tower Embedding** | Handles semantic similarity ("comfortable office shoes" matches formal footwear) | Can miss exact model numbers; requires GPU infrastructure |

**Hybrid Search:** Use both in parallel, then fuse scores (e.g., Reciprocal Rank Fusion or a learned combiner).

#### Two-Tower Model
- Train a **Query Tower** and an **Item Tower** separately.
- Both map their input to a shared embedding space.
- At inference, item embeddings are pre-computed and indexed in a vector DB (FAISS, Pinecone).
- Query embedding is computed in real-time and used for ANN search.

### 5. Learning to Rank (LTR)

Given ~1,000 candidates, a pointwise/pairwise/listwise ranking model scores them:

- **Pointwise:** Predict relevance score for each item independently (standard regression/classification). Simple but ignores inter-item relationships.
- **Pairwise (RankNet, LambdaRank):** Learn that item A should rank above item B. Better captures relative ordering.
- **Listwise (LambdaMART):** Directly optimizes a list-level metric like NDCG. Best offline performance.

**Key Features:**
- Query-item features: BM25 score, embedding similarity score, text overlap
- Item features: historical CTR, conversion rate, average rating, inventory level
- User-query features: user's purchase history for similar items, user's location vs. seller location
- Context: device type, time of day

### 6. Evaluation: The "Window Shopping" Problem

**Online metrics:**
- **Click-Through Rate (CTR):** User clicked on a result. Necessary but not sufficient.
- **Conversion Rate:** User *purchased* the item. Best signal but noisy (many clicks don't convert).
- **Revenue per Search:** Business-level metric.

**If the user doesn't buy anything:**
- It's not necessarily a failure. Measure **dwell time** on the product page — long dwell = interested.
- Use **add-to-cart** or **wishlist** as intermediate positive signals.
- Negative signals: **pogo-sticking** (click → immediate back → click next result) = poor result quality.

**Offline metrics:**
- **NDCG (Normalized Discounted Cumulative Gain):** Standard for ranked list quality, using human-labeled relevance judgments.
- **MRR (Mean Reciprocal Rank):** How high up is the first relevant result?

---

## Interviewer's Scoring Rubric

| Category | Weak (No Hire) | Strong (Hire) | Very Strong (Senior/Lead) |
|---|---|---|---|
| **Retrieval** | Uses only keyword search. | Mentions semantic search + Two-Tower models. | Discusses hybrid retrieval with learned fusion and ANN indexing strategies. |
| **Ranking** | Single ML model, vague features. | Describes LTR and relevant feature groups. | Discusses pairwise/listwise objectives and the trade-off between NDCG vs. revenue. |
| **Metrics** | CTR only. | Discusses conversion and negative signals. | Proposes a holistic success framework distinguishing browsing vs. buying intent. |
| **Freshness / Bias** | Ignores position bias. | Mentions position bias in training data. | Proposes Inverse Propensity Weighting (IPW) to debias click labels from position. |

---

## How to "Fail" a Candidate (Red Flags)

- **Ignoring Position Bias:** Training data is collected from the existing ranker. Items shown at position 1 get clicked more regardless of quality. Not correcting for this leads to a feedback loop (rich-get-richer). A strong candidate mentions debiasing.
- **No query understanding step:** Treating the raw query string as sacred misses basic improvements like spell correction.
- **Confusing retrieval with ranking:** These stages have fundamentally different latency and precision requirements and should not be collapsed into one.
