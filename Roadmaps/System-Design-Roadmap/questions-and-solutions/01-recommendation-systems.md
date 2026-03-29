# Deep Dive: Designing a Recommendation System (Instagram Reels / TikTok Feed)

*Deep dive into Question #1. A strong candidate walks through a multi-stage pipeline and demonstrates awareness of the explore-exploit trade-off.*

---

## The Standard "Very Strong" Answer

### 1. Problem Clarification

Before designing, ask:
- **Scale:** How many users (DAU)? How many items in the candidate pool?
- **Latency:** What is the SLA? (e.g., < 200ms to return a feed)
- **Success Metric:** Engagement rate? Watch time? Long-term retention? (These can conflict!)

### 2. Architecture: The Multi-Stage Pipeline

A production recommendation system is never a single model. It is a funnel:

```
[All Items: ~10M]
       |
  [Candidate Generation / Retrieval]  ~1,000 candidates  (fast, recall-focused)
       |
  [Pre-ranking / Filtering]           ~200 candidates    (cheap model, filter junk)
       |
  [Ranking]                           ~50 candidates     (heavy model, precision-focused)
       |
  [Re-ranking / Business Logic]       ~20 items          (diversity, ads, freshness rules)
       |
  [User's Feed]
```

#### Stage 1: Candidate Generation (Retrieval)
- **Goal:** Recall. Find the ~1,000 most plausible candidates from millions of items quickly.
- **Methods:**
  - **Collaborative Filtering:** Users with similar watch history tend to like similar content. Use matrix factorization (ALS) or a Two-Tower neural network to produce user/item embeddings. Retrieve Top-K via Approximate Nearest Neighbor (ANN) search (e.g., FAISS, ScaNN).
  - **Content-Based Filtering:** Recommend items similar to what the user recently watched (based on item embeddings from video/audio features).
  - **Multiple Retrieval Sources:** Use several retrieval strategies in parallel and merge the candidates. This improves recall.

#### Stage 2: Ranking
- **Goal:** Precision. Given ~200 candidates, score each one carefully.
- **Model:** A deep neural network (e.g., a Wide & Deep model or DIN — Deep Interest Network) that takes rich features as input.
- **Key Features:**
  - **User features:** age, location, historical engagement patterns
  - **Item features:** category, length, creator ID, historical CTR
  - **Context features:** time of day, device type, network speed
  - **Cross features:** user-item interaction features (has the user liked content from this creator before?)
- **Training Objective:** Predict multiple labels simultaneously (Multi-Task Learning): P(like), P(share), P(complete watch), P(not-interested). Use a weighted sum as the final score.

#### Stage 3: Re-ranking & Business Logic
- **Diversity:** Avoid showing 10 videos from the same creator in a row. Use Maximal Marginal Relevance (MMR) or slot-based diversity rules.
- **Freshness Boost:** Apply a small score boost to newer content.
- **Safety & Policy:** Filter out content that violates community guidelines or that the user has already seen.

### 3. Explore vs. Exploit Trade-off

This is the critical "pro" question. The model, if left unchecked, will always show you what you already like → **Filter Bubble**.

- **Thompson Sampling / UCB (Upper Confidence Bound):** Treat recommendations as a multi-armed bandit. Items with uncertainty in their predicted reward get a bonus score, encouraging exploration.
- **Dedicated Exploration Slots:** Reserve 10-15% of feed positions for "exploration" candidates — items outside the user's normal consumption pattern but with strong global popularity.
- **Serendipity Score:** Add a diversity term to the ranking objective that penalizes the model for showing items too similar to the user's recent history.

### 4. Training & Data

- **Labels:** Implicit feedback (click, watch%, share) rather than explicit ratings. Watch >80% = strong positive signal.
- **Data Leakage Risk:** Feature values must be computed from data available *before* the impression, not after.
- **Training Frequency:** Ranking model may be retrained daily or weekly; retrieval models less often.

---

## Interviewer's Scoring Rubric

| Category | Weak (No Hire) | Strong (Hire) | Very Strong (Senior/Lead) |
|---|---|---|---|
| **Architecture** | Proposes a single model over all items. | Describes a two-stage retrieval + ranking pipeline. | Describes a full funnel with pre-ranking and re-ranking stages. |
| **Explore/Exploit** | Says "just show popular content." | Mentions bandit algorithms or exploration slots. | Discusses long-term retention vs. short-term engagement trade-offs and proposes a metric to measure filter bubble effects. |
| **Metrics** | Uses only accuracy. | Proposes engagement rate or watch time. | Proposes multi-objective metrics and discusses how optimizing CTR alone can lead to clickbait. |
| **Cold Start** | Ignores new users/items. | Suggests popularity-based fallback for new items. | Proposes content-based bootstrapping for new items and onboarding questionnaire for new users. |

---

## How to "Fail" a Candidate (Red Flags)

- **Single-stage design:** Proposing to run a complex neural network over all 10M items for every request is a latency non-starter.
- **No cold-start plan:** A new item has zero interaction data. The candidate should address how it gets into the recommendation pool at all.
- **Optimizing only CTR:** Click-Through Rate is easy to game (clickbait thumbnails). A strong candidate discusses watch time, shares, or long-term retention as more meaningful signals.
