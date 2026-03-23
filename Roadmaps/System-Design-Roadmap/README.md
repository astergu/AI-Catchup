# ML System Design Interview Guide


## Top 10 ML System Design Interview Questions

*From the perspective of a technical interviewer — mastering the terminology and "Interviewer Mindset" is key.*

---

### 1. [Recommendation Systems (e.g., Instagram Reels, TikTok Feed)](./questions-and-solutions/01-recommendation-systems.md)

**The Task:** Design a system to recommend personalized video content to millions of users.

**What to Look For:** Can the candidate describe a multi-stage pipeline?

**Candidate Signal:** Mentions "Candidate Generation/Retrieval" (fast, simple) followed by "Ranking" (heavy, precise).

**"Pro" Follow-up:** "How do you handle the Explore vs. Exploit trade-off?" (Do we show what they like, or something new to keep them from getting bored?)

---

### 2. [Search Ranking (e.g., Amazon Product Search, Google Maps)](./questions-and-solutions/02-search-ranking.md)

**The Task:** Rank search results based on user queries and historical behavior.

**What to Look For:** Understanding of Semantic Search vs. Keyword Matching.

**Candidate Signal:** Discusses Learning to Rank (LTR) or Two-Tower Embedding models.

**"Pro" Follow-up:** "How do you measure success if the user doesn't buy anything? Is it a failure or just a 'window shopping' session?"

---

### 3. [Ads Click-Through Rate (CTR) Prediction](./questions-and-solutions/03-ads-ctr-prediction.md)

**The Task:** Predict the probability that a user will click on a specific ad.

**What to Look For:** Knowledge of Data Sparsity and Calibration.

**Candidate Signal:** Mentions Log-loss as an evaluation metric and handles categorical features (like UserID) using embeddings.

**"Pro" Follow-up:** "Why is Calibration important here? If the model says 10% but the real world is 1%, how does that hurt the bidding system?"

---

### 4. [Fraud & Anomaly Detection (e.g., Credit Card Fraud, Bot Detection)](./questions-and-solutions/04-fraud-anomaly-detection.md)

**The Task:** Identify fraudulent transactions in real-time.

**What to Look For:** Handling Extreme Class Imbalance.

**Candidate Signal:** Suggests SMOTE, down-sampling, or specific loss functions (Focal Loss) instead of just "Accuracy."

**"Pro" Follow-up:** "What is the business cost of a False Positive (blocking a legitimate customer) versus a False Negative (allowing a thief)?"

---

### 5. [Content Moderation (e.g., Detecting Hate Speech or NSFW Images)](./questions-and-solutions/05-content-moderation.md)

**The Task:** Automatically flag or remove violating content on a social platform.

**What to Look For:** Human-in-the-Loop design.

**Candidate Signal:** Suggests a model that flags "unsure" cases for human review.

**"Pro" Follow-up:** "How do you handle Concept Drift? (e.g., new slang or memes that become offensive overnight)."

---

### 6. [LLM-Powered Systems (RAG Architecture)](./questions-and-solutions/06-rag-llm-powered-systems.md)

**The Task:** Build a Q&A bot for a company's internal documentation using a Large Language Model.

**What to Look For:** Understanding the RAG (Retrieval-Augmented Generation) stack.

**Candidate Signal:** Discusses Vector Databases (Pinecone/Milvus), Chunking strategies, and "Hallucination" mitigation.

**"Pro" Follow-up:** "How do you evaluate the 'Truthfulness' of the generated answer automatically?"

---

### 7. [Estimated Time of Arrival (ETA) (e.g., Uber, DoorDash)](./questions-and-solutions/07-eta-prediction.md)

**The Task:** Predict how long a delivery or ride will take.

**What to Look For:** Real-time Data Integration.

**Candidate Signal:** Mentions using graph-based features (road segments) and real-time signals (traffic/weather).

**"Pro" Follow-up:** "How does the system self-correct if a driver gets stuck in a sudden protest or accident?"

---

### 8. [Visual Search (e.g., Pinterest "Lens", Google Photos Search)](./questions-and-solutions/08-visual-search.md)

**The Task:** Find similar images based on a user-uploaded photo.

**What to Look For:** Embedding Retrieval at Scale.

**Candidate Signal:** Mentions Approximate Nearest Neighbor (ANN) search and Feature Extraction via CNNs or ViT.

**"Pro" Follow-up:** "How do you ensure the system is fast enough to run on a mobile device's low-bandwidth connection?"

---

### 9. [Dynamic Pricing (e.g., Airbnb Smart Pricing)](./questions-and-solutions/09-dynamic-pricing.md)

**The Task:** Adjust prices based on supply, demand, and seasonality.

**What to Look For:** Time-Series Analysis and Elasticity.

**Candidate Signal:** Uses Reinforcement Learning or Regression with seasonality features.

**"Pro" Follow-up:** "If your model suggests a price 5x higher than usual, how do you prevent brand damage or 'price gouging' accusations?"

---

### 10. [Model Monitoring & Observability](./questions-and-solutions/10-model-monitoring-observability.md)

**The Task:** Design a system to ensure an ML model stays healthy after deployment.

**What to Look For:** Detection of Data Drift.

**Candidate Signal:** Proposes monitoring Feature Distributions and Prediction Distributions.

**"Pro" Follow-up:** "The model's offline AUC was 0.9, but online business revenue is dropping. Where do you look first?" (Training-Serving Skew).

---

## Interviewer's Evaluation Rubric

When listening to a candidate, use this 4-point scale for each section:

| Dimension | What to Check |
|---|---|
| **Clarification** | Did they ask about Scale (QPS), Latency (ms), and Success Metrics (Precision vs. Recall)? |
| **Data** | Did they mention Data Leakage? (e.g., using "future" information to predict the "past") |
| **Architecture** | Did they start with a Simple Baseline before jumping into complex Deep Learning? |
| **Operationalization** | Did they discuss A/B Testing and Rollback plans? |

---

## Resources

- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)

---
