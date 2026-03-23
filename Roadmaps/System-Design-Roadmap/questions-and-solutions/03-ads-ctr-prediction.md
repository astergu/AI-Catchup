# Deep Dive: Ads Click-Through Rate (CTR) Prediction

*Deep dive into Question #3. A strong candidate understands data sparsity, feature engineering for sparse categorical data, and why calibration is critical for the bidding system.*

---

## The Standard "Very Strong" Answer

### 1. Problem Clarification

Before designing, ask:
- **Scale:** How many ad impressions per day? (Facebook/Google scale: trillions)
- **Latency:** Must respond in < 10ms within the ad auction.
- **Success Metric:** Log-loss (calibration quality) and AUC (ranking quality).

### 2. The Core Challenge: Sparsity

Ads CTR prediction involves:
- Millions of unique users
- Millions of unique ads/advertisers
- Billions of (user, ad) pairs — but most pairs have *never been seen before*

This is the **data sparsity problem**. A model can't learn a direct parameter for every user-ad pair.

### 3. Feature Engineering

**Key Feature Categories:**

| Feature | Examples | Encoding |
|---|---|---|
| **User features** | UserID, age, gender, interests | Embedding lookup |
| **Ad features** | AdID, AdvertiserID, category, creative type | Embedding lookup |
| **Context features** | Hour of day, device, placement position | One-hot or numeric |
| **Cross features** | User category × Ad category | Feature crossing or learned interactions |
| **Historical stats** | Historical CTR of ad, CTR of user, CTR of ad-in-placement | Numeric |

**Embeddings for Sparse IDs:** UserID, AdID — instead of one-hot encoding (which would be millions of dimensions), learn a low-dimensional dense embedding (e.g., 32-128 dimensions). This generalizes to unseen (user, ad) pairs via shared semantic structure.

### 4. Model Architecture

**Classic Baseline:** Logistic Regression with feature crosses (Google's approach, highly efficient)

**Modern Deep Approach:** Wide & Deep / DeepFM / DCN (Deep & Cross Network)
- **Wide part:** Linear model on raw and crossed features — good at memorization
- **Deep part:** DNN on learned embeddings — good at generalization
- **Interaction modeling:** Explicit feature interaction layers (FM layers, CrossNet) capture combinatorial feature effects without hand-crafting every cross

**Training:**
- Label: 1 if clicked, 0 if not
- Loss: **Binary Cross-Entropy (Log-loss)**
- Optimizer: Adam or Adagrad (sparse-friendly)
- Sampling: Negative down-sampling — clicks are rare (~1% CTR), so the dataset is heavily imbalanced. Down-sample non-click impressions and *correct the output score accordingly*.

### 5. Calibration — Why It Matters for Bidding

**Calibration** means: if the model predicts CTR = 10%, then ~10% of those predictions should actually result in clicks.

**Why it matters in ad auctions:**
- The ad auction uses `Expected Revenue = Bid × P(Click)`.
- If the model is miscalibrated (predicts 10% but true rate is 1%), the auction *overvalues* the ad, leading to the wrong winner being selected and suboptimal revenue for the platform.
- **Platt Scaling** or **Isotonic Regression** are post-processing calibration techniques applied after training.

**Negative Down-Sampling Correction:** If you down-sample 99% of negatives for training efficiency, you must correct the output probabilities at inference time:
```
p_corrected = p_model / (p_model + (1 - p_model) / q)
```
where `q` is the down-sampling rate.

### 6. Evaluation

| Metric | What it measures |
|---|---|
| **Log-loss** | Calibration quality — is the probability estimate accurate? |
| **AUC-ROC** | Ranking quality — can the model separate clicks from non-clicks? |
| **Normalized Cross Entropy (NE)** | Log-loss normalized by the entropy of the background CTR — accounts for different base rates across datasets |

**Online Metrics:** Revenue per mille (RPM), overall ad revenue, auction win rates.

---

## Interviewer's Scoring Rubric

| Category | Weak (No Hire) | Strong (Hire) | Very Strong (Senior/Lead) |
|---|---|---|---|
| **Feature Engineering** | Mentions basic features, uses one-hot for user IDs. | Uses embeddings for sparse IDs. | Discusses embedding dimensionality trade-offs and pre-training embeddings from co-occurrence data. |
| **Model** | Logistic Regression or basic DNN. | Wide & Deep or FM family. | Discusses DCN/DeepFM and the motivation for explicit interaction modeling. |
| **Calibration** | Not mentioned. | Knows calibration is important for bidding. | Explains down-sampling correction formula and post-hoc calibration methods. |
| **Training at Scale** | No mention of scale. | Mentions distributed training. | Discusses parameter servers, async SGD, and the challenge of training with billions of examples. |

---

## How to "Fail" a Candidate (Red Flags)

- **Using accuracy as the metric:** With 1% CTR, a model that predicts 0 for everything achieves 99% accuracy. This is useless. Always use log-loss or AUC.
- **Ignoring calibration:** Ranking ads (AUC) is not the same as calibrating bid prices. A miscalibrated model breaks the auction mechanism.
- **One-hot encoding user IDs:** This is computationally infeasible at scale (millions of dimensions) and doesn't generalize.
