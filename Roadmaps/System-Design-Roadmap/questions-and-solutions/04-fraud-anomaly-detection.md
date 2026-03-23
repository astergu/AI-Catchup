# Deep Dive: Fraud & Anomaly Detection (Credit Card Fraud / Bot Detection)

*Deep dive into Question #4. A strong candidate handles extreme class imbalance and reasons carefully about the asymmetric cost of False Positives vs. False Negatives.*

---

## The Standard "Very Strong" Answer

### 1. Problem Clarification

Before designing, ask:
- **Scale:** How many transactions per second? (Visa processes ~24,000 TPS)
- **Latency:** Must the decision be made in real-time (< 100ms before authorizing the charge)?
- **Objective:** Minimize fraud loss? Minimize false positives (customer friction)? Both?

### 2. The Core Challenge: Extreme Class Imbalance

In credit card fraud, fraudulent transactions are typically 0.1%–1% of all transactions. Standard training on this dataset will produce a model that predicts "not fraud" for everything and achieves 99%+ accuracy — completely useless.

**Techniques to handle imbalance:**

| Technique | How | When to Use |
|---|---|---|
| **Down-sampling** | Randomly remove majority-class (non-fraud) samples during training | Fast, simple, works well for tree-based models |
| **SMOTE** | Synthetically oversample minority-class (fraud) by interpolating in feature space | When fraud examples are too few to learn from |
| **Class weights** | Penalize misclassifying fraud more heavily in the loss function | Clean, no data duplication; works well with neural nets |
| **Focal Loss** | A modified cross-entropy that down-weights well-classified easy examples | Best for very extreme imbalance, originally from object detection |

### 3. Feature Engineering

Fraud patterns are often temporal and behavioral:

- **Transaction features:** Amount, merchant category, location, time of day
- **Velocity features:** Number of transactions in last 1hr / 24hr / 7days (sudden spike = red flag)
- **Statistical aggregates:** Average transaction amount for this user, deviation from their norm (z-score)
- **Graph features:** Is this merchant or card connected to known fraud nodes? (Graph Neural Networks)
- **Device/Behavioral features (for bot detection):** Mouse movement patterns, typing speed, browser fingerprint, JavaScript execution patterns

### 4. Model Architecture

**Layer 1 — Real-Time Rule Engine (< 5ms):**
- Hard rules: Amount > $10,000 AND new country → block immediately.
- Fast, interpretable, no ML needed.

**Layer 2 — ML Model (< 100ms):**
- **Gradient Boosted Trees (XGBoost/LightGBM):** Industry standard for tabular fraud data. Handles missing values, works well with engineered features, fast inference.
- **Neural Network with LSTM/GRU:** For sequential transaction patterns — can model the temporal context of a session.
- Output: P(fraud) score between 0 and 1.

**Layer 3 — Human Review Queue:**
- Transactions with P(fraud) in a middle "uncertain" zone (e.g., 0.3–0.7) are flagged for human review rather than auto-blocked.

### 5. Decision Threshold & Business Cost

The model outputs a probability. You need a threshold to make a binary decision. This threshold is a **business decision**, not a model decision.

**False Positive (FP):** Block a legitimate transaction.
- Cost: Customer frustration, potential customer churn, lost sale revenue, support call cost.

**False Negative (FN):** Allow a fraudulent transaction.
- Cost: Direct fraud loss (charged back to the bank), potential regulatory fines.

A bank might set the threshold to optimize:
```
Business Loss = FP_count × Cost_FP + FN_count × Cost_FN
```
The right threshold is where this combined cost is minimized. This is why **Precision-Recall curves** (and AUCPR) are far more useful than ROC curves in highly imbalanced settings.

### 6. Online Learning & Concept Drift

Fraudsters adapt. A model trained on last year's fraud patterns will degrade quickly.
- **Frequent retraining:** Retrain the model weekly or daily as new labeled fraud examples arrive.
- **Online learning:** Some systems update model weights incrementally with each new confirmed fraud case.
- **Champion/Challenger A/B testing:** Deploy a new model version alongside the old one on a small traffic slice before full rollout.

---

## Interviewer's Scoring Rubric

| Category | Weak (No Hire) | Strong (Hire) | Very Strong (Senior/Lead) |
|---|---|---|---|
| **Imbalance** | Uses accuracy as metric, ignores imbalance. | Mentions SMOTE or class weights. | Discusses Focal Loss, the trade-off between resampling strategies, and uses AUCPR as the primary offline metric. |
| **Features** | Only uses transaction amount. | Mentions velocity and behavioral features. | Proposes graph-based features and temporal sequence modeling. |
| **Business Cost** | Treats FP and FN as equal errors. | Acknowledges FP and FN have different costs. | Proposes a cost-sensitive threshold selection framework using actual business cost estimates. |
| **Real-Time** | No mention of latency. | Notes that inference must be fast. | Designs a layered system (rules → ML → human review) with explicit latency budgets for each layer. |

---

## How to "Fail" a Candidate (Red Flags)

- **"I'll use accuracy":** Classic imbalance trap. This is an immediate red flag.
- **No real-time consideration:** A fraud model that takes 5 seconds to return a score is useless for authorizing a payment.
- **Static model:** Not discussing model updates or concept drift. Fraudsters iterate continuously.
