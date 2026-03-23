# Deep Dive: Model Monitoring & Observability

*Deep dive into Question #10. A strong candidate designs a comprehensive monitoring system that detects data drift, training-serving skew, and model performance degradation — before users or the business notice.*

---

## The Standard "Very Strong" Answer

### 1. Problem Clarification

Before designing, ask:
- **Model type:** Real-time inference (latency-sensitive) or batch prediction?
- **Label availability:** Do we get ground-truth labels quickly (e.g., click/no-click in seconds) or slowly (e.g., churn in 30 days)?
- **Business criticality:** What is the cost of a degraded model going undetected for 1 hour? 1 day?
- **Scale:** How many predictions per second? How many models to monitor?

### 2. The Core Problem: Silent Failures

Unlike a server crash (which triggers an alert immediately), ML model degradation is **silent**. The model keeps returning predictions — they're just wrong. The business metric suffers, but the cause is invisible without proper monitoring.

Three root causes:
1. **Data Drift:** Input feature distributions change over time (the world changes).
2. **Concept Drift:** The relationship between features and labels changes (the task changes).
3. **Training-Serving Skew:** Features at training time differ from features at serving time (an engineering bug).

### 3. Monitoring Architecture

```
[Live Traffic]
  Model receives requests → returns predictions
         |
  [Logging Layer]
    ├─ Log: input features, prediction, timestamp, model version, latency
    └─ Log: labels (when available) — may be delayed
         |
  [Monitoring Pipeline]
    ├─ Feature Distribution Monitor
    ├─ Prediction Distribution Monitor
    ├─ Model Performance Monitor (when labels available)
    └─ Infrastructure Monitor (latency, error rate, memory)
         |
  [Alerting System]
    └─ PagerDuty / Slack alerts → On-call engineer
```

### 4. What to Monitor

#### A. Feature Distributions (Data Drift Detection)

For each input feature, compare the current distribution against a reference distribution (training data or a recent healthy baseline).

**Statistical tests:**
| Test | Use case |
|---|---|
| **KL Divergence / Jensen-Shannon Divergence** | Compare probability distributions of continuous features |
| **Population Stability Index (PSI)** | Industry standard for detecting feature drift; common in finance |
| **Chi-squared test** | Categorical features |
| **Two-sample Kolmogorov–Smirnov test** | Non-parametric test for continuous features |

Alert when PSI > 0.2 (moderate drift) or PSI > 0.25 (significant drift).

#### B. Prediction Distribution

Monitor the distribution of model output scores even if you don't have labels yet:
- **Score distribution shift:** If the average predicted CTR suddenly drops from 2% to 0.5%, something changed upstream.
- **Prediction rate anomaly:** If the fraction of high-confidence predictions changes dramatically.
- **Null/error rates:** Spikes in NaN predictions or fallback values.

This is your **leading indicator** — it fires before business metrics degrade.

#### C. Model Performance (Lagging Indicator)

When labels are available:
- Track AUC, F1, Precision, Recall, MAE, etc. over time in rolling windows (hourly, daily).
- Set alert thresholds relative to a baseline: "Alert if AUC drops more than 5% relative to last week."

#### D. Infrastructure Metrics

- **Latency:** P50, P95, P99 inference latency. A sudden latency spike may indicate a bad model deployment or resource contention.
- **Error rate:** % of requests returning errors or timeouts.
- **Throughput:** RPS (requests per second) anomalies can indicate upstream issues.

### 5. Training-Serving Skew

The "pro" follow-up: *"Model AUC was 0.9 offline but revenue is dropping. Where do you look first?"*

**Answer: Training-Serving Skew.** This is the most common cause of an "offline success, online failure" gap.

**Common causes:**
- A feature is computed differently at training time vs. inference time (e.g., training uses a 30-day window, serving uses a 7-day window due to a bug).
- A feature is logged at training time from a ground-truth source but served from a faster but noisier source.
- Training data was collected under a different traffic distribution than current production traffic.

**Detection:**
- **Shadow mode logging:** Log training-time feature values alongside serving-time feature values for the same entities. Compare them directly.
- **Feature consistency checks:** Assert that features computed at training time and serving time are statistically identical for the same input.

### 6. Alerting and Response

| Severity | Trigger | Response |
|---|---|---|
| **P1 (Critical)** | Model error rate > 10%, prediction distribution completely collapsed | Page on-call immediately, consider rollback |
| **P2 (High)** | Model performance drops > 10% from baseline | Alert team, investigate within hours |
| **P3 (Medium)** | Significant feature drift detected (PSI > 0.25) | Alert within 24 hours, schedule investigation |
| **P4 (Low)** | Moderate drift or gradual performance decay | Weekly summary report, schedule retraining |

**Rollback plan:** Every model deployment should tag the current version and have a one-click rollback to the previous version.

### 7. Retraining Strategy

Monitoring should feed into a retraining pipeline:
- **Scheduled retraining:** Retrain on a fixed cadence (daily/weekly) using fresh data. Simple but may lag behind rapid drift.
- **Triggered retraining:** Automatically trigger a retraining job when drift is detected above a threshold.
- **Continuous training:** Stream new labeled data into the model in near-real-time (complex, suitable for very dynamic environments like ad CTR).

---

## Interviewer's Scoring Rubric

| Category | Weak (No Hire) | Strong (Hire) | Very Strong (Senior/Lead) |
|---|---|---|---|
| **Drift Detection** | "Watch the accuracy metric." | Monitors feature distributions and uses PSI/KS tests. | Distinguishes data drift vs. concept drift, uses leading indicators (prediction distribution) before labels arrive. |
| **Training-Serving Skew** | Does not mention. | Knows it exists and is the common failure mode. | Proposes shadow mode logging and feature consistency assertions as a systematic detection approach. |
| **Alerting** | Single threshold on one metric. | Multi-metric alerts with severity levels. | Designs a tiered alert system with rollback SLAs and runbooks for each alert type. |
| **Retraining** | "Retrain when it breaks." | Proposes scheduled retraining. | Proposes triggered retraining with a champion/challenger evaluation framework before promoting the new model. |

---

## How to "Fail" a Candidate (Red Flags)

- **Monitoring only business KPIs:** Revenue dropping is a lagging indicator — by the time it's visible, the model has been degraded for hours or days. Strong candidates monitor ML metrics directly.
- **No discussion of leading vs. lagging indicators:** Prediction distribution shifts are detectable before labels arrive. A candidate who waits for labels before alarming has a blind spot.
- **No rollback plan:** Every production model deployment without a rollback mechanism is a single point of failure.
