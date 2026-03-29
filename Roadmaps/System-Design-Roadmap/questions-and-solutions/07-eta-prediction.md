# Deep Dive: Estimated Time of Arrival (ETA) Prediction (Uber / DoorDash)

*Deep dive into Question #7. A strong candidate integrates real-time signals, models the road network as a graph, and explains how the system self-corrects during live incidents.*

---

## The Standard "Very Strong" Answer

### 1. Problem Clarification

Before designing, ask:
- **Use case:** Ride ETA (Uber), Delivery ETA (DoorDash), or Navigation (Google Maps)? Each has different time horizons and precision requirements.
- **Latency:** ETA at booking time must be < 200ms. In-trip updates can be slightly slower.
- **Success Metric:** Mean Absolute Error (MAE) in minutes? P90 error (tail latency of predictions)?

### 2. Architecture Overview

ETA prediction has two distinct components:

```
[Routing Component]
  Map Graph → Shortest Path (Dijkstra / A*)
  → Segment-level travel time estimates
  → Sum of segment times = Route Duration Estimate

[ML Adjustment Component]
  Route Duration Estimate + Real-time signals + Historical patterns
  → Final ETA prediction
```

### 3. The Road Network as a Graph

The map is modeled as a weighted directed graph:
- **Nodes:** Intersections
- **Edges:** Road segments with:
  - Base travel time (speed limit × distance)
  - ML-predicted travel time based on current conditions

**Segment-level travel time prediction:**
- For each road segment, predict expected traversal time given current conditions.
- Features: time of day, day of week, weather, historical speed at this segment, probe vehicle speeds (GPS pings from active drivers).
- Model: Gradient Boosted Trees or a Graph Neural Network (GNN) for spatial correlation between adjacent segments.

### 4. Feature Engineering

**Static features (pre-computed):**
- Road type (highway, residential, one-way)
- Number of lanes, presence of traffic lights
- Historical speed by hour/day-of-week (7 × 24 = 168 time bins)

**Real-time features:**
- Current probe vehicle speeds on each segment (GPS pings from active drivers)
- Incidents reported (accidents, road closures) from reporting services
- Weather API: rain/snow increases travel time significantly

**Trip-level features:**
- Time of pickup request
- Origin and destination neighborhoods
- Estimated number of turns/stops

### 5. Self-Correction During Live Incidents

This is the key "pro" follow-up. How does the system adapt to a sudden protest or accident?

**1. Probe vehicle feedback loop:** Active drivers on the road are GPS-tracked. If all drivers on a segment suddenly slow to 5mph (from 35mph), the system detects this in near-real-time (30–60 second lag as new GPS pings arrive).

**2. Exponential Moving Average update:** Segment travel time estimates are updated with a fast decay:
```
estimate_t = α × observed_t + (1 - α) × estimate_{t-1}
```
A small α means slower adaptation; a large α means rapid response to new observations.

**3. Re-routing trigger:** If a driver's current route develops a severe delay (detected via probe data), the system can push a re-route suggestion mid-trip.

**4. Incident reporting integration:** If a driver reports an accident via the app, that segment's travel time is immediately penalized in the graph weight.

### 6. Post-Trip Learning

Each completed trip provides ground-truth ETA accuracy data:
- Actual trip duration vs. predicted duration at booking time.
- This feedback is used for model retraining on a continuous basis.
- Segment-level accuracy can be tracked: "We consistently underestimate travel time on Segment X during Friday evening rush" → targeted feature improvement.

### 7. Evaluation

| Metric | Description |
|---|---|
| **MAE (Mean Absolute Error)** | Average absolute error in minutes. Primary metric. |
| **MAPE (Mean Absolute Percentage Error)** | Relative error — important for long trips where absolute error scales. |
| **P90 / P95 error** | Tail error — DoorDash users don't care about average ETA, they care about whether the driver *ever* arrives extremely late. |
| **Pre- vs. Post-trip MAE** | Separate metrics for ETA at booking time vs. real-time ETA during the trip. |

---

## Interviewer's Scoring Rubric

| Category | Weak (No Hire) | Strong (Hire) | Very Strong (Senior/Lead) |
|---|---|---|---|
| **Modeling** | "Train a model on past trips." | Describes segment-level features and graph routing. | Discusses GNNs for spatial correlation, EMA for real-time updates, and separate models for short vs. long time horizons. |
| **Real-time signals** | Uses only historical data. | Mentions probe vehicle GPS. | Details the feedback loop from live driver GPS pings to segment travel time updates with lag analysis. |
| **Self-correction** | Does not address. | Mentions re-routing if traffic changes. | Explains the EMA update mechanism, incident integration, and re-routing threshold trigger. |
| **Evaluation** | Uses accuracy or RMSE. | Uses MAE at booking time. | Proposes P90 error and separates booking-time ETA accuracy from in-trip ETA accuracy. |

---

## How to "Fail" a Candidate (Red Flags)

- **"Just train on historical routes":** A model without real-time signals is blind to current conditions. Historical data alone cannot handle a road closure that happened 10 minutes ago.
- **No graph modeling:** Treating origin-destination as a straight-line distance ignores the road network entirely.
- **Single-point prediction:** ETAs should ideally come with confidence intervals. "30–45 minutes" is more honest and useful than a false-precision "37 minutes."
