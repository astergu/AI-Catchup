# Ad Prediction Models

Implementation of popular CTR/CVR prediction models for computational advertising.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test all models on the same dataset
python test_models.py
```

This will train and compare all implemented models on synthetic CTR data and save results to `results.csv`.

## Implemented Models

| Model | File | Task | Status |
| --- | --- | --- | --- |
| Wide & Deep (WDL) | [wdl.py](wdl.py) | CTR | ✅ Complete |
| ESMM | [esmm.py](esmm.py) | CTR + CVR | ✅ Complete |
| DCN / DCN-v2 | [dcn.py](dcn.py) | CTR | ✅ Complete |
| DIN | [din.py](din.py) | CTR | ✅ Complete |
| MMoE | [mmoe.py](mmoe.py) | CTR + CVR | ✅ Complete |
| PLE | [ple.py](ple.py) | CTR + CVR | ✅ Complete |
| DIEN | [dien.py](dien.py) | CTR | ✅ Complete |
| DeepFM | [deepfm.py](deepfm.py) | CTR | ✅ Complete |
| AutoInt | [autoint.py](autoint.py) | CTR | ✅ Complete |

## Model Descriptions

| Model | Year | Key Idea | Architecture | Strength |
| --- | --- | --- | --- | --- |
| **Wide & Deep** | 2016 · Google | Joint training of a linear model (memorization) and a DNN (generalization) | Wide: `Linear(x)` · Deep: `DNN(x)` · Output: `σ(wide + deep)` | Simple, production-proven; the wide component excels at sparse categorical feature co-occurrences |
| **ESMM** | 2018 · Alibaba | Factorize CTCVR = CTR × CVR and train both on the entire impression space to eliminate sample selection bias in CVR modeling | Shared embedding → CTR Tower + CVR Tower · `p_ctcvr = p_ctr × p_cvr` · Loss: `BCE(y_ctr, p_ctr) + BCE(y_ctcvr, p_ctcvr)` | Solves sample selection bias and data sparsity; CVR is learned implicitly via CTCVR supervision—no direct CVR labels on clicked-only data needed |
| **DCN-v1** | 2017 · Google | Cross network with vector weights explicitly learns bounded-degree polynomial feature interactions | Cross layer: `x_{l+1} = x_0·(x_l·w_l) + b_l + x_l` (O(d) params) · Parallel DNN · Concat → output | Efficient; after L cross layers captures all feature interactions up to degree L+1 with no manual feature engineering |
| **DCN-v2** | 2021 · Google | Upgrades DCN-v1 cross layer from a vector to a full (or low-rank) matrix for richer interaction modeling | Cross layer: `x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l` (O(d²) or O(dr) with low-rank) · Parallel or stacked DNN | Significantly more expressive than v1 at the same depth; parallel structure outperforms stacked |
| **DIN** | 2018 · Alibaba | Attention-weighted pooling over a user's click history, conditioned on the target ad | Attention score: `a(h_i, e_target)` · Weighted sum of history embeddings → DNN | Captures variable-length user interest; attention focuses on behavior most relevant to the current candidate ad |
| **MMoE** | 2018 · Google | Multiple shared expert networks with task-specific gating networks; each task gets its own weighted combination of experts | K shared experts (DNNs) · Per-task softmax gate → weighted sum of experts → task tower | Explicitly models task relationships; gates learn how much each task should rely on each shared expert |
| **PLE** | 2020 · Tencent | Separates shared and task-specific experts to avoid negative transfer, then progressively extracts task representations | Task-specific experts + shared experts per layer · Gating: `gate_k = softmax(W_k · x)` · Output: task towers | Outperforms MMoE on conflicting tasks by giving each task its own experts that aren't forced to compromise |
| **DIEN** | 2019 · Alibaba | Models the temporal evolution of user interest with a two-stage RNN: GRU extracts interest states (with auxiliary next-click supervision), AUGRU evolves them conditioned on the target item | Stage 1: GRU + auxiliary loss · Stage 2: AUGRU (`u'_t = a_t ⊙ u_t`) · DNN | Captures interest drift over time; AUGRU update gate scales with target relevance, freezing evolution at irrelevant history steps |
| **DeepFM** | 2017 · Huawei | Replace Wide & Deep's wide component with a Factorization Machine and share the embedding table between FM and DNN, eliminating the need for hand-crafted cross features | 1st-order: `Σ wᵢxᵢ` · 2nd-order FM: `½(‖Σeᵢ‖²−Σ‖eᵢ‖²)` · Deep: `DNN(flatten(e))` · Output: `σ(1st + 2nd + DNN)` | No manual feature engineering; FM captures all pairwise interactions in O(n·k); shared embeddings reduce parameters vs Wide & Deep |
| **AutoInt** | 2019 · BIT | Apply multi-head self-attention to the stacked field embeddings so every field can attend to every other field, learning which pairs matter most in a sample-adaptive way | L interacting layers: `α = softmax(QKᵀ/√d_h)`, `ẽ = αV`, residual+ReLU · Flatten → Dense(1) · Optional parallel DNN (AutoInt+) | Interactions are explicit and interpretable; stacking L layers captures up to (L+1)-th order interactions without combinatorial enumeration |

## Benchmark Results

Results on synthetic CTR data (`test_models.py`). AUC is the primary metric.

| Model | AUC | Accuracy | Log Loss | Notes |
| --- | --- | --- | --- | --- |
| WDL (Adam) | 0.883 | 0.933 | 0.196 | Single optimizer, recommended default |
| WDL (FTRL+Adagrad) | 0.601 | 0.325 | 0.922 | Dual optimizers; underperforms on dense synthetic data, excels on truly sparse production features |
| DCN-v2 Parallel | **0.978** | **0.966** | **0.121** | Best single-task CTR model |
| DCN-v2 Stacked | 0.948 | 0.948 | 0.234 | DNN loses access to raw input |
| DCN-v1 Parallel | 0.975 | 0.964 | 0.121 | Near-identical to v2 |
| DIN | 0.888 | 0.918 | 0.297 | Sequence dataset (N=30k); behavior-conditioned attention |
| DIEN | **0.901** | **0.925** | **0.260** | Same dataset as DIN; GRU + AUGRU with auxiliary supervision |
| DeepFM | 0.881 | 0.918 | 0.321 | Categorical dataset (N=30k); FM + DNN with shared embeddings |
| AutoInt | **0.887** | **0.930** | **0.261** | Same dataset as DeepFM; 3 interacting layers, 2 heads, d_h=8 |

ESMM and MMoE are evaluated on a separate CTR+CVR funnel dataset (not directly comparable to single-task models):

| Model | CTR AUC | CTCVR AUC | Notes |
| --- | --- | --- | --- |
| ESMM | **0.849** | **0.870** | Hard probabilistic coupling `p_ctcvr = p_ctr × p_cvr` |
| MMoE | 0.794 | 0.815 | All tasks share all K experts |
| PLE | 0.803 | 0.816 | Task-specific + shared experts; 2 extraction layers |

## Usage Examples

### Wide & Deep

```python
from wdl import WDL
import numpy as np

X_wide = np.random.binomial(1, 0.3, (10000, 15))  # sparse binary features
X_deep = np.random.randn(10000, 25)                # dense embeddings / numericals
y = np.random.binomial(1, 0.08, 10000).astype(np.float32)

model = WDL(wide_input_dim=15, deep_input_dim=25, deep_hidden_units=[128, 64, 32])
model.fit(X_wide, X_deep, y, epochs=10, batch_size=64)
probs = model.predict(X_wide, X_deep)   # shape (N, 1)
```

### ESMM (CTR + CVR on full impression space)

```python
from esmm import ESMM
import numpy as np

X = np.random.randn(30000, 30).astype(np.float32)      # all impressions
y_ctr   = np.random.binomial(1, 0.15, 30000).astype(np.float32)  # clicked?
y_ctcvr = (y_ctr * np.random.binomial(1, 0.10, 30000)).astype(np.float32)  # clicked AND converted?

model = ESMM(input_dim=30, ctr_tower_units=[128, 64], cvr_tower_units=[128, 64])
model.fit(X, y_ctr, y_ctcvr, epochs=20, batch_size=256)

preds = model.predict(X)
p_ctr   = preds["p_ctr"]    # CTR probability
p_cvr   = preds["p_cvr"]    # CVR probability (no SSB — trained on all impressions)
p_ctcvr = preds["p_ctcvr"]  # CTCVR = p_ctr × p_cvr
```

### DCN / DCN-v2

```python
from dcn import DCN
import numpy as np

# DCN takes a single stacked feature vector
X = np.concatenate([X_wide, X_deep], axis=1)   # shape (N, 40)

# DCN-v2 parallel (recommended)
model = DCN(
    input_dim=40,
    num_cross_layers=3,          # polynomial degree up to 4
    deep_units=[128, 64, 32],
    version="v2",                # 'v1' for vector weights, 'v2' for matrix weights
    structure="parallel",        # 'parallel' (default) or 'stacked'
    low_rank=None,               # set to e.g. 10 to use W ≈ U Vᵀ in v2
)
model.fit(X, y, epochs=10, batch_size=64)
probs = model.predict(X)        # shape (N, 1)
```

### DIN (Deep Interest Network)

```python
from din import DIN
import numpy as np

N_ITEMS, MAX_SEQ = 200, 20
item_seq    = np.random.randint(0, N_ITEMS + 1, (10000, MAX_SEQ))  # 0 = padding
target_item = np.random.randint(1, N_ITEMS + 1, 10000)
y = np.random.binomial(1, 0.2, 10000).astype(np.float32)

model = DIN(
    n_items=N_ITEMS,
    max_seq_len=MAX_SEQ,
    embed_dim=16,
    attention_units=(64, 16),  # attention MLP hidden sizes
    dnn_units=[256, 128, 64],
    use_dice=True,             # Dice activation vs ReLU in attention MLP
)
model.fit(item_seq, target_item, y, epochs=10, batch_size=256)
probs = model.predict(item_seq, target_item)  # shape (N, 1)
```

### DIEN (Deep Interest Evolution Network)

```python
from dien import DIEN
import numpy as np

N_ITEMS, MAX_SEQ = 200, 20
item_seq    = np.random.randint(0, N_ITEMS + 1, (10000, MAX_SEQ))  # 0 = padding
target_item = np.random.randint(1, N_ITEMS + 1, 10000)
y = np.random.binomial(1, 0.2, 10000).astype(np.float32)

# Random negatives for auxiliary loss (items not in user's history)
neg_seq = np.random.randint(1, N_ITEMS + 1, (10000, MAX_SEQ))

model = DIEN(
    n_items=N_ITEMS,
    max_seq_len=MAX_SEQ,
    embed_dim=16,               # gru_units defaults to embed_dim
    attention_units=(64, 16),   # attention MLP in Stage 2
    dnn_units=[256, 128, 64],
    aux_loss_weight=0.5,        # weight on Stage-1 auxiliary loss
)
model.fit(item_seq, target_item, y, neg_seq=neg_seq, epochs=10, batch_size=256)
probs = model.predict(item_seq, target_item)  # neg_seq not needed at inference
```

### AutoInt

```python
from autoint import AutoInt
import numpy as np

N_USERS, N_ITEMS, N_CATS, N = 200, 200, 5, 30000

user_ids = np.random.randint(0, N_USERS, N).astype(np.int32)
item_ids = np.random.randint(0, N_ITEMS, N).astype(np.int32)
cat_ids  = np.random.randint(0, N_CATS,  N).astype(np.int32)
sparse_feats = [user_ids, item_ids, cat_ids]

dense_feats = np.random.randn(N, 5).astype(np.float32)
y = np.random.binomial(1, 0.1, N).astype(np.float32)

model = AutoInt(
    field_dims=[N_USERS, N_ITEMS, N_CATS],
    embed_dim=16,
    dense_feat_dim=5,
    num_heads=2,
    att_dim=8,         # per-head; total = 2×8 = 16 = embed_dim (no projection)
    num_layers=3,      # stacking L layers captures up to (L+1)-th order interactions
    use_residual=True,
    dnn_units=[],      # empty = pure AutoInt; non-empty = AutoInt+ (parallel DNN)
    dropout_rate=0.1,
)
model.fit(sparse_feats, y, dense_feats=dense_feats, epochs=20, batch_size=256)
probs = model.predict(sparse_feats, dense_feats=dense_feats)  # shape (N, 1)
```

### DeepFM

```python
from deepfm import DeepFM
import numpy as np

N_USERS, N_ITEMS, N_CATS, N = 200, 200, 5, 30000

# Sparse categorical features (integer IDs, 0-indexed)
user_ids = np.random.randint(0, N_USERS, N).astype(np.int32)
item_ids = np.random.randint(0, N_ITEMS, N).astype(np.int32)
cat_ids  = np.random.randint(0, N_CATS,  N).astype(np.int32)
sparse_feats = [user_ids, item_ids, cat_ids]   # one array per field

# Optional dense features
dense_feats = np.random.randn(N, 5).astype(np.float32)

y = np.random.binomial(1, 0.1, N).astype(np.float32)

model = DeepFM(
    field_dims=[N_USERS, N_ITEMS, N_CATS],  # vocabulary size per categorical field
    embed_dim=16,                            # k — shared by FM and DNN
    dense_feat_dim=5,                        # number of continuous features (0 = none)
    dnn_units=[256, 128, 64],
    dropout_rate=0.1,
)
model.fit(sparse_feats, y, dense_feats=dense_feats, epochs=20, batch_size=256)
probs = model.predict(sparse_feats, dense_feats=dense_feats)  # shape (N, 1)
```

### MMoE (Multi-gate Mixture of Experts)

```python
from mmoe import MMoE
import numpy as np

X      = np.random.randn(30000, 30).astype(np.float32)
y_ctr  = np.random.binomial(1, 0.15, 30000).astype(np.float32)
y_cvr  = np.random.binomial(1, 0.05, 30000).astype(np.float32)

model = MMoE(
    input_dim=30,
    num_experts=8,             # K shared expert DNNs
    expert_units=[128, 64],
    task_names=["ctr", "cvr"],
    tower_units=[64, 32],
)
model.fit(X, labels={"ctr": y_ctr, "cvr": y_cvr}, epochs=20, batch_size=256)

preds = model.predict(X)
p_ctr = preds["ctr"]          # shape (N, 1)
p_cvr = preds["cvr"]          # shape (N, 1)
```

### PLE (Progressive Layered Extraction)

```python
from ple import PLE
import numpy as np

X      = np.random.randn(30000, 30).astype(np.float32)
y_ctr  = np.random.binomial(1, 0.15, 30000).astype(np.float32)
y_cvr  = np.random.binomial(1, 0.05, 30000).astype(np.float32)

model = PLE(
    input_dim=30,
    num_task_experts=3,       # task-specific experts per task (not shared)
    num_shared_experts=3,     # shared experts across all tasks
    expert_units=[128, 64],
    num_extraction_layers=2,  # L=1 is equivalent to plain CGC
    task_names=["ctr", "cvr"],
    tower_units=[64, 32],
)
model.fit(X, labels={"ctr": y_ctr, "cvr": y_cvr}, epochs=20, batch_size=256)

preds = model.predict(X)
p_ctr = preds["ctr"]          # shape (N, 1)
p_cvr = preds["cvr"]          # shape (N, 1)
```

## References

| Model | Paper |
| --- | --- |
| Wide & Deep | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) `2016` `Google` |
| ESMM | [Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931) `2018` `Alibaba` |
| DCN | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123) `2017` `Google` |
| DCN-v2 | [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535) `2021` `Google` |
| DIN | [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978) `2018` `Alibaba` |
| MMoE | [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007) `2018` `Google` |
| PLE | [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalization](https://dl.acm.org/doi/10.1145/3383313.3412236) `2020` `Tencent` |
| DIEN | [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/abs/1809.03672) `2019` `Alibaba` |
| DeepFM | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) `2017` `Huawei` |
| AutoInt | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921) `2019` `BIT` |

---

# Test Framework

This directory contains a comprehensive test framework for evaluating Click-Through Rate (CTR) and Conversion Rate (CVR) prediction models.

## Overview

All test-related files are in the **[test/](test/)** directory:

- **[test/test_data.py](test/test_data.py)**: Synthetic data generator for realistic ad prediction datasets
- **[test/test_ctr.py](test/test_ctr.py)**: Test framework for CTR prediction models (binary classification)
- **[test/test_cvr.py](test/test_cvr.py)**: Test framework for CVR prediction models (binary classification)
- **[test/tf_adapter.py](test/tf_adapter.py)**: TensorFlow/Keras model adapters for seamless integration
- **[test/TENSORFLOW_GUIDE.md](test/TENSORFLOW_GUIDE.md)**: Complete guide for using TensorFlow models
- **[test/example_usage.py](test/example_usage.py)**: Example demonstrating how to use the test framework

## Quick Start

### 1. Run Example

```bash
cd test
python example_usage.py
```

### 2. Test Your Model

```python
from test.test_ctr import CTRModelTest
from test.test_data import AdDataGenerator

# Your model should implement:
# - fit(X: Dict, y: np.ndarray, **kwargs) -> None
# - predict(X: Dict) -> np.ndarray

model = YourCTRModel()
test = CTRModelTest(model, model_name="Your Model Name")
metrics = test.run_full_test(n_samples=10000)
```

## Data Format

### Features Dictionary

Both CTR and CVR datasets provide features as a dictionary where keys are feature names and values are numpy arrays:

```python
{
    'user_id': np.ndarray,        # Categorical
    'user_age': np.ndarray,        # Numerical
    'item_id': np.ndarray,         # Categorical
    'item_price': np.ndarray,      # Numerical
    # ... more features
}
```

### CTR Features

**Task**: Binary Classification (click or not)
**Typical Rate**: 10-20%

**Sparse Features** (categorical):
- `user_id`: User identifier
- `item_id`: Item/ad identifier
- `item_category`: Item category
- `user_gender`: User gender (0: Female, 1: Male)
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0-6)
- `device_type`: Device type (0: Mobile, 1: Desktop, 2: Tablet)

**Dense Features** (numerical):
- `user_age`: User age (18-70)
- `item_price`: Item price
- `user_click_history`: Historical click count
- `user_show_history`: Historical impression count
- `user_historical_ctr`: Historical CTR

**Labels**: Binary (0: no click, 1: click)

### CVR Features

**Task**: Binary Classification (conversion or not)
**Typical Rate**: 1-5% (much lower than CTR)
**Key Characteristics**:
- Conversion delay issues (users may click now but convert later)
- Strong dependency on click behavior (CVR is conditioned on CTR)
- Much sparser signal than CTR

**Sparse Features** (categorical):
- `user_id`: User identifier
- `item_id`: Item/ad identifier
- `item_category`: Item category
- `user_gender`: User gender
- `user_income_level`: Income level (0-4: Low to High)
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0-6)
- `device_type`: Device type

**Dense Features** (numerical):
- `user_age`: User age (18-70)
- `item_price`: Item price
- `item_rating`: Item rating (3.0-5.0)
- `is_weekend`: Weekend indicator (0 or 1)
- `user_purchase_history`: Historical purchase count
- `user_avg_purchase_value`: Average purchase value
- `user_days_since_last_purchase`: Days since last purchase
- `user_click_history`: Historical click count
- `user_show_history`: Historical impression count

**Labels**: Binary (0: no conversion, 1: conversion)

## Model Requirements

### CTR Models

Your CTR model class must implement:

```python
class YourCTRModel:
    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray, **kwargs) -> None:
        """
        Train the model.

        Args:
            X: Dictionary of features
            y: Binary labels (0 or 1)
            **kwargs: Additional training arguments
        """
        pass

    def predict(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict click probabilities.

        Args:
            X: Dictionary of features

        Returns:
            Array of probabilities in range [0, 1]
        """
        pass
```

### CVR Models

Your CVR model class must implement:

```python
class YourCVRModel:
    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray, **kwargs) -> None:
        """
        Train the model.

        Args:
            X: Dictionary of features
            y: Binary labels (0: no conversion, 1: conversion)
            **kwargs: Additional training arguments
        """
        pass

    def predict(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict conversion probabilities.

        Args:
            X: Dictionary of features

        Returns:
            Array of probabilities in range [0, 1]
        """
        pass
```

## Evaluation Metrics

### CTR Metrics

- **AUC**: Area Under ROC Curve (higher is better, > 0.5 is baseline)
- **Log Loss**: Cross-entropy loss (lower is better)
- **Accuracy**: Classification accuracy
- **Precision**: Positive prediction precision
- **Recall**: True positive rate
- **F1 Score**: Harmonic mean of precision and recall

### CVR Metrics

Same as CTR (both are binary classification):
- **AUC**: Area Under ROC Curve
- **Log Loss**: Cross-entropy loss
- **Accuracy**: Classification accuracy
- **Precision**: Positive prediction precision
- **Recall**: True positive rate
- **F1 Score**: Harmonic mean of precision and recall

## Advanced Usage

### Custom Data Generation

```python
from test.test_data import AdDataGenerator

generator = AdDataGenerator(seed=42)

# Generate CTR data (typical rate: 10-20%)
train_X, train_y, test_X, test_y = generator.generate_ctr_data(
    n_samples=10000,
    n_users=1000,
    n_items=500,
    n_categories=20,
    train_ratio=0.8
)

# Generate CVR data (typical rate: 1-5%, much lower than CTR)
train_X, train_y, test_X, test_y = generator.generate_cvr_data(
    n_samples=10000,
    n_users=1000,
    n_items=500,
    n_categories=20,
    train_ratio=0.8
)
```

### Step-by-Step Testing

```python
from test.test_ctr import CTRModelTest

model = YourCTRModel()
test = CTRModelTest(model, model_name="Your Model")

# Step 1: Prepare data
train_X, train_y, test_X, test_y = test.prepare_data(n_samples=10000)

# Step 2: Train model
test.train_model(train_X, train_y, epochs=10, batch_size=256)

# Step 3: Evaluate
metrics = test.evaluate_model(test_X, test_y)

print(f"AUC: {metrics['auc']:.4f}")
```

### Custom Training Parameters

```python
# Pass custom parameters to model.fit()
metrics = test.run_full_test(
    n_samples=10000,
    epochs=20,           # Passed to model.fit()
    batch_size=256,      # Passed to model.fit()
    learning_rate=0.001  # Passed to model.fit()
)
```

## Popular Models to Implement

### Feature Interaction Models
1. **Neural Factorization Machine** (NFM) - Bi-interaction pooling
2. **Extreme Deep Factorization Machine** (xDeepFM) - Compressed interaction network

## Implementation Tips

### 1. Feature Processing
- **Sparse features**: Use embedding layers for categorical features
- **Dense features**: Consider normalization (min-max or standardization)
- **Feature combinations**: Most modern CTR models focus on learning feature interactions

### 2. Architecture Design
- **CTR models**: Use sigmoid activation for binary classification
- **CVR models**: Use sigmoid activation for binary classification
- **Embedding size**: Common choices are 8, 16, 32, 64 (based on cardinality)
- **Deep layers**: Typical architecture is [256, 128, 64] or similar

### 3. Training Strategies
- **Class imbalance**: CVR is much sparser than CTR, consider:
  - Focal loss or weighted loss
  - Oversampling positive examples
  - Appropriate evaluation metrics (AUC > accuracy)
- **Multi-task learning**: Joint training of CTR and CVR can improve performance
- **Conversion delay**: In real systems, need to handle delayed feedback

### 4. Model Specific Considerations
- **DIN**: Requires user behavior sequence data
- **MMoE/PLE**: Designed for multi-task learning (CTR + CVR together)
- **DCN**: Explicitly learns cross features, good for sparse feature interactions

## TensorFlow/Keras Integration

The test framework works seamlessly with TensorFlow models! Use the provided adapters:

### Simple Approach (Concatenated Features)

```python
from test.tf_adapter import SimpleDNNModel
from test.test_ctr import CTRModelTest
from test.test_data import AdDataGenerator

# Prepare data
generator = AdDataGenerator(seed=42)
feature_info = generator.get_feature_info()

# Get feature lists and vocab sizes
sparse_features = feature_info['ctr']['sparse_features']
dense_features = feature_info['ctr']['dense_features']
train_X, train_y, test_X, test_y = generator.generate_ctr_data(n_samples=10000)

# Calculate vocab sizes
feature_vocab_sizes = {
    feat: int(train_X[feat].max() + 1) for feat in sparse_features
}

# Create TensorFlow model
model = SimpleDNNModel(
    sparse_features=sparse_features,
    dense_features=dense_features,
    feature_vocab_sizes=feature_vocab_sizes,
    hidden_units=[256, 128, 64]
)

# Test it!
test = CTRModelTest(model, model_name="TensorFlow DNN")
metrics = test.run_full_test(n_samples=10000, epochs=10, batch_size=256)
```

### Advanced Approach (Multi-Input with Embeddings)

For better performance with categorical features:

```python
from test.tf_adapter import EmbeddingDNNModel

model = EmbeddingDNNModel(
    sparse_features=sparse_features,
    dense_features=dense_features,
    feature_vocab_sizes=feature_vocab_sizes,
    embedding_dim=8,
    hidden_units=[256, 128, 64]
)

test = CTRModelTest(model, model_name="TensorFlow Embedding DNN")
metrics = test.run_full_test(n_samples=10000, epochs=10)
```

### Custom TensorFlow Models

Inherit from the adapter and override `build_model()`:

```python
from test.tf_adapter import TensorFlowCTRMultiInput
import tensorflow as tf
from tensorflow import keras

class MyCustomModel(TensorFlowCTRMultiInput):
    def build_model(self):
        # Your custom architecture here
        inputs = []
        embeddings = []

        # Build your model...
        # (see tf_adapter.py for examples)

        model = keras.Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['AUC']
        )
        return model
```

See **[test/tf_adapter.py](test/tf_adapter.py)** and **[test/TENSORFLOW_GUIDE.md](test/TENSORFLOW_GUIDE.md)** for complete examples and more details.

## Dependencies

- **Required**: NumPy
- **Optional**:
  - scikit-learn (for more accurate metrics, but not required)
  - TensorFlow 2.x (for using TensorFlow models)

The test framework will automatically use manual metric implementations if scikit-learn is not available.

## Example Output

```
############################################################
# CTR Model Test: Wide & Deep
############################################################

============================================================
Preparing CTR test data (10000 samples)...
============================================================
✓ Train samples: 8000, CTR: 0.1523
✓ Test samples: 2000, CTR: 0.1489
✓ Number of features: 12

============================================================
Training Wide & Deep...
============================================================
✓ Training completed

============================================================
Evaluating Wide & Deep...
============================================================

📊 Test Results:
  AUC: 0.7234
  Log Loss: 0.3421
  Accuracy: 0.8512
  Precision: 0.6234
  Recall: 0.4523
  F1 Score: 0.5234

✓ Sanity Checks:
  Predictions in [0,1]: True
  AUC > 0.5: True
  Mean predicted CTR: 0.1501 (actual: 0.1489)

############################################################
# Test Completed Successfully! ✓
############################################################
```

## License

This test framework is provided as-is for educational purposes.
