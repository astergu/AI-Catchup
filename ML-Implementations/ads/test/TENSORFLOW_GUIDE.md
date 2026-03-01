# TensorFlow Integration Guide

This guide shows how to use TensorFlow/Keras models with the CTR/CVR test framework.

## Installation

```bash
pip install tensorflow numpy
# Optional: pip install scikit-learn
```

## Quick Start

### Option 1: Use Pre-built Adapters

The simplest way to get started:

```python
from tf_adapter import SimpleDNNModel, EmbeddingDNNModel
from test_ctr import CTRModelTest
from test_data import AdDataGenerator

# Generate data
generator = AdDataGenerator(seed=42)
feature_info = generator.get_feature_info()

sparse_features = feature_info['ctr']['sparse_features']
dense_features = feature_info['ctr']['dense_features']

# Get vocab sizes
train_X, _, _, _ = generator.generate_ctr_data(n_samples=10000)
feature_vocab_sizes = {
    feat: int(train_X[feat].max() + 1) for feat in sparse_features
}

# Create and test model
model = EmbeddingDNNModel(
    sparse_features=sparse_features,
    dense_features=dense_features,
    feature_vocab_sizes=feature_vocab_sizes,
    embedding_dim=8,
    hidden_units=[256, 128, 64]
)

test = CTRModelTest(model, model_name="TensorFlow DNN")
metrics = test.run_full_test(n_samples=10000, epochs=10, batch_size=256)
```

### Option 2: Build Your Own Model

Inherit from the adapter and implement `build_model()`:

```python
from tf_adapter import TensorFlowCTRMultiInput
import tensorflow as tf
from tensorflow import keras

class WideAndDeep(TensorFlowCTRMultiInput):
    """Wide & Deep model implementation."""

    def build_model(self):
        # Wide inputs (for memorization)
        wide_inputs = []
        wide_features = []

        for feat in self.sparse_features:
            inp = keras.layers.Input(shape=(1,), name=feat, dtype='int32')
            wide_inputs.append(inp)
            # One-hot encode for wide part
            one_hot = keras.layers.CategoryEncoding(
                num_tokens=self.feature_vocab_sizes[feat],
                output_mode='one_hot'
            )(inp)
            wide_features.append(keras.layers.Flatten()(one_hot))

        wide = keras.layers.Concatenate()(wide_features)

        # Deep inputs (for generalization)
        deep_inputs = []
        deep_features = []

        # Embeddings for categorical features
        for feat in self.sparse_features:
            inp = wide_inputs[self.sparse_features.index(feat)]
            emb = keras.layers.Embedding(
                self.feature_vocab_sizes[feat],
                self.embedding_dim
            )(inp)
            deep_features.append(keras.layers.Flatten()(emb))

        # Dense features
        for feat in self.dense_features:
            inp = keras.layers.Input(shape=(1,), name=feat, dtype='float32')
            deep_inputs.append(inp)
            deep_features.append(inp)

        # Deep network
        deep = keras.layers.Concatenate()(deep_features)
        deep = keras.layers.BatchNormalization()(deep)
        deep = keras.layers.Dense(256, activation='relu')(deep)
        deep = keras.layers.Dropout(0.2)(deep)
        deep = keras.layers.Dense(128, activation='relu')(deep)
        deep = keras.layers.Dropout(0.2)(deep)
        deep = keras.layers.Dense(64, activation='relu')(deep)

        # Combine wide and deep
        combined = keras.layers.Concatenate()([wide, deep])
        output = keras.layers.Dense(1, activation='sigmoid')(combined)

        # Build model
        all_inputs = wide_inputs + deep_inputs
        model = keras.Model(inputs=all_inputs, outputs=output)

        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='binary_crossentropy',
            metrics=['AUC', 'accuracy']
        )

        return model

# Use it
model = WideAndDeep(
    sparse_features=sparse_features,
    dense_features=dense_features,
    feature_vocab_sizes=feature_vocab_sizes,
    embedding_dim=8
)

test = CTRModelTest(model, model_name="Wide & Deep")
metrics = test.run_full_test(n_samples=10000, epochs=10)
```

## Adapter Types

### 1. TensorFlowCTRAdapter (Simple)

Best for: Quick prototyping, simple models

- Concatenates all features into a single input
- Easy to use, but less flexible
- Good for models without embeddings

```python
class MyModel(TensorFlowCTRAdapter):
    def build_model(self):
        input_dim = len(self.sparse_features) + len(self.dense_features)

        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
        return model
```

### 2. TensorFlowCTRMultiInput (Advanced)

Best for: Production models, models with embeddings

- Each feature is a separate input
- Proper embedding handling for categorical features
- More flexible for complex architectures
- Recommended for CTR/CVR models

```python
class MyModel(TensorFlowCTRMultiInput):
    def build_model(self):
        # Each feature gets its own input and embedding
        # See examples above
        pass
```

## Feature Processing

### Categorical Features (Sparse)

The adapter automatically handles categorical features. You just need to specify vocab sizes:

```python
feature_vocab_sizes = {
    'user_id': 1000,    # 1000 unique users
    'item_id': 500,     # 500 unique items
    'category': 20,     # 20 categories
    # ...
}
```

Or calculate from data:

```python
feature_vocab_sizes = {
    feat: int(train_X[feat].max() + 1)
    for feat in sparse_features
}
```

### Numerical Features (Dense)

Dense features are passed as-is. Consider normalization:

```python
# In your build_model():
dense = keras.layers.Concatenate()(dense_features)
dense = keras.layers.BatchNormalization()(dense)  # Normalize
```

## Training Parameters

Pass training parameters through `run_full_test()` or `fit()`:

```python
metrics = test.run_full_test(
    n_samples=10000,
    epochs=20,              # Number of epochs
    batch_size=256,         # Batch size
    verbose=1,              # Keras verbosity
    validation_split=0.1,   # Validation split
)
```

## Common Architectures

### 1. Simple DNN

```python
from tf_adapter import SimpleDNNModel

model = SimpleDNNModel(
    sparse_features=sparse_features,
    dense_features=dense_features,
    feature_vocab_sizes=feature_vocab_sizes,
    hidden_units=[256, 128, 64]
)
```

### 2. DNN with Embeddings

```python
from tf_adapter import EmbeddingDNNModel

model = EmbeddingDNNModel(
    sparse_features=sparse_features,
    dense_features=dense_features,
    feature_vocab_sizes=feature_vocab_sizes,
    embedding_dim=8,
    hidden_units=[256, 128, 64]
)
```

### 3. Wide & Deep

See example above.

### 4. DeepFM

```python
class DeepFM(TensorFlowCTRMultiInput):
    def build_model(self):
        # FM part: pairwise interactions
        # Deep part: DNN
        # Combine both
        pass
```

### 5. DCN (Deep & Cross Network)

```python
class DCN(TensorFlowCTRMultiInput):
    def build_model(self):
        # Cross network: explicit feature interactions
        # Deep network: implicit patterns
        # Combine both
        pass
```

## Tips & Best Practices

### 1. Embedding Dimensions

Common choices based on cardinality:

```python
def get_embedding_dim(vocab_size):
    if vocab_size < 10:
        return 4
    elif vocab_size < 100:
        return 8
    elif vocab_size < 1000:
        return 16
    elif vocab_size < 10000:
        return 32
    else:
        return 64
```

### 2. Handle Class Imbalance (CVR)

CVR has much lower positive rate (~1-5%):

```python
# Option 1: Class weights
pos_weight = (1 - cvr_rate) / cvr_rate
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    weighted_metrics=['AUC']
)

# Option 2: Focal loss
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        ce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return alpha_t * (1 - p_t) ** gamma * ce
    return loss

model.compile(optimizer='adam', loss=focal_loss())
```

### 3. Learning Rate Scheduling

```python
lr_schedule = keras.callbacks.ReduceLROnPlateau(
    monitor='val_auc',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

# In fit():
self.model.fit(X, y, callbacks=[lr_schedule], ...)
```

### 4. Early Stopping

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_auc',
    patience=3,
    restore_best_weights=True
)
```

### 5. Multi-Task Learning (CTR + CVR)

```python
class MultiTask(TensorFlowCTRMultiInput):
    def build_model(self):
        # Shared layers
        shared = self.build_shared_network()

        # Task-specific towers
        ctr_tower = keras.layers.Dense(64, activation='relu')(shared)
        ctr_output = keras.layers.Dense(1, activation='sigmoid', name='ctr')(ctr_tower)

        cvr_tower = keras.layers.Dense(64, activation='relu')(shared)
        cvr_output = keras.layers.Dense(1, activation='sigmoid', name='cvr')(cvr_tower)

        model = keras.Model(inputs=self.inputs, outputs=[ctr_output, cvr_output])

        model.compile(
            optimizer='adam',
            loss={'ctr': 'binary_crossentropy', 'cvr': 'binary_crossentropy'},
            metrics={'ctr': ['AUC'], 'cvr': ['AUC']}
        )

        return model
```

## Debugging

### Check Model Architecture

```python
model.model.summary()
```

### Verify Input Shapes

```python
for feat in sparse_features:
    print(f"{feat}: {train_X[feat].shape}, min={train_X[feat].min()}, max={train_X[feat].max()}")
```

### Monitor Training

```python
metrics = test.run_full_test(
    n_samples=10000,
    epochs=10,
    verbose=1  # Show Keras training progress
)
```

## Complete Example

See **[tf_adapter.py](tf_adapter.py)** for runnable examples, or run:

```bash
python tf_adapter.py
```

This will train and test both SimpleDNNModel and EmbeddingDNNModel if TensorFlow is installed.

## Next Steps

1. Implement Wide & Deep (see examples above)
2. Implement DeepFM
3. Implement DCN
4. Implement DIN (with user behavior sequences)
5. Implement MMoE/PLE for multi-task learning

All these models can use the same test framework with the TensorFlow adapters!
