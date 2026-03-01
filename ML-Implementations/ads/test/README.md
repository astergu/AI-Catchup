# CTR/CVR Model Test Framework

Comprehensive test framework for evaluating Click-Through Rate (CTR) and Conversion Rate (CVR) prediction models.

## Files

- **[test_data.py](test_data.py)**: Synthetic data generator for realistic ad prediction datasets
- **[test_ctr.py](test_ctr.py)**: Test framework for CTR prediction models (binary classification)
- **[test_cvr.py](test_cvr.py)**: Test framework for CVR prediction models (binary classification)
- **[tf_adapter.py](tf_adapter.py)**: TensorFlow/Keras model adapters for seamless integration
- **[TENSORFLOW_GUIDE.md](TENSORFLOW_GUIDE.md)**: Complete guide for using TensorFlow models
- **[example_usage.py](example_usage.py)**: Example demonstrating how to use the test framework

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

# Generate data
generator = AdDataGenerator(seed=42)
train_X, train_y, test_X, test_y = generator.generate_ctr_data(n_samples=10000)

# Create and test your model
model = YourCTRModel()
test = CTRModelTest(model, model_name="Your Model Name")
metrics = test.run_full_test(n_samples=10000)
```

### 3. TensorFlow Integration

```python
from test.tf_adapter import EmbeddingDNNModel
from test.test_ctr import CTRModelTest
from test.test_data import AdDataGenerator

# Generate data and get feature info
generator = AdDataGenerator(seed=42)
feature_info = generator.get_feature_info()
sparse_features = feature_info['ctr']['sparse_features']
dense_features = feature_info['ctr']['dense_features']

# Calculate vocab sizes
train_X, _, _, _ = generator.generate_ctr_data(n_samples=10000)
feature_vocab_sizes = {
    feat: int(train_X[feat].max() + 1) for feat in sparse_features
}

# Create TensorFlow model
model = EmbeddingDNNModel(
    sparse_features=sparse_features,
    dense_features=dense_features,
    feature_vocab_sizes=feature_vocab_sizes,
    embedding_dim=8,
    hidden_units=[256, 128, 64]
)

# Test it!
test = CTRModelTest(model, model_name="TensorFlow DNN")
metrics = test.run_full_test(n_samples=10000, epochs=10, batch_size=256)
```

## Documentation

See the main [README.md](../README.md) for detailed documentation on:
- Data format and features
- Model requirements
- Evaluation metrics
- Implementation tips
- Popular models to implement

See [TENSORFLOW_GUIDE.md](TENSORFLOW_GUIDE.md) for TensorFlow-specific documentation.

## Dependencies

- **Required**: NumPy
- **Optional**:
  - scikit-learn (for more accurate metrics, but not required)
  - TensorFlow 2.x (for using TensorFlow models)

The test framework will automatically use manual metric implementations if scikit-learn is not available.
