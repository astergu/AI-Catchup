"""
TensorFlow model adapter for CTR/CVR test framework.

This module provides utilities to integrate TensorFlow/Keras models with the test framework.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")


class TensorFlowCTRAdapter:
    """
    Adapter to use TensorFlow models with CTR test framework.

    Your TensorFlow model should be built and compiled before wrapping.
    This adapter handles:
    - Converting dict of features to TensorFlow inputs
    - Feature preprocessing (embeddings, normalization)
    - Proper train/predict interface
    """

    def __init__(
        self,
        model: Optional['keras.Model'] = None,
        sparse_features: Optional[List[str]] = None,
        dense_features: Optional[List[str]] = None,
        embedding_dims: Optional[Dict[str, int]] = None,
        feature_vocab_sizes: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the adapter.

        Args:
            model: Pre-built Keras model (optional, can be built in build_model)
            sparse_features: List of categorical feature names
            dense_features: List of numerical feature names
            embedding_dims: Dict mapping feature name to embedding dimension
            feature_vocab_sizes: Dict mapping feature name to vocabulary size
        """
        self.model = model
        self.sparse_features = sparse_features or []
        self.dense_features = dense_features or []
        self.embedding_dims = embedding_dims or {}
        self.feature_vocab_sizes = feature_vocab_sizes or {}

    def build_model(self) -> 'keras.Model':
        """
        Override this method to build your custom model.
        Should return a compiled Keras model.
        """
        raise NotImplementedError("Subclass should implement build_model()")

    def _prepare_features(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert feature dictionary to model input.

        For simple models, concatenate all features into a single array.
        For complex models (multi-input), return dict or list of arrays.
        """
        features = []

        # Add sparse features (as integers for embedding)
        for feat in self.sparse_features:
            features.append(X[feat].reshape(-1, 1))

        # Add dense features
        for feat in self.dense_features:
            features.append(X[feat].reshape(-1, 1))

        # Concatenate all features
        return np.hstack(features)

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray, **kwargs):
        """
        Train the model.

        Args:
            X: Dictionary of features
            y: Labels
            **kwargs: Additional arguments passed to model.fit()
        """
        if self.model is None:
            self.model = self.build_model()

        # Prepare features
        X_prepared = self._prepare_features(X)

        # Default training parameters
        fit_params = {
            'epochs': kwargs.get('epochs', 10),
            'batch_size': kwargs.get('batch_size', 256),
            'verbose': kwargs.get('verbose', 1),
            'validation_split': kwargs.get('validation_split', 0.1),
        }

        # Train model
        self.model.fit(X_prepared, y, **fit_params)

    def predict(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict probabilities.

        Args:
            X: Dictionary of features

        Returns:
            Array of probabilities
        """
        X_prepared = self._prepare_features(X)
        predictions = self.model.predict(X_prepared, verbose=0)
        return predictions.flatten()

    def predict_proba(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """Alias for predict (for compatibility)."""
        return self.predict(X)


class TensorFlowCTRMultiInput:
    """
    Advanced adapter for multi-input TensorFlow models.

    This adapter creates separate inputs for each feature, which is better for
    models with embeddings and feature interactions (DeepFM, DCN, etc.).
    """

    def __init__(
        self,
        sparse_features: List[str],
        dense_features: List[str],
        feature_vocab_sizes: Dict[str, int],
        embedding_dim: int = 8
    ):
        """
        Initialize the adapter.

        Args:
            sparse_features: List of categorical feature names
            dense_features: List of numerical feature names
            feature_vocab_sizes: Dict mapping feature name to vocabulary size
            embedding_dim: Dimension for embeddings (default 8)
        """
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.feature_vocab_sizes = feature_vocab_sizes
        self.embedding_dim = embedding_dim
        self.model = None

    def build_model(self) -> 'keras.Model':
        """
        Override this method to build your custom model.
        Should return a compiled Keras model with multi-input architecture.
        """
        raise NotImplementedError("Subclass should implement build_model()")

    def _prepare_features(self, X: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert feature dictionary to model input dict.

        Each feature becomes a separate input to the model.
        """
        inputs = {}

        # Sparse features
        for feat in self.sparse_features:
            inputs[feat] = X[feat]

        # Dense features
        for feat in self.dense_features:
            inputs[feat] = X[feat]

        return inputs

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray, **kwargs):
        """Train the model."""
        if self.model is None:
            self.model = self.build_model()

        X_prepared = self._prepare_features(X)

        fit_params = {
            'epochs': kwargs.get('epochs', 10),
            'batch_size': kwargs.get('batch_size', 256),
            'verbose': kwargs.get('verbose', 1),
            'validation_split': kwargs.get('validation_split', 0.1),
        }

        self.model.fit(X_prepared, y, **fit_params)

    def predict(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict probabilities."""
        X_prepared = self._prepare_features(X)
        predictions = self.model.predict(X_prepared, verbose=0)
        return predictions.flatten()

    def predict_proba(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """Alias for predict."""
        return self.predict(X)


# Example: Simple DNN model for CTR prediction
class SimpleDNNModel(TensorFlowCTRAdapter):
    """
    Example: Simple Deep Neural Network for CTR prediction.
    """

    def __init__(
        self,
        sparse_features: List[str],
        dense_features: List[str],
        feature_vocab_sizes: Dict[str, int],
        hidden_units: List[int] = [256, 128, 64]
    ):
        super().__init__(
            sparse_features=sparse_features,
            dense_features=dense_features,
            feature_vocab_sizes=feature_vocab_sizes
        )
        self.hidden_units = hidden_units

    def build_model(self) -> 'keras.Model':
        """Build a simple DNN model."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required")

        # Calculate input dimension
        input_dim = len(self.sparse_features) + len(self.dense_features)

        # Build model
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.BatchNormalization(),
        ])

        # Add hidden layers
        for units in self.hidden_units:
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.Dropout(0.2))

        # Output layer
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['AUC', 'accuracy']
        )

        return model


# Example: Multi-input model with embeddings
class EmbeddingDNNModel(TensorFlowCTRMultiInput):
    """
    Example: DNN with embedding layers for categorical features.
    """

    def __init__(
        self,
        sparse_features: List[str],
        dense_features: List[str],
        feature_vocab_sizes: Dict[str, int],
        embedding_dim: int = 8,
        hidden_units: List[int] = [256, 128, 64]
    ):
        super().__init__(
            sparse_features=sparse_features,
            dense_features=dense_features,
            feature_vocab_sizes=feature_vocab_sizes,
            embedding_dim=embedding_dim
        )
        self.hidden_units = hidden_units

    def build_model(self) -> 'keras.Model':
        """Build DNN model with embeddings."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required")

        inputs = []
        embeddings = []

        # Sparse feature inputs with embeddings
        for feat in self.sparse_features:
            vocab_size = self.feature_vocab_sizes[feat]
            inp = keras.layers.Input(shape=(1,), name=feat, dtype='int32')
            inputs.append(inp)

            # Embedding layer
            emb = keras.layers.Embedding(
                vocab_size,
                self.embedding_dim,
                name=f'{feat}_embedding'
            )(inp)
            emb = keras.layers.Flatten()(emb)
            embeddings.append(emb)

        # Dense feature inputs
        dense_inputs = []
        for feat in self.dense_features:
            inp = keras.layers.Input(shape=(1,), name=feat, dtype='float32')
            inputs.append(inp)
            dense_inputs.append(inp)

        # Concatenate dense inputs
        if dense_inputs:
            dense_concat = keras.layers.Concatenate()(dense_inputs)
            dense_concat = keras.layers.BatchNormalization()(dense_concat)
            embeddings.append(dense_concat)

        # Concatenate all features
        x = keras.layers.Concatenate()(embeddings)

        # Deep layers
        for units in self.hidden_units:
            x = keras.layers.Dense(units, activation='relu')(x)
            x = keras.layers.Dropout(0.2)(x)

        # Output
        output = keras.layers.Dense(1, activation='sigmoid', name='ctr')(x)

        # Build model
        model = keras.Model(inputs=inputs, outputs=output)

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['AUC', 'accuracy']
        )

        return model


if __name__ == '__main__':
    # Example usage
    if TF_AVAILABLE:
        try:
            from .test_data import AdDataGenerator
            from .test_ctr import CTRModelTest
        except ImportError:
            from test_data import AdDataGenerator
            from test_ctr import CTRModelTest

        print("Testing TensorFlow model with CTR test framework\n")

        # Generate data
        generator = AdDataGenerator(seed=42)
        train_X, train_y, test_X, test_y = generator.generate_ctr_data(n_samples=5000)

        # Get feature info
        feature_info = generator.get_feature_info()
        sparse_features = feature_info['ctr']['sparse_features']
        dense_features = feature_info['ctr']['dense_features']

        # Calculate vocab sizes from data
        feature_vocab_sizes = {}
        for feat in sparse_features:
            feature_vocab_sizes[feat] = int(max(train_X[feat].max(), test_X[feat].max()) + 1)

        print(f"Sparse features: {sparse_features}")
        print(f"Dense features: {dense_features}")
        print(f"Vocab sizes: {feature_vocab_sizes}\n")

        # Test simple DNN model
        print("="*70)
        print("Testing Simple DNN Model")
        print("="*70)
        model = SimpleDNNModel(
            sparse_features=sparse_features,
            dense_features=dense_features,
            feature_vocab_sizes=feature_vocab_sizes,
            hidden_units=[128, 64]
        )

        test = CTRModelTest(model, model_name="TensorFlow Simple DNN")
        metrics = test.run_full_test(
            n_samples=5000,
            epochs=5,
            batch_size=256,
            verbose=0
        )

        # Test embedding DNN model
        print("\n" + "="*70)
        print("Testing Embedding DNN Model")
        print("="*70)
        model2 = EmbeddingDNNModel(
            sparse_features=sparse_features,
            dense_features=dense_features,
            feature_vocab_sizes=feature_vocab_sizes,
            embedding_dim=8,
            hidden_units=[128, 64]
        )

        test2 = CTRModelTest(model2, model_name="TensorFlow Embedding DNN")
        metrics2 = test2.run_full_test(
            n_samples=5000,
            epochs=5,
            batch_size=256,
            verbose=0
        )

        print("\n✓ TensorFlow integration test passed!")
    else:
        print("TensorFlow not available. Install with: pip install tensorflow")
