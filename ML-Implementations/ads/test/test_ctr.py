"""
Test case for Click-Through Rate (CTR) prediction models.

This module provides a standardized test framework for evaluating CTR prediction models.
All CTR models should be able to pass this test case.
"""

import numpy as np
from typing import Dict, Any

try:
    # When used as a package
    from .test_data import AdDataGenerator
except ImportError:
    # When run as a script
    from test_data import AdDataGenerator


class CTRModelTest:
    """
    Base test case for CTR prediction models.

    All CTR models should implement:
    - fit(train_features, train_labels) method
    - predict(test_features) method that returns click probabilities
    - predict_proba(test_features) method (optional, same as predict)
    """

    def __init__(self, model, model_name: str = "CTR Model"):
        """
        Initialize test case with a model.

        Args:
            model: CTR prediction model instance
            model_name: Name of the model for logging
        """
        self.model = model
        self.model_name = model_name
        self.generator = AdDataGenerator(seed=42)

    def prepare_data(self, n_samples: int = 10000) -> tuple:
        """Prepare train and test data."""
        print(f"\n{'='*60}")
        print(f"Preparing CTR test data ({n_samples} samples)...")
        print(f"{'='*60}")

        train_X, train_y, test_X, test_y = self.generator.generate_ctr_data(
            n_samples=n_samples,
            n_users=1000,
            n_items=500,
            n_categories=20,
            train_ratio=0.8
        )

        print(f"✓ Train samples: {len(train_y)}, CTR: {train_y.mean():.4f}")
        print(f"✓ Test samples: {len(test_y)}, CTR: {test_y.mean():.4f}")
        print(f"✓ Number of features: {len(train_X)}")

        return train_X, train_y, test_X, test_y

    def train_model(self, train_X: Dict, train_y: np.ndarray, **kwargs) -> None:
        """Train the model."""
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}...")
        print(f"{'='*60}")

        self.model.fit(train_X, train_y, **kwargs)
        print(f"✓ Training completed")

    def evaluate_model(self, test_X: Dict, test_y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {self.model_name}...")
        print(f"{'='*60}")

        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            y_pred = self.model.predict_proba(test_X)
        else:
            y_pred = self.model.predict(test_X)

        # Ensure predictions are probabilities
        assert np.all(y_pred >= 0) and np.all(y_pred <= 1), \
            "Predictions should be probabilities between 0 and 1"

        # Calculate metrics
        metrics = self._calculate_metrics(test_y, y_pred)

        # Print results
        print(f"\n📊 Test Results:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Log Loss: {metrics['log_loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")

        # Sanity checks
        print(f"\n✓ Sanity Checks:")
        print(f"  Predictions in [0,1]: {np.all(y_pred >= 0) and np.all(y_pred <= 1)}")
        print(f"  AUC > 0.5: {metrics['auc'] > 0.5}")
        print(f"  Mean predicted CTR: {y_pred.mean():.4f} (actual: {test_y.mean():.4f})")

        return metrics

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        try:
            from sklearn.metrics import (
                roc_auc_score, log_loss, accuracy_score,
                precision_score, recall_score, f1_score
            )

            # Binary predictions for classification metrics
            y_pred_binary = (y_pred >= 0.5).astype(int)

            metrics = {
                'auc': roc_auc_score(y_true, y_pred),
                'log_loss': log_loss(y_true, y_pred),
                'accuracy': accuracy_score(y_true, y_pred_binary),
                'precision': precision_score(y_true, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true, y_pred_binary, zero_division=0),
                'f1': f1_score(y_true, y_pred_binary, zero_division=0),
            }
        except ImportError:
            # Fallback to manual implementations if sklearn not available
            y_pred_binary = (y_pred >= 0.5).astype(int)
            metrics = {
                'auc': self._manual_auc(y_true, y_pred),
                'log_loss': self._manual_log_loss(y_true, y_pred),
                'accuracy': self._manual_accuracy(y_true, y_pred_binary),
                'precision': self._manual_precision(y_true, y_pred_binary),
                'recall': self._manual_recall(y_true, y_pred_binary),
                'f1': self._manual_f1(y_true, y_pred_binary),
            }

        return metrics

    def _manual_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Manual AUC calculation."""
        # Sort by prediction score
        sorted_indices = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[sorted_indices]

        # Count positive and negative samples
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        # Calculate AUC using trapezoidal rule
        tpr = np.cumsum(y_true_sorted) / n_pos
        fpr = np.cumsum(1 - y_true_sorted) / n_neg

        # Add boundary points
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])

        # Calculate area
        auc = 0
        for i in range(len(fpr) - 1):
            auc += (fpr[i+1] - fpr[i]) * (tpr[i+1] + tpr[i]) / 2

        return auc

    def _manual_log_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Manual log loss calculation."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def _manual_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Manual accuracy calculation."""
        return np.mean(y_true == y_pred)

    def _manual_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Manual precision calculation."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    def _manual_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Manual recall calculation."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    def _manual_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Manual F1 score calculation."""
        precision = self._manual_precision(y_true, y_pred)
        recall = self._manual_recall(y_true, y_pred)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def run_full_test(self, n_samples: int = 10000, **train_kwargs) -> Dict[str, float]:
        """
        Run complete test pipeline: data preparation, training, and evaluation.

        Args:
            n_samples: Number of samples to generate
            **train_kwargs: Additional arguments for model training

        Returns:
            Dictionary containing evaluation metrics
        """
        print(f"\n{'#'*60}")
        print(f"# CTR Model Test: {self.model_name}")
        print(f"{'#'*60}")

        # Prepare data
        train_X, train_y, test_X, test_y = self.prepare_data(n_samples)

        # Train model
        self.train_model(train_X, train_y, **train_kwargs)

        # Evaluate model
        metrics = self.evaluate_model(test_X, test_y)

        print(f"\n{'#'*60}")
        print(f"# Test Completed Successfully! ✓")
        print(f"{'#'*60}\n")

        return metrics


def test_baseline_model():
    """Test with a simple baseline model."""
    from sklearn.linear_model import LogisticRegression

    class SimpleBaselineModel:
        """Simple baseline using logistic regression on dense features only."""

        def __init__(self):
            self.model = LogisticRegression(max_iter=100)
            self.dense_features = ['user_age', 'item_price', 'user_click_history',
                                  'user_show_history', 'user_historical_ctr']

        def fit(self, X: Dict, y: np.ndarray, **kwargs):
            """Train the model."""
            X_dense = self._prepare_features(X)
            self.model.fit(X_dense, y)

        def predict(self, X: Dict) -> np.ndarray:
            """Predict click probabilities."""
            X_dense = self._prepare_features(X)
            return self.model.predict_proba(X_dense)[:, 1]

        def predict_proba(self, X: Dict) -> np.ndarray:
            """Predict click probabilities."""
            return self.predict(X)

        def _prepare_features(self, X: Dict) -> np.ndarray:
            """Extract and stack dense features."""
            features = []
            for feat in self.dense_features:
                features.append(X[feat].reshape(-1, 1))
            return np.hstack(features)

    # Run test
    model = SimpleBaselineModel()
    test = CTRModelTest(model, model_name="Logistic Regression Baseline")
    metrics = test.run_full_test(n_samples=5000)

    # Assert minimum performance
    assert metrics['auc'] > 0.5, "AUC should be better than random"
    print("✓ Baseline test passed!")


if __name__ == '__main__':
    print("Running CTR Model Test Suite\n")
    test_baseline_model()
