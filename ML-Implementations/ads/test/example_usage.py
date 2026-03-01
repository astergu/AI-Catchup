"""
Example usage of CTR and CVR test cases.

This shows how to use the test framework with your custom models.
"""

import numpy as np
try:
    # When run as a module
    from .test_ctr import CTRModelTest
    from .test_cvr import CVRModelTest
except ImportError:
    # When run as a script
    from test_ctr import CTRModelTest
    from test_cvr import CVRModelTest


# Example 1: Simple dummy CTR model
class DummyCTRModel:
    """A simple dummy model for demonstration."""

    def __init__(self):
        self.baseline_ctr = 0.1

    def fit(self, X, y, **kwargs):
        """Train by computing average CTR."""
        self.baseline_ctr = y.mean()
        print(f"  Learned baseline CTR: {self.baseline_ctr:.4f}")

    def predict(self, X):
        """Predict using baseline CTR."""
        n_samples = len(X['user_id'])
        return np.ones(n_samples) * self.baseline_ctr


# Example 2: Simple dummy CVR model
class DummyCVRModel:
    """A simple dummy model for demonstration."""

    def __init__(self):
        self.baseline_cvr = 0.02

    def fit(self, X, y, **kwargs):
        """Train by computing average conversion rate."""
        self.baseline_cvr = y.mean()
        print(f"  Learned baseline CVR: {self.baseline_cvr:.1%}")

    def predict(self, X):
        """Predict using baseline conversion probability."""
        n_samples = len(X['user_id'])
        return np.ones(n_samples) * self.baseline_cvr

    def predict_proba(self, X):
        """Predict using baseline conversion probability."""
        return self.predict(X)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("EXAMPLE: How to test your CTR/CVR models")
    print("="*70)

    # Test CTR model
    print("\n1. Testing CTR Model:")
    print("-" * 70)
    ctr_model = DummyCTRModel()
    ctr_test = CTRModelTest(ctr_model, model_name="Dummy CTR Model")
    ctr_metrics = ctr_test.run_full_test(n_samples=2000)

    # Test CVR model
    print("\n2. Testing CVR Model:")
    print("-" * 70)
    cvr_model = DummyCVRModel()
    cvr_test = CVRModelTest(cvr_model, model_name="Dummy CVR Model")
    cvr_metrics = cvr_test.run_full_test(n_samples=2000)

    print("\n" + "="*70)
    print("HOW TO USE WITH YOUR MODELS")
    print("="*70)
    print("""
Your model class should implement:

For CTR models (binary classification):
    - fit(X: Dict, y: np.ndarray, **kwargs) -> None
    - predict(X: Dict) -> np.ndarray  # Returns click probabilities [0, 1]

For CVR models (binary classification):
    - fit(X: Dict, y: np.ndarray, **kwargs) -> None
    - predict(X: Dict) -> np.ndarray  # Returns conversion probabilities [0, 1]

Note: CVR (1-5%) << CTR (10-20%)

Then test like this:
    from test_ctr import CTRModelTest

    model = YourCTRModel()
    test = CTRModelTest(model, model_name="Your Model Name")
    metrics = test.run_full_test(n_samples=10000)
    """)
