"""
Test framework for CTR/CVR prediction models.

This package provides:
- Synthetic data generation (test_data.py)
- CTR model testing (test_ctr.py)
- CVR model testing (test_cvr.py)
- TensorFlow integration (tf_adapter.py)
"""

from .test_data import AdDataGenerator
from .test_ctr import CTRModelTest
from .test_cvr import CVRModelTest

try:
    from .tf_adapter import (
        TensorFlowCTRAdapter,
        TensorFlowCTRMultiInput,
        SimpleDNNModel,
        EmbeddingDNNModel
    )
except ImportError:
    # TensorFlow not available
    pass

__all__ = [
    'AdDataGenerator',
    'CTRModelTest',
    'CVRModelTest',
    'TensorFlowCTRAdapter',
    'TensorFlowCTRMultiInput',
    'SimpleDNNModel',
    'EmbeddingDNNModel',
]
