"""
Synthetic data generator for CTR/CVR prediction model testing.

This module provides utilities to generate realistic ad prediction datasets
with common features found in real-world advertising systems.
"""

import numpy as np
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class AdDataGenerator:
    """Generate synthetic advertising data for testing CTR/CVR models."""

    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.

        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed

    def generate_ctr_data(
        self,
        n_samples: int = 10000,
        n_users: int = 1000,
        n_items: int = 500,
        n_categories: int = 20,
        train_ratio: float = 0.8
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Generate synthetic CTR prediction data.

        Args:
            n_samples: Number of samples to generate
            n_users: Number of unique users
            n_items: Number of unique items/ads
            n_categories: Number of item categories
            train_ratio: Ratio of training data

        Returns:
            Tuple of (train_features, train_labels, test_features, test_labels)
        """
        # Generate user features
        user_ids = np.random.randint(0, n_users, size=n_samples)
        user_age = np.random.randint(18, 70, size=n_samples)
        user_gender = np.random.randint(0, 2, size=n_samples)  # 0: Female, 1: Male

        # Generate item features
        item_ids = np.random.randint(0, n_items, size=n_samples)
        item_category = np.random.randint(0, n_categories, size=n_samples)
        item_price = np.random.lognormal(mean=3.0, sigma=1.0, size=n_samples)

        # Generate context features
        hour = np.random.randint(0, 24, size=n_samples)
        day_of_week = np.random.randint(0, 7, size=n_samples)
        device_type = np.random.randint(0, 3, size=n_samples)  # 0: Mobile, 1: Desktop, 2: Tablet

        # Generate behavioral features
        user_click_history = np.random.poisson(lam=5, size=n_samples)
        user_show_history = np.random.poisson(lam=50, size=n_samples)
        user_historical_ctr = np.clip(user_click_history / (user_show_history + 1), 0, 1)

        # Generate labels with realistic patterns
        # CTR depends on multiple factors
        click_prob = self._compute_ctr_probability(
            user_age, user_gender, user_historical_ctr,
            item_price, item_category, hour, day_of_week, device_type
        )
        labels = (np.random.random(n_samples) < click_prob).astype(np.float32)

        # Package features
        features = {
            'user_id': user_ids,
            'user_age': user_age,
            'user_gender': user_gender,
            'item_id': item_ids,
            'item_category': item_category,
            'item_price': item_price,
            'hour': hour,
            'day_of_week': day_of_week,
            'device_type': device_type,
            'user_click_history': user_click_history,
            'user_show_history': user_show_history,
            'user_historical_ctr': user_historical_ctr,
        }

        # Split train/test
        n_train = int(n_samples * train_ratio)
        train_features = {k: v[:n_train] for k, v in features.items()}
        test_features = {k: v[n_train:] for k, v in features.items()}
        train_labels = labels[:n_train]
        test_labels = labels[n_train:]

        return train_features, train_labels, test_features, test_labels

    def generate_cvr_data(
        self,
        n_samples: int = 10000,
        n_users: int = 1000,
        n_items: int = 500,
        n_categories: int = 20,
        train_ratio: float = 0.8
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Generate synthetic CVR (Conversion Rate) prediction data.

        CVR prediction is binary classification (conversion or not) with:
        - Much lower conversion rate than CTR (typically 1-5% vs 10-20%)
        - Conversion delay issues
        - Strong dependency on click behavior

        Args:
            n_samples: Number of samples to generate
            n_users: Number of unique users
            n_items: Number of unique items/ads
            n_categories: Number of item categories
            train_ratio: Ratio of training data

        Returns:
            Tuple of (train_features, train_labels, test_features, test_labels)
        """
        # Generate user features
        user_ids = np.random.randint(0, n_users, size=n_samples)
        user_age = np.random.randint(18, 70, size=n_samples)
        user_gender = np.random.randint(0, 2, size=n_samples)
        user_income_level = np.random.randint(0, 5, size=n_samples)  # 0-4: Low to High

        # Generate item features
        item_ids = np.random.randint(0, n_items, size=n_samples)
        item_category = np.random.randint(0, n_categories, size=n_samples)
        item_price = np.random.lognormal(mean=3.0, sigma=1.0, size=n_samples)
        item_rating = np.random.uniform(3.0, 5.0, size=n_samples)

        # Generate context features
        hour = np.random.randint(0, 24, size=n_samples)
        day_of_week = np.random.randint(0, 7, size=n_samples)
        device_type = np.random.randint(0, 3, size=n_samples)
        is_weekend = (day_of_week >= 5).astype(np.float32)

        # Generate behavioral features (CVR depends heavily on purchase history)
        user_purchase_history = np.random.poisson(lam=2, size=n_samples)
        user_avg_purchase_value = np.random.lognormal(mean=3.5, sigma=1.2, size=n_samples)
        user_days_since_last_purchase = np.random.exponential(scale=30, size=n_samples)

        # Generate click features (CVR happens after click)
        user_click_history = np.random.poisson(lam=5, size=n_samples)
        user_show_history = np.random.poisson(lam=50, size=n_samples)

        # Generate conversion labels with realistic patterns
        # Key: CVR is MUCH lower than CTR (1-5% vs 10-20%)
        conversion_prob = self._compute_cvr_probability(
            user_age, user_gender, user_income_level,
            item_price, item_rating, item_category,
            user_purchase_history, user_avg_purchase_value,
            user_days_since_last_purchase,
            hour, is_weekend, device_type
        )
        labels = (np.random.random(n_samples) < conversion_prob).astype(np.float32)

        # Package features
        features = {
            'user_id': user_ids,
            'user_age': user_age,
            'user_gender': user_gender,
            'user_income_level': user_income_level,
            'item_id': item_ids,
            'item_category': item_category,
            'item_price': item_price,
            'item_rating': item_rating,
            'hour': hour,
            'day_of_week': day_of_week,
            'device_type': device_type,
            'is_weekend': is_weekend,
            'user_purchase_history': user_purchase_history,
            'user_avg_purchase_value': user_avg_purchase_value,
            'user_days_since_last_purchase': user_days_since_last_purchase,
            'user_click_history': user_click_history,
            'user_show_history': user_show_history,
        }

        # Split train/test
        n_train = int(n_samples * train_ratio)
        train_features = {k: v[:n_train] for k, v in features.items()}
        test_features = {k: v[n_train:] for k, v in features.items()}
        train_labels = labels[:n_train]
        test_labels = labels[n_train:]

        return train_features, train_labels, test_features, test_labels

    def _compute_ctr_probability(
        self,
        user_age: np.ndarray,
        user_gender: np.ndarray,
        user_historical_ctr: np.ndarray,
        item_price: np.ndarray,
        item_category: np.ndarray,
        hour: np.ndarray,
        day_of_week: np.ndarray,
        device_type: np.ndarray
    ) -> np.ndarray:
        """Compute click probability based on features."""
        # Base probability
        prob = np.ones_like(user_age, dtype=np.float32) * 0.05

        # User age effect (younger users click more)
        prob += 0.05 * np.exp(-(user_age - 25) ** 2 / 500)

        # Gender effect
        prob += 0.02 * user_gender

        # Historical CTR is a strong signal
        prob += 0.3 * user_historical_ctr

        # Price effect (cheaper items get more clicks)
        prob += 0.05 * np.exp(-item_price / 50)

        # Category effect (some categories are more attractive)
        prob += 0.03 * np.sin(item_category / 2)

        # Hour effect (peak hours: 12-14, 19-22)
        prob += 0.03 * (np.sin((hour - 13) * np.pi / 12) + np.sin((hour - 20) * np.pi / 12))

        # Weekend effect
        prob += 0.02 * (day_of_week >= 5)

        # Mobile users click more
        prob += 0.03 * (device_type == 0)

        # Clip to valid probability range
        return np.clip(prob, 0.01, 0.5)

    def _compute_cvr_probability(
        self,
        user_age: np.ndarray,
        user_gender: np.ndarray,
        user_income_level: np.ndarray,
        item_price: np.ndarray,
        item_rating: np.ndarray,
        item_category: np.ndarray,
        user_purchase_history: np.ndarray,
        user_avg_purchase_value: np.ndarray,
        user_days_since_last_purchase: np.ndarray,
        hour: np.ndarray,
        is_weekend: np.ndarray,
        device_type: np.ndarray
    ) -> np.ndarray:
        """
        Compute conversion probability based on features.

        Key: CVR is MUCH lower than CTR (typically 1-5% vs 10-20%)
        """
        # Base probability - start very low (CVR << CTR)
        prob = np.ones_like(user_age, dtype=np.float32) * 0.005

        # Purchase history is the strongest signal for conversion
        prob += 0.03 * np.log1p(user_purchase_history)

        # Income level effect (higher income -> higher conversion)
        prob += 0.008 * user_income_level

        # Recent purchases increase conversion likelihood
        prob += 0.01 * np.exp(-user_days_since_last_purchase / 10)

        # Item rating effect (high quality items convert better)
        prob += 0.01 * (item_rating - 3.0) / 2.0

        # Price effect (moderate prices convert better)
        # Too cheap = low quality perception, too expensive = hesitation
        optimal_price = 50
        prob += 0.005 * np.exp(-(item_price - optimal_price) ** 2 / 1000)

        # Age effect (middle-aged users convert more)
        prob += 0.005 * np.exp(-(user_age - 40) ** 2 / 300)

        # Category effect
        prob += 0.003 * np.sin(item_category / 3)

        # Weekend effect (more time to convert)
        prob += 0.005 * is_weekend

        # Peak shopping hours (evening)
        prob += 0.003 * np.maximum(0, np.sin((hour - 20) * np.pi / 12))

        # Desktop users convert slightly better (easier checkout)
        prob += 0.005 * (device_type == 1)

        # Clip to realistic CVR range (0.5% - 8%)
        return np.clip(prob, 0.005, 0.08)

    def get_feature_info(self) -> Dict[str, Dict]:
        """
        Get information about features for model configuration.

        Returns:
            Dictionary containing feature specifications
        """
        ctr_features = {
            'sparse_features': ['user_id', 'item_id', 'item_category', 'user_gender',
                               'hour', 'day_of_week', 'device_type'],
            'dense_features': ['user_age', 'item_price', 'user_click_history',
                              'user_show_history', 'user_historical_ctr'],
            'task': 'binary_classification',
            'label_type': 'click (0/1)',
            'typical_rate': '10-20%'
        }

        cvr_features = {
            'sparse_features': ['user_id', 'item_id', 'item_category', 'user_gender',
                               'user_income_level', 'hour', 'day_of_week', 'device_type'],
            'dense_features': ['user_age', 'item_price', 'item_rating', 'is_weekend',
                              'user_purchase_history', 'user_avg_purchase_value',
                              'user_days_since_last_purchase', 'user_click_history',
                              'user_show_history'],
            'task': 'binary_classification',
            'label_type': 'conversion (0/1)',
            'typical_rate': '1-5%',
            'notes': 'CVR << CTR, has conversion delay issues'
        }

        return {
            'ctr': ctr_features,
            'cvr': cvr_features
        }


if __name__ == '__main__':
    # Example usage
    generator = AdDataGenerator(seed=42)

    print("=== CTR Data Generation ===")
    train_X, train_y, test_X, test_y = generator.generate_ctr_data(n_samples=1000)
    print(f"Train samples: {len(train_y)}, Click rate: {train_y.mean():.1%}")
    print(f"Test samples: {len(test_y)}, Click rate: {test_y.mean():.1%}")
    print(f"Features: {list(train_X.keys())}")

    print("\n=== CVR Data Generation ===")
    train_X, train_y, test_X, test_y = generator.generate_cvr_data(n_samples=1000)
    print(f"Train samples: {len(train_y)}, Conversion rate: {train_y.mean():.1%}")
    print(f"Test samples: {len(test_y)}, Conversion rate: {test_y.mean():.1%}")
    print(f"Features: {list(train_X.keys())}")
    print(f"Note: CVR ({train_y.mean():.1%}) << CTR (typically 10-20%)")

    print("\n=== Feature Info ===")
    feature_info = generator.get_feature_info()
    print("CTR sparse features:", feature_info['ctr']['sparse_features'])
    print("CTR dense features:", feature_info['ctr']['dense_features'])
    print(f"CTR task: {feature_info['ctr']['task']}, typical rate: {feature_info['ctr']['typical_rate']}")
    print("\nCVR sparse features:", feature_info['cvr']['sparse_features'])
    print("CVR dense features:", feature_info['cvr']['dense_features'])
    print(f"CVR task: {feature_info['cvr']['task']}, typical rate: {feature_info['cvr']['typical_rate']}")
    print(f"CVR notes: {feature_info['cvr']['notes']}")
