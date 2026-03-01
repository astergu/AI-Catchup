"""
Deep & Cross Network (DCN) for CTR prediction.

The key idea is the Cross Network: a stack of cross layers that efficiently learns
bounded-degree polynomial feature interactions. Each cross layer is:

    v1 (Wang et al. 2017):
        x_{l+1} = x_0 · (x_l^T w_l) + b_l + x_l
        w_l ∈ R^d — a weight *vector*; O(d) params per layer.

    v2 (Wang et al. 2021):
        x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l
        W_l ∈ R^{d×d} — a weight *matrix*; O(d²) params per layer.

        Low-rank variant (W_l ≈ U_l V_l^T):
        x_{l+1} = x_0 ⊙ (U_l (V_l^T x_l) + b_l) + x_l
        U_l, V_l ∈ R^{d×r} — reduces params to O(dr).

    In both versions, x_0 is the original input (held fixed across all layers)
    and x_l flows through the cross layers. After L cross layers, the network
    can represent all polynomial interactions up to degree L+1.

The DNN runs in parallel (or sequentially in the stacked variant) to capture
implicit high-order patterns that the cross network misses.

Two structure modes:
    Parallel (default):
        Input ──► Cross Network ─────┐
        Input ──► Deep  Network ─────┴──► Concat ──► Dense(1) ──► Sigmoid

    Stacked:
        Input ──► Cross Network ──► Deep Network ──► Dense(1) ──► Sigmoid

References:
    v1: Wang et al. 2017 — "Deep & Cross Network for Ad Click Predictions"
        https://arxiv.org/abs/1708.05123
    v2: Wang et al. 2021 — "DCN V2: Improved Deep & Cross Network"
        https://arxiv.org/abs/2008.13535
"""

import numpy as np
import tensorflow as tf


class CrossLayer(tf.keras.layers.Layer):
    """A single cross layer for explicit polynomial feature interaction.

    v1: x_{l+1} = x_0 · (x_l^T w_l) + b_l + x_l
        — scalar dot product, then scale x_0. O(d) params.

    v2: x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l
        — full matrix transform, then element-wise gate with x_0. O(d²) params.

    v2 low-rank: W_l ≈ U_l V_l^T, so the transform becomes U_l (V_l^T x_l).
        Reduces params to O(dr) while keeping most of the expressiveness.

    In all variants x_0 (the original layer input) is passed separately and held
    constant; x_l changes at each layer.
    """

    def __init__(self, version: str = "v2", low_rank: int = None, **kwargs):
        """
        Args:
            version:  'v1' or 'v2'.
            low_rank: v2 only. If given, decomposes W ≈ U V^T with rank r.
                      Recommended: r = input_dim // 4.
        """
        super().__init__(**kwargs)
        if version not in ("v1", "v2"):
            raise ValueError(f"version must be 'v1' or 'v2', got {version!r}")
        self.version = version
        self.low_rank = low_rank

    def build(self, input_shape):
        d = int(input_shape[-1])
        if self.version == "v1":
            # Weight vector: dot-product with x_l yields a per-sample scalar
            self.w = self.add_weight(shape=(d, 1), initializer="glorot_uniform", name="w")
        else:
            if self.low_rank:
                self.U = self.add_weight(shape=(d, self.low_rank), initializer="glorot_uniform", name="U")
                self.V = self.add_weight(shape=(d, self.low_rank), initializer="glorot_uniform", name="V")
            else:
                self.W = self.add_weight(shape=(d, d), initializer="glorot_uniform", name="W")
        self.b = self.add_weight(shape=(d,), initializer="zeros", name="b")
        super().build(input_shape)

    def call(self, x0, xl):
        """
        Args:
            x0: Original model input — stays constant across all cross layers. Shape (B, d).
            xl: Output of the previous cross layer. Shape (B, d).
        Returns:
            Cross layer output, shape (B, d).
        """
        if self.version == "v1":
            # x_l^T w_l → scalar per sample (B, 1); then scale x_0 element-wise
            scalar = xl @ self.w          # (B, 1)
            return x0 * scalar + self.b + xl
        else:
            if self.low_rank:
                # W x_l = U (V^T x_l) via two cheap projections
                xl_proj = xl @ self.V                      # (B, r)
                xl_w = xl_proj @ tf.transpose(self.U)      # (B, d)
            else:
                xl_w = xl @ tf.transpose(self.W)           # (B, d)
            return x0 * (xl_w + self.b) + xl

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"version": self.version, "low_rank": self.low_rank})
        return cfg


class DCN:
    """Deep & Cross Network for CTR / binary classification.

    Takes a single stacked feature vector (embeddings + dense features concatenated).

    Args:
        input_dim:        Dimension of the input feature vector.
        num_cross_layers: Number of cross layers L. After L layers the model can
                          represent all polynomial interactions up to degree L+1.
        deep_units:       Hidden layer sizes for the DNN component.
        version:          Cross layer type: 'v1' (vector) or 'v2' (matrix). Default 'v2'.
        structure:        'parallel' (default) or 'stacked'. Parallel concatenates the
                          cross and DNN outputs; stacked feeds cross output into the DNN.
        low_rank:         v2 only. Rank r for W ≈ U V^T. None = full rank.
        dropout_rate:     Dropout after each DNN hidden layer (0 = off).
        l2_reg:           L2 regularization on DNN Dense kernels.
        use_batch_norm:   Apply BatchNorm before each ReLU in the DNN.
        learning_rate:    Adam learning rate used when fit() auto-compiles.
    """

    def __init__(
        self,
        input_dim: int,
        num_cross_layers: int = 3,
        deep_units: list = [128, 64],
        version: str = "v2",
        structure: str = "parallel",
        low_rank: int = None,
        dropout_rate: float = 0.0,
        l2_reg: float = 0.0,
        use_batch_norm: bool = False,
        learning_rate: float = 0.001,
    ):
        self.input_dim = input_dim
        self.num_cross_layers = num_cross_layers
        self.deep_units = deep_units
        self.version = version
        self.structure = structure
        self.low_rank = low_rank
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self._compiled = False

        self.model = self._build_model()

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def _dense_block(self, x, units: int, name: str, regularizer):
        """Dense → [BN] → ReLU → [Dropout]."""
        x = tf.keras.layers.Dense(
            units, activation=None, kernel_regularizer=regularizer, name=f"{name}_fc"
        )(x)
        if self.use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
        x = tf.keras.layers.Activation("relu", name=f"{name}_relu")(x)
        if self.dropout_rate > 0:
            x = tf.keras.layers.Dropout(self.dropout_rate, name=f"{name}_drop")(x)
        return x

    def _build_model(self) -> tf.keras.Model:
        regularizer = tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None

        inp = tf.keras.Input(shape=(self.input_dim,), name="input")

        # Cross network — x_0 = inp, threaded through all L layers
        x_cross = inp
        for i in range(self.num_cross_layers):
            x_cross = CrossLayer(
                version=self.version,
                low_rank=self.low_rank,
                name=f"cross_{i}",
            )(inp, x_cross)

        if self.structure == "stacked":
            # DNN sits on top of the cross network output
            x = x_cross
            for i, units in enumerate(self.deep_units):
                x = self._dense_block(x, units, f"deep_{i}", regularizer)
            combined = x
        else:
            # DNN runs on the raw input in parallel with the cross network
            x = inp
            for i, units in enumerate(self.deep_units):
                x = self._dense_block(x, units, f"deep_{i}", regularizer)
            combined = tf.keras.layers.Concatenate(name="concat")([x_cross, x])

        output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(combined)
        return tf.keras.Model(inputs=inp, outputs=output)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compile(self, optimizer=None):
        """Compile the model. Called automatically by fit() on first call."""
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        self._compiled = True

    def fit(self, X, y, epochs=10, batch_size=256, validation_split=0.0, verbose=1):
        """Train the model.

        Args:
            X:                Feature matrix, shape (N, input_dim).
            y:                Binary labels, shape (N,).
            epochs:           Training epochs.
            batch_size:       Mini-batch size.
            validation_split: Fraction held out for validation.
            verbose:          Keras verbosity (0 / 1 / 2).

        Returns:
            Keras History object.
        """
        if not self._compiled:
            self.compile()
        return self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        """Return predicted click probabilities, shape (N, 1)."""
        return self.model.predict(X, verbose=0)

    def evaluate(self, X, y, verbose=1):
        """Evaluate loss and metrics on labelled data."""
        if not self._compiled:
            self.compile()
        return self.model.evaluate(X, y, verbose=verbose)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        self.model.save(filepath)

    def load(self, filepath: str):
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={"CrossLayer": CrossLayer},
        )
        self._compiled = True
