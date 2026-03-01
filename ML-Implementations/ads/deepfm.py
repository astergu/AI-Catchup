"""
DeepFM: A Factorization-Machine based Neural Network for CTR Prediction.

Motivation — gaps in prior work:
    FM:          Learns first- and second-order interactions efficiently
                 (O(kn) trick), but cannot capture higher-order patterns.
    Wide & Deep: Wide component needs hand-crafted cross features.
                 DNN alone cannot efficiently model low-order interactions.

DeepFM's solution:
    Replace the wide component with a Factorization Machine and share the
    embedding table between FM and DNN. No manual feature engineering needed.

    FM component:
        First-order:   y₁ = Σᵢ wᵢ xᵢ         (bias-like linear term)
        Second-order:  y₂ = ½ (‖Σᵢ eᵢ‖² − Σᵢ ‖eᵢ‖²)
                           where eᵢ = embedding(field i)
        The second-order sum runs over the k embedding dimensions in O(n·k).

    Deep component:
        [e₁ ‖ e₂ ‖ … ‖ eₙ] → DNN → scalar
        Same k-D embeddings shared with FM → co-adaptation, fewer parameters.

    Output:  σ(y₁ + y₂ + DNN_out)

Dense features:
    Continuous feature j gets a field embedding  eⱼ = vⱼ · xⱼ
    where vⱼ ∈ Rᵏ is learned and xⱼ is the scalar value.
    These are handled by DenseFieldEmbedding and participate in both FM and DNN.

Reference: Guo et al., 2017 — "DeepFM: A Factorization-Machine based Neural
           Network for CTR Prediction"
           https://arxiv.org/abs/1703.04247
"""

import numpy as np
import tensorflow as tf


# ------------------------------------------------------------------
# Dense field embedding (for continuous features in FM)
# ------------------------------------------------------------------

class DenseFieldEmbedding(tf.keras.layers.Layer):
    """Maps each dense feature to a k-D field embedding for FM.

    For continuous field j with scalar value xⱼ:
        eⱼ = Vⱼ · xⱼ    where Vⱼ ∈ Rᵏ is learned.

    This is the FM treatment of continuous features: each feature gets
    its own k-D embedding vector, scaled by the feature value.

    Input:  (B, n_dense) float  — dense feature values
    Output: (B, n_dense, k)     — per-field embeddings
    """

    def __init__(self, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        n_dense = int(input_shape[-1])
        self.V = self.add_weight(
            shape=(n_dense, self.embed_dim),
            initializer="glorot_uniform",
            name="V",
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, n_dense)  →  result[b, j, :] = V[j, :] * x[b, j]
        return x[:, :, None] * self.V[None, :, :]  # (B, n_dense, k)

    def get_config(self):
        return {**super().get_config(), "embed_dim": self.embed_dim}


# ------------------------------------------------------------------
# FM second-order interaction layer
# ------------------------------------------------------------------

class FMLayer(tf.keras.layers.Layer):
    """FM second-order feature interaction (O(n·k)).

    Given stacked field embeddings e ∈ R^{n_fields × k}:
        y = ½ Σₜ [(Σᵢ eᵢₜ)² − Σᵢ eᵢₜ²]
          = ½ (‖sum_pool(e)‖² − sum_pool(e²))

    This computes all n(n−1)/2 pairwise interactions without enumerating them.
    """

    def call(self, x):
        # x: (B, n_fields, k)
        square_of_sum = tf.square(tf.reduce_sum(x, axis=1))   # (B, k)
        sum_of_squares = tf.reduce_sum(tf.square(x), axis=1)  # (B, k)
        return 0.5 * tf.reduce_sum(
            square_of_sum - sum_of_squares, axis=-1, keepdims=True
        )  # (B, 1)


# ------------------------------------------------------------------
# DeepFM wrapper
# ------------------------------------------------------------------

class DeepFM:
    """DeepFM for CTR / binary classification.

    Architecture:
        Sparse fields [f₁, f₂, …, fₙ]  +  Dense features x_dense
          │
          ├─ Linear (1st-order) ──────────────────────────────────┐
          │    Σᵢ embed_1D(fᵢ)  +  w_dense · x_dense             │
          │                                                        │
          ├─ FM (2nd-order, shared k-D embeddings) ───────────────┤ → σ → ŷ
          │    ½(‖Σᵢ eᵢ‖² − Σᵢ ‖eᵢ‖²)                          │
          │                                                        │
          └─ DNN (shared k-D embeddings) ────────────────────────┘
               Flatten[e₁‖…‖eₙ] → DNN → scalar

    Args:
        field_dims:      Vocabulary sizes for each categorical field,
                         e.g. [n_users, n_items, n_categories].
                         Field i values must be in [0, field_dims[i]).
        embed_dim:       k — embedding dimension per field. Shared by FM and DNN.
        dense_feat_dim:  Number of continuous features (0 = none).
        dnn_units:       Hidden layer sizes for the deep component.
        dropout_rate:    Dropout after each DNN hidden layer (0 = off).
        l2_reg:          L2 regularisation on DNN Dense kernels.
        use_batch_norm:  BatchNorm before each ReLU in DNN.
        learning_rate:   Adam learning rate.
    """

    def __init__(
        self,
        field_dims: list,
        embed_dim: int = 16,
        dense_feat_dim: int = 0,
        dnn_units: list = [256, 128, 64],
        dropout_rate: float = 0.0,
        l2_reg: float = 0.0,
        use_batch_norm: bool = False,
        learning_rate: float = 0.001,
    ):
        self.field_dims     = list(field_dims)
        self.embed_dim      = embed_dim
        self.dense_feat_dim = dense_feat_dim
        self.dnn_units      = dnn_units
        self.dropout_rate   = dropout_rate
        self.l2_reg         = l2_reg
        self.use_batch_norm = use_batch_norm
        self.learning_rate  = learning_rate
        self._compiled      = False

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
        n_sparse = len(self.field_dims)

        # ----- Inputs -----
        sparse_inputs = [
            tf.keras.Input(shape=(), dtype=tf.int32, name=f"field_{i}")
            for i in range(n_sparse)
        ]
        model_inputs = list(sparse_inputs)

        if self.dense_feat_dim > 0:
            dense_input = tf.keras.Input(
                shape=(self.dense_feat_dim,), dtype=tf.float32, name="dense"
            )
            model_inputs.append(dense_input)

        # ----- Embedding tables shared between FM (2nd-order) and DNN -----
        # +1 to accommodate 1-indexed IDs; index 0 unused / serves as padding
        emb_tables = [
            tf.keras.layers.Embedding(
                self.field_dims[i] + 1, self.embed_dim, name=f"emb_{i}"
            )
            for i in range(n_sparse)
        ]

        # ----- 1-D embedding tables for FM first-order (linear) term -----
        linear_tables = [
            tf.keras.layers.Embedding(self.field_dims[i] + 1, 1, name=f"linear_emb_{i}")
            for i in range(n_sparse)
        ]

        # ----- k-D sparse field embeddings -----
        sparse_embs = [emb_tables[i](sparse_inputs[i]) for i in range(n_sparse)]
        # sparse_embs: list of (B, k) tensors

        # Reshape each (B, k) → (B, 1, k) then concatenate → (B, n_sparse, k)
        sparse_3d = [
            tf.keras.layers.Reshape((1, self.embed_dim), name=f"emb_{i}_3d")(emb)
            for i, emb in enumerate(sparse_embs)
        ]
        sparse_embs_stacked = tf.keras.layers.Concatenate(axis=1, name="sparse_stack")(
            sparse_3d
        )  # (B, n_sparse, k)

        # ----- Stack all field embeddings for FM and Deep -----
        if self.dense_feat_dim > 0:
            # Dense field embeddings: (B, dense_feat_dim, k)
            dense_embs = DenseFieldEmbedding(self.embed_dim, name="dense_field_emb")(
                dense_input
            )
            emb_stacked = tf.keras.layers.Concatenate(axis=1, name="emb_stack")(
                [sparse_embs_stacked, dense_embs]
            )  # (B, n_sparse + dense_feat_dim, k)
        else:
            emb_stacked = sparse_embs_stacked  # (B, n_sparse, k)

        # ----- Linear (first-order) term -----
        # Categorical: 1-D embedding per field → (B, 1) each
        linear_terms = [linear_tables[i](sparse_inputs[i]) for i in range(n_sparse)]
        linear_out = tf.keras.layers.Add(name="linear_sum")(linear_terms)  # (B, 1)

        if self.dense_feat_dim > 0:
            # Dense linear: learned weight vector dotted with x_dense
            dense_linear = tf.keras.layers.Dense(
                1, use_bias=False, name="dense_linear"
            )(dense_input)  # (B, 1)
            linear_out = tf.keras.layers.Add(name="linear_total")(
                [linear_out, dense_linear]
            )

        # ----- FM second-order interaction -----
        fm_out = FMLayer(name="fm")(emb_stacked)  # (B, 1)

        # ----- Deep component -----
        deep_in = tf.keras.layers.Flatten(name="deep_flatten")(emb_stacked)
        # (B, n_total_fields * k)
        x = deep_in
        for i, units in enumerate(self.dnn_units):
            x = self._dense_block(x, units, f"dnn_{i}", regularizer)
        deep_out = tf.keras.layers.Dense(1, use_bias=False, name="dnn_out")(x)  # (B, 1)

        # ----- Combine: σ(linear + FM + Deep) -----
        logit  = tf.keras.layers.Add(name="logit")([linear_out, fm_out, deep_out])
        output = tf.keras.layers.Activation("sigmoid", name="output")(logit)

        return tf.keras.Model(inputs=model_inputs, outputs=output)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compile(self, optimizer=None):
        """Compile the model. Called automatically on first fit()."""
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        self._compiled = True

    def _make_inputs(self, sparse_feats, dense_feats=None):
        """Assemble model inputs from numpy arrays."""
        inputs = list(sparse_feats)
        if self.dense_feat_dim > 0:
            if dense_feats is None:
                raise ValueError("dense_feats required when dense_feat_dim > 0")
            inputs.append(dense_feats)
        return inputs

    def fit(
        self,
        sparse_feats,
        y,
        dense_feats=None,
        epochs: int = 10,
        batch_size: int = 256,
        validation_split: float = 0.0,
        verbose: int = 1,
    ):
        """Train DeepFM.

        Args:
            sparse_feats: List of (N,) int32 arrays, one per categorical field.
                          Values for field i must be in [0, field_dims[i]).
            y:            (N,) float32 binary labels.
            dense_feats:  (N, dense_feat_dim) float32 — optional dense features.
            epochs:       Training epochs.
            batch_size:   Mini-batch size.
            validation_split: Fraction held out for validation.
            verbose:      Keras verbosity (0 / 1 / 2).

        Returns:
            Keras History object.
        """
        if not self._compiled:
            self.compile()
        return self.model.fit(
            self._make_inputs(sparse_feats, dense_feats),
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, sparse_feats, dense_feats=None) -> np.ndarray:
        """Return predicted click probabilities, shape (N, 1)."""
        return self.model.predict(
            self._make_inputs(sparse_feats, dense_feats), verbose=0
        )

    def evaluate(self, sparse_feats, y, dense_feats=None, verbose: int = 1):
        """Evaluate loss and metrics on labelled data."""
        if not self._compiled:
            self.compile()
        return self.model.evaluate(
            self._make_inputs(sparse_feats, dense_feats), y, verbose=verbose
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        self.model.save(filepath)

    def load(self, filepath: str):
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={
                "FMLayer": FMLayer,
                "DenseFieldEmbedding": DenseFieldEmbedding,
            },
        )
        self._compiled = True
