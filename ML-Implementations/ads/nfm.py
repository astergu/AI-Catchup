"""
NFM: Neural Factorization Machine for Sparse Predictive Analytics.

Motivation — gaps in prior work:
    FM:    Second-order interactions are a single scalar: ½ Σₜ [(Σᵢ eᵢₜ)² − Σᵢ eᵢₜ²].
           No higher-order interactions beyond degree 2.
    DeepFM DNN: Concatenates all field embeddings → (B, n·k) before the DNN.
           High-dimensional input; DNN must implicitly deduplicate the pairwise
           signals that FM already computed analytically.

NFM's solution — Bi-Interaction Pooling:
    Compute the FM second-order formula but stop BEFORE summing over k:

        f_BI = ½ (‖Σᵢ eᵢ‖² − Σᵢ ‖eᵢ‖²)    ∈ R^k   (element-wise operations)

    f_BI[d] = ½ [(Σᵢ eᵢd)² − Σᵢ eᵢd²]
            = Σᵢ<ⱼ eᵢd · eⱼd

    So the d-th element of f_BI is the sum of all pairwise interactions in
    embedding dimension d. The k-dimensional vector encodes ALL n(n-1)/2
    pairwise interaction signals in a compact form.

    This compressed representation (k dimensions, not n·k) is then fed into
    a DNN that can learn higher-order combinations of these signals.

Architecture:
    Sparse fields [f₁, …, fₙ]  +  Dense features x_dense
      │
      ├─ Linear (1st-order) ─────────────────────────────────────┐
      │    Σᵢ embed_1D(fᵢ)  +  w_dense · x_dense               │
      │                                                           │
      └─ Bi-Interaction Pooling (2nd-order compressed) ──────────┤ → σ → ŷ
           f_BI ∈ R^k → [BN] → [Dropout] → DNN → Dense(1)      │
                                                                  ┘

    Key difference from DeepFM DNN:
        DeepFM: Flatten(all embeddings) → (B, n·k) → DNN
        NFM:    Bi-Interaction(all embeddings) → (B, k) → DNN

    The NFM DNN input is n× more compact and already captures 2nd-order structure,
    so a shallower/narrower DNN can achieve similar (or better) results.

Dense features:
    Continuous feature j → e_j = v_j · x_j  (DenseFieldEmbedding),
    where v_j ∈ R^k is learned. Dense field embeddings join the
    Bi-Interaction pooling alongside sparse embeddings.

Reference: He & Chua, 2017 — "Neural Factorization Machines for Sparse
           Predictive Analytics"
           https://arxiv.org/abs/1708.05027
"""

import numpy as np
import tensorflow as tf


# ------------------------------------------------------------------
# Dense field embedding (identical to DeepFM / AutoInt / xDeepFM)
# ------------------------------------------------------------------

class DenseFieldEmbedding(tf.keras.layers.Layer):
    """Maps each continuous feature to a k-D field embedding.

    For continuous field j with scalar value xⱼ:
        eⱼ = Vⱼ · xⱼ    where Vⱼ ∈ Rᵏ is learned.

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
        return x[:, :, None] * self.V[None, :, :]  # (B, n_dense, k)

    def get_config(self):
        return {**super().get_config(), "embed_dim": self.embed_dim}


# ------------------------------------------------------------------
# Bi-Interaction Pooling
# ------------------------------------------------------------------

class BiInteractionPooling(tf.keras.layers.Layer):
    """Bi-Interaction Pooling: collapses field embeddings into a k-D summary.

    Given stacked field embeddings e ∈ R^{n_fields × k}:

        f_BI[d] = ½ [(Σᵢ eᵢd)² − Σᵢ eᵢd²]  =  Σᵢ<ⱼ eᵢd · eⱼd

    This is the FM second-order formula applied element-wise and retaining
    the full k-dimensional result (no sum over d).

    Compare with FMLayer (DeepFM):
        FMLayer  → scalar:  ½ Σ_d [(Σᵢ eᵢd)² − Σᵢ eᵢd²]
        BiIP     → vector:  ½    [(Σᵢ eᵢ)²   − Σᵢ eᵢ²  ]  ∈ R^k

    The k-D output is the input to the NFM DNN, letting higher layers model
    complex combinations of the pairwise interaction signals.

    Input:  (B, n_fields, k)
    Output: (B, k)
    """

    def call(self, x):
        # x: (B, n_fields, k)
        square_of_sum = tf.square(tf.reduce_sum(x, axis=1))   # (B, k)
        sum_of_squares = tf.reduce_sum(tf.square(x), axis=1)  # (B, k)
        return 0.5 * (square_of_sum - sum_of_squares)         # (B, k)


# ------------------------------------------------------------------
# NFM wrapper
# ------------------------------------------------------------------

class NFM:
    """NFM for CTR / binary classification.

    Architecture:
        Sparse fields [f₁, …, fₙ]  +  Dense features x_dense
          │
          ├─ Linear (1st-order) ─────────────────────────────────┐
          │    Σᵢ embed_1D(fᵢ)  +  w_dense · x_dense            │
          │                                                       │
          └─ Bi-Interaction Pooling ──────────────────────────────┤ → σ → ŷ
               f_BI = ½(‖Σeᵢ‖² − Σ‖eᵢ‖²)  ∈ R^k              │
               → [BN] → [Dropout] → DNN → Dense(1)             │
                                                                  ┘

    The Bi-Interaction Pooling compresses all pairwise embedding interactions
    into a k-D vector (vs. n·k in DeepFM's DNN), making the DNN input far
    more compact and already interaction-aware.

    Args:
        field_dims:      Vocabulary sizes for each categorical field,
                         e.g. [n_users, n_items, n_categories].
                         Field i values must be in [0, field_dims[i]).
        embed_dim:       k — embedding dimension per field.
        dense_feat_dim:  Number of continuous features (0 = none).
        dnn_units:       Hidden layer sizes for the DNN after Bi-Interaction.
                         The DNN input is always k-dimensional, so these can
                         be smaller than in DeepFM (which uses n·k input).
        dropout_rate:    Dropout after Bi-Interaction output and each DNN
                         hidden layer (0 = off). The NFM paper applies dropout
                         to the Bi-Interaction layer to prevent co-adaptation.
        l2_reg:          L2 regularisation on embeddings and DNN Dense kernels.
        use_batch_norm:  BatchNorm after Bi-Interaction and before each DNN ReLU.
                         Strongly recommended; the paper shows BN is critical for
                         stable training of NFM.
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
        self.dnn_units      = list(dnn_units)
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

        # ----- Shared k-D embedding tables (used by Bi-Interaction and DNN) -----
        # +1 vocab size so 1-indexed IDs are valid; index 0 is unused padding.
        emb_tables = [
            tf.keras.layers.Embedding(
                self.field_dims[i] + 1, self.embed_dim,
                embeddings_regularizer=regularizer,
                name=f"emb_{i}",
            )
            for i in range(n_sparse)
        ]

        # ----- Separate 1-D embedding tables for linear (1st-order) term -----
        linear_tables = [
            tf.keras.layers.Embedding(
                self.field_dims[i] + 1, 1,
                embeddings_regularizer=regularizer,
                name=f"linear_emb_{i}",
            )
            for i in range(n_sparse)
        ]

        # ----- k-D sparse embeddings: list of (B, k) tensors -----
        sparse_embs = [emb_tables[i](sparse_inputs[i]) for i in range(n_sparse)]

        # Reshape each (B, k) → (B, 1, k), then cat → (B, n_sparse, k)
        sparse_3d = [
            tf.keras.layers.Reshape((1, self.embed_dim), name=f"emb_{i}_3d")(emb)
            for i, emb in enumerate(sparse_embs)
        ]
        emb_stacked = tf.keras.layers.Concatenate(axis=1, name="sparse_stack")(sparse_3d)

        # ----- Dense field embeddings (join Bi-Interaction pool) -----
        if self.dense_feat_dim > 0:
            dense_embs = DenseFieldEmbedding(self.embed_dim, name="dense_field_emb")(
                dense_input
            )
            emb_stacked = tf.keras.layers.Concatenate(axis=1, name="emb_stack")(
                [emb_stacked, dense_embs]
            )
        # emb_stacked: (B, n_total_fields, k)

        # ----- Linear (1st-order) term -----
        linear_terms = [linear_tables[i](sparse_inputs[i]) for i in range(n_sparse)]
        linear_out   = tf.keras.layers.Add(name="linear_sum")(linear_terms)  # (B, 1)

        if self.dense_feat_dim > 0:
            dense_linear = tf.keras.layers.Dense(
                1, use_bias=False, name="dense_linear"
            )(dense_input)
            linear_out = tf.keras.layers.Add(name="linear_total")([linear_out, dense_linear])

        # ----- Bi-Interaction Pooling → (B, k) -----
        bi_out = BiInteractionPooling(name="bi_interaction")(emb_stacked)

        # Optional BN + dropout on the Bi-Interaction output (before DNN).
        # The paper shows BN here is especially important for training stability.
        if self.use_batch_norm:
            bi_out = tf.keras.layers.BatchNormalization(name="bi_bn")(bi_out)
        if self.dropout_rate > 0:
            bi_out = tf.keras.layers.Dropout(self.dropout_rate, name="bi_drop")(bi_out)

        # ----- DNN on Bi-Interaction output (B, k) -----
        x = bi_out
        for i, units in enumerate(self.dnn_units):
            x = self._dense_block(x, units, f"dnn_{i}", regularizer)
        dnn_logit = tf.keras.layers.Dense(1, use_bias=False, name="dnn_logit")(x)

        # ----- Combine: σ(linear + DNN(f_BI)) -----
        logit  = tf.keras.layers.Add(name="logit")([linear_out, dnn_logit])
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
        """Train NFM.

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
                "BiInteractionPooling": BiInteractionPooling,
                "DenseFieldEmbedding": DenseFieldEmbedding,
            },
        )
        self._compiled = True
