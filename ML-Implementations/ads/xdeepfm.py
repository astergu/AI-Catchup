"""
xDeepFM: eXtreme Deep Factorization Machine.

Motivation — gaps in prior work:
    FM:    Explicitly models 2nd-order interactions at the bit level.
           Limited to pairs (degree 2); no higher-order interactions.
    DNN:   Implicitly models arbitrary-order interactions, but at the
           bit level — individual neurons mix dimensions from different
           fields with no explicit field structure.
    DCN:   Cross network learns bounded-degree polynomial interactions,
           but also at the bit level (a single vector scales x_0).
           No explicit field-vector structure is preserved.

xDeepFM's solution — Compressed Interaction Network (CIN):
    Learns EXPLICIT, bounded-degree, VECTOR-wise interactions.
    "Vector-wise" means each interaction preserves the embedding
    dimension D, so a field's entire embedding participates as a unit
    (unlike bit-level methods that collapse it to a scalar early).

    CIN layer k computes:
        Z^k_{i,j,:} = X^{k-1}_{i,:} ⊙ X^0_{j,:}      (element-wise ⊗, vector-wise)
        X^k_{h,:}   = Σ_{i,j} W^k_{h,i,j} · Z^k_{i,j,:}

    Equivalently: reshape Z^k to (B, H_{k-1}·n, D), transpose to
    (B, D, H_{k-1}·n), apply Conv1D(H_k filters, kernel=1) → (B, D, H_k),
    transpose → (B, H_k, D).

    After T CIN layers, sum-pool each X^k over D → (B, H_k),
    concatenate all layers → (B, H_1+…+H_T), then Dense(1).

    The CIN explicitly represents degree-(k+1) interactions at layer k
    (X^k depends on X^{k-1} which depends on X^{k-2}, …, X^0).
    With T layers: highest interaction degree = T+1.

xDeepFM combines:
    Linear   (1st-order)      ──────────────────────────┐
    CIN      (explicit, high-order, vector-level)  ──────┤ → σ → ŷ
    DNN      (implicit, high-order, bit-level)     ──────┘
    (Shared embedding table across CIN and DNN — same idea as DeepFM.)

Dense features:
    Continuous feature j → e_j = v_j · x_j  (DenseFieldEmbedding),
    so it participates in CIN and DNN alongside sparse fields.

Reference: Lian et al., 2018 — "xDeepFM: Combining Explicit and Implicit Feature
           Interactions for Recommender Systems"
           https://arxiv.org/abs/1803.05170
"""

import numpy as np
import tensorflow as tf


# ------------------------------------------------------------------
# Dense field embedding (identical to DeepFM / AutoInt)
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
# Compressed Interaction Network (CIN)
# ------------------------------------------------------------------

class CIN(tf.keras.layers.Layer):
    """Compressed Interaction Network — vector-wise explicit interactions.

    Input:  x0 (B, n, D)  — stacked field embeddings (same tensor used at every layer)
    Output: (B, Σ H_k)    — concatenated sum-pooled feature maps from all T layers

    At layer k:
        1. Interaction:  Z[i,j] = X^{k-1}[i] ⊙ X^0[j]  →  (B, H_{k-1}·n, D)
        2. Compress:     X^k = Conv1D(H_k, kernel=1)(Z^T) ^T    →  (B, H_k, D)
        3. Sum-pool:     p^k = Σ_D X^k                           →  (B, H_k)

    The Conv1D over the H_{k-1}·n dimension with H_k filters is equivalent
    to the learnable weight matrix W^k ∈ R^{H_k × H_{k-1}·n} in the paper.

    Args:
        layer_sizes:  List of feature-map counts [H_1, H_2, …, H_T].
        l2_reg:       L2 regularisation on the Conv1D kernels (0 = off).
    """

    def __init__(self, layer_sizes: list, l2_reg: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.layer_sizes = list(layer_sizes)
        self.l2_reg      = l2_reg

    def build(self, input_shape):
        self._D = int(input_shape[-1])   # embedding dim (static)
        regularizer = (
            tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None
        )

        # One Conv1D per CIN layer.
        # Conv1D(H_k, kernel_size=1) on input (B, D, H_prev*n) → (B, D, H_k)
        # Assigned as a list attribute so Keras tracks the sublayers.
        self._convs = [
            tf.keras.layers.Conv1D(
                filters=H_k, kernel_size=1, use_bias=False,
                kernel_regularizer=regularizer,
                name=f"cin_conv_{k}",
            )
            for k, H_k in enumerate(self.layer_sizes)
        ]
        super().build(input_shape)

    def call(self, x0):
        # x0: (B, n, D)
        # Use static shape integers where possible so Keras can infer Conv1D's
        # kernel shape during symbolic tracing of the functional model.
        D = self._D           # static Python int — embedding dim
        n = x0.shape[1]       # static Python int — total number of fields
        xk = x0               # running feature maps; starts as X^0 with H_0 = n

        pooled = []
        for conv in self._convs:
            # xk.shape[1] is a static Python int at every CIN step:
            #   step 0 → n (from x0), step k → layer_sizes[k-1] (from Conv1D filters)
            H_prev = xk.shape[1]

            # --- Step 1: outer product (field-vector interaction) ---
            # interaction[b, i, j, d] = xk[b, i, d] * x0[b, j, d]
            interaction = xk[:, :, None, :] * x0[:, None, :, :]  # (B, H_prev, n, D)
            # Flatten field-pair dim with static integers → (B, H_prev*n, D)
            interaction = tf.reshape(interaction, [-1, H_prev * n, D])

            # --- Step 2: compress via Conv1D (kernel=1 over the H_prev*n dim) ---
            # Conv1D expects (batch, steps, channels): transpose so D is "steps"
            # and H_prev*n is "channels" (known statically → kernel can be built).
            interaction = tf.transpose(interaction, [0, 2, 1])  # (B, D, H_prev*n)
            compressed  = conv(interaction)                      # (B, D, H_k)
            xk = tf.transpose(compressed, [0, 2, 1])            # (B, H_k, D)

            # --- Step 3: sum-pool over D → (B, H_k) ---
            pooled.append(tf.reduce_sum(xk, axis=-1))

        return tf.concat(pooled, axis=-1)  # (B, H_1 + H_2 + ... + H_T)

    def get_config(self):
        return {**super().get_config(), "layer_sizes": self.layer_sizes,
                "l2_reg": self.l2_reg}


# ------------------------------------------------------------------
# xDeepFM wrapper
# ------------------------------------------------------------------

class XDeepFM:
    """xDeepFM for CTR / binary classification.

    Architecture:
        Sparse fields [f₁, …, fₙ]  +  Dense features x_dense
          │
          ├─ Linear (1st-order) ─────────────────────────────────┐
          │    Σᵢ embed_1D(fᵢ)  +  w_dense · x_dense            │
          │                                                       │
          ├─ CIN (explicit vector-wise high-order) ──────────────┤ → σ → ŷ
          │    T layers of compressed interaction + sum-pool      │
          │                                                       │
          └─ DNN (implicit bit-level high-order) ────────────────┘
               Flatten[e₁‖…‖eₙ] → DNN → scalar

    The shared k-D embedding table is used by all three components.
    Separate 1-D embeddings are used only for the linear term.

    Args:
        field_dims:      Vocabulary sizes for each categorical field,
                         e.g. [n_users, n_items, n_categories].
                         Field i values must be in [0, field_dims[i]).
        embed_dim:       k — embedding dimension per field.
                         Shared by CIN and DNN; separate 1-D tables for linear.
        dense_feat_dim:  Number of continuous features (0 = none).
        cin_layer_sizes: Feature-map counts for each CIN layer, e.g. [128, 128].
                         More layers → higher interaction degree.
                         Wider layers → more interaction patterns per degree.
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
        cin_layer_sizes: list = [64, 64],
        dnn_units: list = [256, 128, 64],
        dropout_rate: float = 0.0,
        l2_reg: float = 0.0,
        use_batch_norm: bool = False,
        learning_rate: float = 0.001,
    ):
        self.field_dims      = list(field_dims)
        self.embed_dim       = embed_dim
        self.dense_feat_dim  = dense_feat_dim
        self.cin_layer_sizes = list(cin_layer_sizes)
        self.dnn_units       = list(dnn_units)
        self.dropout_rate    = dropout_rate
        self.l2_reg          = l2_reg
        self.use_batch_norm  = use_batch_norm
        self.learning_rate   = learning_rate
        self._compiled       = False

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

        # ----- Shared k-D embedding tables (used by CIN and DNN) -----
        # +1 vocab size so 1-indexed IDs are valid; index 0 is unused padding.
        # Embedding regularization helps prevent per-entity overfitting when
        # the number of entities is large relative to training samples.
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

        # ----- Dense field embeddings (join CIN and DNN) -----
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

        # ----- CIN (explicit vector-wise high-order interactions) -----
        # CIN output: (B, H_1 + H_2 + ... + H_T) — sum-pooled feature maps
        cin_out   = CIN(self.cin_layer_sizes, l2_reg=self.l2_reg, name="cin")(emb_stacked)
        cin_logit = tf.keras.layers.Dense(1, use_bias=False, name="cin_logit")(cin_out)

        # ----- DNN (implicit bit-level high-order interactions) -----
        deep_in = tf.keras.layers.Flatten(name="dnn_flatten")(emb_stacked)
        x = deep_in
        for i, units in enumerate(self.dnn_units):
            x = self._dense_block(x, units, f"dnn_{i}", regularizer)
        dnn_logit = tf.keras.layers.Dense(1, use_bias=False, name="dnn_logit")(x)

        # ----- Combine: σ(linear + CIN + DNN) -----
        logit  = tf.keras.layers.Add(name="logit")([linear_out, cin_logit, dnn_logit])
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
        """Train xDeepFM.

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
                "CIN": CIN,
                "DenseFieldEmbedding": DenseFieldEmbedding,
            },
        )
        self._compiled = True
