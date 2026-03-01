"""
AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks.

Motivation — gaps in prior work:
    FM/DeepFM: FM term explicitly models 2nd-order interactions only; DNN models
               high-order interactions implicitly (no transparency into which
               field pairs drive the prediction).
    DCN:       Cross network learns bounded-degree polynomial interactions but
               all fields are treated with a single vector (v1) or matrix (v2)
               weight — no dynamic, sample-adaptive field weighting.

AutoInt's solution:
    Apply multi-head self-attention to the stacked field embeddings.
    Each interacting layer lets every field attend over every other field,
    learning WHICH field-pairs matter most — in a sample-adaptive way.
    Stacking L layers lets the network combine these pairwise signals into
    progressively higher-order interactions.

    Interacting layer (multi-head self-attention + residual):
        For each head h (per-head dim d_h):
            Q_h = e W_Q^h,   K_h = e W_K^h,   V_h = e W_V^h    ∈ R^{n × d_h}
            α_h = softmax(Q_h K_h^T / √d_h)                     ∈ R^{n × n}
            ẽ_h = α_h V_h                                        ∈ R^{n × d_h}

        Concat H heads:
            ẽ = [ẽ_1 ‖ … ‖ ẽ_H]                                ∈ R^{n × H·d_h}

        Residual + ReLU:
            ẽ' = ReLU(ẽ + e W_res)    (W_res: d→H·d_h; identity if H·d_h = d)

    After L interacting layers: Flatten(ẽ') → Dense(1) → σ

AutoInt+ variant:
    Adds a parallel DNN branch on the raw (layer-0) embeddings:
        output = σ(attn_logit + DNN_logit)
    The DNN captures implicit patterns that attention layers may miss.

Dense features:
    Continuous feature j → e_j = v_j · x_j  (DenseFieldEmbedding),
    where v_j ∈ R^k is learned and x_j is the scalar value.
    Dense field embeddings join the same self-attention pool as sparse fields.

Reference: Song et al., 2019 — "AutoInt: Automatic Feature Interaction Learning
           via Self-Attentive Neural Networks"
           https://arxiv.org/abs/1810.11921
"""

import numpy as np
import tensorflow as tf


# ------------------------------------------------------------------
# Dense field embedding (reused from DeepFM — same concept)
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
# Multi-head self-attention interacting layer
# ------------------------------------------------------------------

class InteractingLayer(tf.keras.layers.Layer):
    """One AutoInt interacting layer: multi-head self-attention + residual.

    Input:  (B, n_fields, d)          — stacked field embeddings
    Output: (B, n_fields, H · d_h)   — attended field representations

    For each head h:
        Q_h = e W_Q^h,  K_h = e W_K^h,  V_h = e W_V^h      W·∈ R^{d × d_h}
        α_h = softmax(Q_h K_h^T / √d_h)                      ∈ R^{n × n}
        ẽ_h = α_h V_h                                         ∈ R^{n × d_h}

    Concat heads → (B, n, H·d_h)
    Residual + ReLU → ẽ' = ReLU(concat + e W_res)
        W_res ∈ R^{d × H·d_h}; omitted (identity) if H·d_h == d.
    """

    def __init__(self, num_heads: int, att_dim: int,
                 use_residual: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.num_heads   = num_heads
        self.att_dim     = att_dim      # per-head output dimension
        self.use_residual = use_residual
        self._scale      = att_dim ** -0.5
        self._res_proj   = False        # set in build()

    def build(self, input_shape):
        d       = int(input_shape[-1])
        out_dim = self.num_heads * self.att_dim

        # Per-head Q / K / V projection matrices: (H, d, d_h)
        self.W_Q = self.add_weight(
            shape=(self.num_heads, d, self.att_dim),
            initializer="glorot_uniform", name="W_Q",
        )
        self.W_K = self.add_weight(
            shape=(self.num_heads, d, self.att_dim),
            initializer="glorot_uniform", name="W_K",
        )
        self.W_V = self.add_weight(
            shape=(self.num_heads, d, self.att_dim),
            initializer="glorot_uniform", name="W_V",
        )

        if self.use_residual and d != out_dim:
            self._res_proj = True
            self.W_res = self.add_weight(
                shape=(d, out_dim),
                initializer="glorot_uniform", name="W_res",
            )
        super().build(input_shape)

    def call(self, x):
        # x: (B, n, d)
        # Multi-head projections via einsum
        Q = tf.einsum("bnd,hdk->bhnk", x, self.W_Q)  # (B, H, n, d_h)
        K = tf.einsum("bnd,hdk->bhnk", x, self.W_K)
        V = tf.einsum("bnd,hdk->bhnk", x, self.W_V)

        # Scaled dot-product attention: (B, H, n, n)
        scores = tf.einsum("bhnk,bhmk->bhnm", Q, K) * self._scale
        attn   = tf.nn.softmax(scores, axis=-1)

        # Context: (B, H, n, d_h) → transpose → (B, n, H, d_h) → reshape → (B, n, H·d_h)
        ctx = tf.einsum("bhnm,bhmk->bhnk", attn, V)  # (B, H, n, d_h)
        ctx = tf.transpose(ctx, [0, 2, 1, 3])         # (B, n, H, d_h)
        out_dim = self.num_heads * self.att_dim
        ctx = tf.reshape(ctx, (tf.shape(x)[0], tf.shape(x)[1], out_dim))

        if self.use_residual:
            res = tf.matmul(x, self.W_res) if self._res_proj else x
            ctx = tf.nn.relu(ctx + res)

        return ctx  # (B, n, H·d_h)

    def get_config(self):
        return {
            **super().get_config(),
            "num_heads":    self.num_heads,
            "att_dim":      self.att_dim,
            "use_residual": self.use_residual,
        }


# ------------------------------------------------------------------
# AutoInt wrapper
# ------------------------------------------------------------------

class AutoInt:
    """AutoInt for CTR / binary classification.

    Architecture:
        Sparse fields [f₁, …, fₙ]  +  Dense features x_dense
          │
          ├─ Embedding: (B, n_total_fields, k)
          │
          ├─ Interacting Layer 1  (multi-head self-attention + residual)
          ├─ Interacting Layer 2
          │   ⋮
          └─ Interacting Layer L  →  (B, n_total_fields, H·d_h)
               │
               Flatten → Dense(1)                ← attention path
               [+ Flatten(raw_emb) → DNN → Dense(1)]  ← optional DNN (AutoInt+)
               │
               σ → ŷ

    Args:
        field_dims:      Vocabulary sizes for each categorical field,
                         e.g. [n_users, n_items, n_categories].
                         Field i values must be in [0, field_dims[i]).
        embed_dim:       k — embedding dimension. Shared across all fields.
        dense_feat_dim:  Number of continuous features (0 = none).
        num_heads:       H — attention heads per interacting layer.
        att_dim:         d_h — per-head output dimension.
                         Total attention output per field = H · d_h.
                         Set att_dim = embed_dim // num_heads to keep
                         all interacting layers at the same width with
                         no residual projection needed.
        num_layers:      L — stacked interacting layers. L=1 is a single
                         round of self-attention; L≥2 captures higher-order.
        use_residual:    Add residual (+ ReLU) in each interacting layer.
        dnn_units:       If non-empty, adds a parallel DNN on the raw
                         embeddings and sums its logit with the attention
                         logit (AutoInt+ variant).
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
        num_heads: int = 2,
        att_dim: int = 8,          # default: embed_dim // num_heads → no W_res needed
        num_layers: int = 3,
        use_residual: bool = True,
        dnn_units: list = [],
        dropout_rate: float = 0.0,
        l2_reg: float = 0.0,
        use_batch_norm: bool = False,
        learning_rate: float = 0.001,
    ):
        self.field_dims      = list(field_dims)
        self.embed_dim       = embed_dim
        self.dense_feat_dim  = dense_feat_dim
        self.num_heads       = num_heads
        self.att_dim         = att_dim
        self.num_layers      = num_layers
        self.use_residual    = use_residual
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

        # ----- Sparse embeddings: (B, embed_dim) each -----
        # +1 vocab size so 1-indexed IDs are valid; index 0 is unused padding.
        emb_tables = [
            tf.keras.layers.Embedding(
                self.field_dims[i] + 1, self.embed_dim, name=f"emb_{i}"
            )
            for i in range(n_sparse)
        ]
        sparse_embs = [emb_tables[i](sparse_inputs[i]) for i in range(n_sparse)]

        # Reshape each (B, embed_dim) → (B, 1, embed_dim), then cat → (B, n_sparse, embed_dim)
        sparse_3d = [
            tf.keras.layers.Reshape((1, self.embed_dim), name=f"emb_{i}_3d")(emb)
            for i, emb in enumerate(sparse_embs)
        ]
        emb_stacked = tf.keras.layers.Concatenate(axis=1, name="sparse_stack")(sparse_3d)

        # ----- Dense field embeddings (participate in self-attention) -----
        if self.dense_feat_dim > 0:
            dense_embs = DenseFieldEmbedding(self.embed_dim, name="dense_field_emb")(
                dense_input
            )
            emb_stacked = tf.keras.layers.Concatenate(axis=1, name="emb_stack")(
                [emb_stacked, dense_embs]
            )
        # emb_stacked: (B, n_total_fields, embed_dim)

        # ----- L interacting layers (multi-head self-attention) -----
        x = emb_stacked
        for i in range(self.num_layers):
            x = InteractingLayer(
                num_heads=self.num_heads,
                att_dim=self.att_dim,
                use_residual=self.use_residual,
                name=f"interacting_{i}",
            )(x)
        # x: (B, n_total_fields, H·d_h)  after the last interacting layer

        # ----- Attention path logit -----
        attn_flat  = tf.keras.layers.Flatten(name="attn_flat")(x)
        attn_logit = tf.keras.layers.Dense(1, use_bias=True, name="attn_logit")(attn_flat)

        # ----- Optional parallel DNN (AutoInt+) -----
        if self.dnn_units:
            raw_flat = tf.keras.layers.Flatten(name="raw_flat")(emb_stacked)
            h = raw_flat
            for j, units in enumerate(self.dnn_units):
                h = self._dense_block(h, units, f"dnn_{j}", regularizer)
            dnn_logit = tf.keras.layers.Dense(1, use_bias=False, name="dnn_logit")(h)
            logit = tf.keras.layers.Add(name="logit")([attn_logit, dnn_logit])
        else:
            logit = attn_logit

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
        """Train AutoInt.

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
                "InteractingLayer": InteractingLayer,
                "DenseFieldEmbedding": DenseFieldEmbedding,
            },
        )
        self._compiled = True
