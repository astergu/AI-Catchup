"""
Deep Interest Evolution Network (DIEN) for CTR prediction.

DIN limitation:
    DIN applies attention over the full behavior sequence and pools it into a
    single interest vector — it treats user interest as static. But interests
    shift over time: a user who mostly bought sports gear last month and
    recently switched to cooking items has an interest that DIN's static pool
    cannot capture.

DIEN's solution — two-stage RNN:

    Stage 1 — Interest Extraction Layer (GRU):
        h_t = GRU(h_{t-1}, e_t)
        Each h_t is the user's interest state after interacting with item t.

        Auxiliary supervision: for each step t, predict whether the user will
        interact with item_{t+1} (positive) vs a randomly sampled item (neg):
            score(h_t, e) = sigmoid(h_t · e)
            aux_loss = −Σ_t [log score(h_t, e_{t+1}) + log(1 − score(h_t, e_neg))]
        This forces h_t to encode predictive interest, not just sequence context.

    Stage 2 — Interest Evolution Layer (AUGRU):
        Compute relevance of each interest state to the target:
            a_t = Attention(h_t, e_target)   ∈ [0, 1]

        AUGRU modifies the standard GRU update gate with this attention weight:
            u'_t = a_t ⊙ u_t                     (attentional update gate)
            h'_t = (1 − u'_t) ⊙ h'_{t-1} + u'_t ⊙ h̃_t

        Effect: when a_t ≈ 0 (behavior unrelated to target), u'_t ≈ 0 and the
        hidden state barely changes — the evolution "pauses". When a_t ≈ 1, the
        full GRU update fires. h'_T is the user's evolved interest aligned with
        the target ad.

        Padding property: at padded positions, a_t is forced to 0 → u'_t = 0
        → h'_t = h'_{t-1}. So h'_T = h'_{last valid position} for any
        padded suffix, and we can safely use the final time-step output.

    DNN: [h'_T ‖ e_target ‖ other_features] → DNN → sigmoid → CTR

Reference: Feng et al., 2019 — "Deep Interest Evolution Network for
           Click-Through Rate Prediction"
           https://arxiv.org/abs/1809.03672
"""

import numpy as np
import tensorflow as tf


# ------------------------------------------------------------------
# AUGRU Cell
# ------------------------------------------------------------------

class AUGRUCell(tf.keras.layers.Layer):
    """GRU cell with Attentional Update Gate.

    Input at each time step: [e_t | a_t] — behaviour embedding concatenated
    with a scalar attention score.

    Standard GRU:  h_t = (1 − u_t) ⊙ h_{t-1} + u_t ⊙ h̃_t
    AUGRU change:  u'_t = a_t · u_t
                   h'_t = (1 − u'_t) ⊙ h_{t-1} + u'_t ⊙ h̃_t
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        input_dim = int(input_shape[-1]) - 1  # -1 for appended attention scalar
        d = input_dim + self.units             # concat of [e_t; h_{t-1}]

        self.W_r = self.add_weight(shape=(d, self.units), initializer="glorot_uniform", name="W_r")
        self.b_r = self.add_weight(shape=(self.units,),   initializer="zeros",          name="b_r")
        self.W_u = self.add_weight(shape=(d, self.units), initializer="glorot_uniform", name="W_u")
        self.b_u = self.add_weight(shape=(self.units,),   initializer="zeros",          name="b_u")
        # Candidate uses [e_t; r ⊙ h_{t-1}]
        self.W_h = self.add_weight(shape=(d, self.units), initializer="glorot_uniform", name="W_h")
        self.b_h = self.add_weight(shape=(self.units,),   initializer="zeros",          name="b_h")
        super().build(input_shape)

    @property
    def state_size(self):
        return self.units

    @property
    def output_size(self):
        return self.units

    def call(self, inputs, states):
        """
        Args:
            inputs: (B, embed_dim + 1) — [e_t | a_t]
            states: [(B, units)]
        Returns:
            (h'_t, [h'_t])
        """
        e_t = inputs[:, :-1]   # (B, embed_dim)
        a_t = inputs[:, -1:]   # (B, 1) attention score
        h   = states[0]        # (B, units)

        c       = tf.concat([e_t, h], axis=-1)
        r       = tf.sigmoid(c @ self.W_r + self.b_r)
        u       = tf.sigmoid(c @ self.W_u + self.b_u)
        h_tilde = tf.tanh(tf.concat([e_t, r * h], axis=-1) @ self.W_h + self.b_h)

        u_prime = a_t * u                                  # attentional update gate
        h_new   = (1.0 - u_prime) * h + u_prime * h_tilde
        return h_new, [h_new]

    def get_config(self):
        return {**super().get_config(), "units": self.units}


# ------------------------------------------------------------------
# Auxiliary Loss Layer
# ------------------------------------------------------------------

class AuxLossLayer(tf.keras.layers.Layer):
    """Attaches Stage-1 auxiliary next-click supervision as a model loss.

    For each valid time step t (item_seq_next[t] > 0):
        pos_logit = h_t · e_{t+1}          (next item, label = 1)
        neg_logit = h_t · e_neg_t          (random item, label = 0)
        aux_loss -= log σ(pos_logit) + log σ(−neg_logit)

    The loss is added via self.add_loss() and automatically included in the
    training objective. This layer is a no-op for inference: it returns
    h_stage1 unchanged.

    If gru_units ≠ embed_dim, a learned projection aligns dimensions before
    the dot products.
    """

    def __init__(self, weight: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.weight  = weight
        self._proj_W = None  # set in build() if dimensions differ

    def build(self, input_shapes):
        H = int(input_shapes[0][-1])  # gru_units
        D = int(input_shapes[1][-1])  # embed_dim
        if H != D:
            self._proj_W = self.add_weight(
                shape=(H, D), initializer="glorot_uniform", name="proj"
            )
        super().build(input_shapes)

    def call(self, inputs):
        h_stage1, next_emb, neg_emb, next_ids = inputs
        # h_stage1: (B, T, H) | next_emb/neg_emb: (B, T, D) | next_ids: (B, T)

        h = (h_stage1 @ self._proj_W) if self._proj_W is not None else h_stage1

        mask       = tf.cast(next_ids > 0, tf.float32)            # (B, T)
        pos_logits = tf.reduce_sum(h * next_emb, axis=-1)         # (B, T)
        neg_logits = tf.reduce_sum(h * neg_emb,  axis=-1)         # (B, T)

        aux_loss = -tf.reduce_sum(
            mask * (tf.math.log_sigmoid(pos_logits) + tf.math.log_sigmoid(-neg_logits))
        ) / (tf.reduce_sum(mask) + 1e-9)

        self.add_loss(self.weight * aux_loss)
        return h_stage1  # pass through unchanged

    def get_config(self):
        return {**super().get_config(), "weight": self.weight}


# ------------------------------------------------------------------
# DIEN wrapper
# ------------------------------------------------------------------

class DIEN:
    """Deep Interest Evolution Network for CTR prediction.

    Inputs:
        item_seq:       (N, max_seq_len) int32  — padded behaviour sequence. 0 = padding.
        target_item:    (N,) int32              — candidate item ID.
        neg_seq:        (N, max_seq_len) int32  — negative samples for auxiliary loss.
                        Pass zeros (or omit) during inference.
        other_features: (N, other_feat_dim) float32 — optional dense features.
        y:              (N,) float32            — click labels.

    Args:
        n_items:          Vocabulary size (max item ID).
        max_seq_len:      Padded sequence length T.
        embed_dim:        Embedding dimension D shared by all item IDs.
        gru_units:        Hidden size H for Stage-1 GRU and Stage-2 AUGRU.
                          Defaults to embed_dim so H = D (enables Hadamard features
                          in attention and avoids a projection in aux loss).
        other_feat_dim:   Dimension of dense features (0 = none).
        attention_units:  Hidden sizes for the attention MLP in Stage 2.
        dnn_units:        Hidden sizes for the final DNN.
        aux_loss_weight:  Weight λ on the auxiliary loss. Total loss = BCE + λ·aux.
        dropout_rate:     Dropout after each DNN hidden layer (0 = off).
        l2_reg:           L2 regularisation on DNN Dense kernels.
        use_batch_norm:   BatchNorm before each ReLU in DNN.
        learning_rate:    Adam learning rate.
    """

    def __init__(
        self,
        n_items: int,
        max_seq_len: int = 50,
        embed_dim: int = 16,
        gru_units: int = None,
        other_feat_dim: int = 0,
        attention_units: tuple = (64, 16),
        dnn_units: list = [256, 128, 64],
        aux_loss_weight: float = 0.5,
        dropout_rate: float = 0.0,
        l2_reg: float = 0.0,
        use_batch_norm: bool = False,
        learning_rate: float = 0.001,
    ):
        self.n_items         = n_items
        self.max_seq_len     = max_seq_len
        self.embed_dim       = embed_dim
        self.gru_units       = gru_units if gru_units is not None else embed_dim
        self.other_feat_dim  = other_feat_dim
        self.attention_units = tuple(attention_units)
        self.dnn_units       = dnn_units
        self.aux_loss_weight = aux_loss_weight
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
        T = self.max_seq_len

        # ----- Inputs -----
        item_seq_inp    = tf.keras.Input(shape=(T,), dtype=tf.int32,    name="item_seq")
        target_item_inp = tf.keras.Input(shape=(),   dtype=tf.int32,    name="target_item")
        neg_seq_inp     = tf.keras.Input(shape=(T,), dtype=tf.int32,    name="neg_seq")
        model_inputs    = [item_seq_inp, target_item_inp, neg_seq_inp]

        if self.other_feat_dim > 0:
            other_feat_inp = tf.keras.Input(
                shape=(self.other_feat_dim,), dtype=tf.float32, name="other_features"
            )
            model_inputs.append(other_feat_inp)

        # ----- Shared item embedding table -----
        item_emb_layer = tf.keras.layers.Embedding(
            input_dim=self.n_items + 1,
            output_dim=self.embed_dim,
            name="item_embedding",
        )
        behavior_emb = item_emb_layer(item_seq_inp)     # (B, T, D)
        target_emb   = item_emb_layer(target_item_inp)  # (B, D)

        # ----- Stage 1: Interest Extraction GRU -----
        h_stage1 = tf.keras.layers.GRU(
            self.gru_units, return_sequences=True, name="stage1_gru"
        )(behavior_emb)  # (B, T, H)

        # ----- Auxiliary loss -----
        # Positive: shift item_seq by 1 (item_seq_next[t] = item_seq[t+1], 0 at last pos)
        item_seq_next = tf.keras.layers.Lambda(
            lambda x: tf.concat([x[:, 1:], tf.zeros_like(x[:, :1])], axis=1),
            name="item_seq_next",
        )(item_seq_inp)  # (B, T)

        next_emb = item_emb_layer(item_seq_next)  # (B, T, D) — positive items
        neg_emb  = item_emb_layer(neg_seq_inp)    # (B, T, D) — caller-provided negatives

        h_stage1 = AuxLossLayer(weight=self.aux_loss_weight, name="aux_loss")(
            [h_stage1, next_emb, neg_emb, item_seq_next]
        )  # (B, T, H) — adds aux_loss, h_stage1 passes through unchanged

        # ----- Attention scores for AUGRU -----
        e_t = tf.keras.layers.Lambda(
            lambda x: tf.tile(tf.expand_dims(x, axis=1), [1, T, 1]),
            name="expand_target",
        )(target_emb)  # (B, T, D)

        if self.gru_units == self.embed_dim:
            # Include Hadamard product (h_t ⊙ e_target) as an interaction feature
            hadamard    = tf.keras.layers.Multiply(name="attn_hadamard")([h_stage1, e_t])
            attn_input  = tf.keras.layers.Concatenate(axis=-1, name="attn_input")(
                [h_stage1, e_t, hadamard]
            )  # (B, T, 3H)
        else:
            attn_input = tf.keras.layers.Concatenate(axis=-1, name="attn_input")(
                [h_stage1, e_t]
            )  # (B, T, H+D)

        x = attn_input
        for i, units in enumerate(self.attention_units):
            x = tf.keras.layers.Dense(units, activation="relu", name=f"attn_fc_{i}")(x)
        attn_scores = tf.keras.layers.Dense(
            1, activation="sigmoid", name="attn_score"
        )(x)  # (B, T, 1)

        # Zero attention at padded positions so AUGRU state freezes there
        pad_mask = tf.keras.layers.Lambda(
            lambda seq: tf.expand_dims(tf.cast(tf.equal(seq, 0), tf.float32), axis=-1),
            name="pad_mask",
        )(item_seq_inp)  # (B, T, 1): 1 at padding
        attn_scores = tf.keras.layers.Lambda(
            lambda args: args[0] * (1.0 - args[1]),
            name="masked_attn",
        )([attn_scores, pad_mask])  # (B, T, 1)

        # ----- Stage 2: Interest Evolution (AUGRU) -----
        augru_input = tf.keras.layers.Concatenate(axis=-1, name="augru_input")(
            [behavior_emb, attn_scores]
        )  # (B, T, D + 1)

        h_augru = tf.keras.layers.RNN(
            AUGRUCell(self.gru_units, name="augru_cell"),
            return_sequences=True,
            name="augru",
        )(augru_input)  # (B, T, H)

        # At padding positions u'_t = a_t·u_t = 0 → h'_t = h'_{t-1}.
        # Therefore h_augru[:, -1, :] equals the output at the last valid position.
        evolved_interest = tf.keras.layers.Lambda(
            lambda h: h[:, -1, :], name="evolved_interest"
        )(h_augru)  # (B, H)

        # ----- Final DNN -----
        parts = [evolved_interest, target_emb]
        if self.other_feat_dim > 0:
            parts.append(other_feat_inp)
        x = tf.keras.layers.Concatenate(name="concat")(parts)

        for i, units in enumerate(self.dnn_units):
            x = self._dense_block(x, units, f"dnn_{i}", regularizer)

        output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

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

    def _make_inputs(self, item_seq, target_item, neg_seq=None, other_features=None):
        if neg_seq is None:
            neg_seq = np.zeros_like(item_seq)
        inputs = [item_seq, target_item, neg_seq]
        if self.other_feat_dim > 0:
            if other_features is None:
                raise ValueError("other_features is required when other_feat_dim > 0")
            inputs.append(other_features)
        return inputs

    def fit(
        self,
        item_seq,
        target_item,
        y,
        neg_seq=None,
        other_features=None,
        epochs: int = 10,
        batch_size: int = 256,
        validation_split: float = 0.0,
        verbose: int = 1,
    ):
        """Train DIEN.

        Args:
            item_seq:     (N, max_seq_len) int32 — padded behaviour history.
            target_item:  (N,) int32 — candidate item IDs.
            y:            (N,) float32 — click labels.
            neg_seq:      (N, max_seq_len) int32 — negative samples for aux loss.
                          If None, passes zeros (aux loss still runs but on
                          embedding[0] which has no gradient by default).
            other_features: (N, other_feat_dim) float32 — optional dense features.
            epochs:       Training epochs.
            batch_size:   Mini-batch size.
            validation_split: Fraction held out for validation.
            verbose:      Keras verbosity.

        Returns:
            Keras History object.
        """
        if not self._compiled:
            self.compile()
        return self.model.fit(
            self._make_inputs(item_seq, target_item, neg_seq, other_features),
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, item_seq, target_item, neg_seq=None, other_features=None) -> np.ndarray:
        """Return predicted click probabilities, shape (N, 1)."""
        return self.model.predict(
            self._make_inputs(item_seq, target_item, neg_seq, other_features), verbose=0
        )

    def evaluate(self, item_seq, target_item, y, neg_seq=None, other_features=None, verbose=1):
        """Evaluate loss and metrics on labelled data."""
        if not self._compiled:
            self.compile()
        return self.model.evaluate(
            self._make_inputs(item_seq, target_item, neg_seq, other_features),
            y, verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        self.model.save(filepath)

    def load(self, filepath: str):
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={"AUGRUCell": AUGRUCell, "AuxLossLayer": AuxLossLayer},
        )
        self._compiled = True
