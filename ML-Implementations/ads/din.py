"""
Deep Interest Network (DIN) for CTR prediction.

The core insight: user interests are diverse and not always fully activated by a
candidate ad. Rather than pooling all behavior embeddings equally (like a simple
average), DIN uses a local activation unit to compute an attention score for each
behavior item conditioned on the target item. The scores are used as weights in a
weighted sum — no softmax, by design.

Why no softmax?
    Softmax forces all weights to sum to 1 and squashes absolute magnitudes. But
    the strength of a user's interest matters: a user who clicked 50 gaming items
    should have a stronger interest signal than one who clicked 2, even if both
    show 100% relevance to a gaming ad. Skipping softmax preserves this.

Key components:
    1. Shared item embedding table (behaviors and target share weights).
    2. Activation Unit: for each behavior item h_i and target e_t, computes
           features = [h_i, e_t, h_i ⊙ e_t, |h_i − e_t|]
           score_i  = MLP(features)   ← raw score, no softmax
    3. Attention pooling: user_interest = Σ_i score_i · h_i  (padded positions zeroed)
    4. Main DNN: [user_interest ‖ target_emb ‖ other_features] → DNN → sigmoid

Dice activation (proposed alongside DIN):
    Dice(x) = p(x) · x + (1 − p(x)) · α · x
    p(x)    = σ(BN(x))   where BN = BatchNorm (center=False, scale=False)
    Unlike PReLU which always switches at 0, Dice's threshold adapts to the batch
    mean/variance. α is a learned per-channel scale, initialized to 0.

Reference: Zhou et al., 2018 — "Deep Interest Network for Click-Through Rate Prediction"
           https://arxiv.org/abs/1706.06978
"""

import numpy as np
import tensorflow as tf


# ------------------------------------------------------------------
# Custom activation: Dice
# ------------------------------------------------------------------

class Dice(tf.keras.layers.Layer):
    """Data-Adaptive Activation Function.

    Dice(x) = p(x) · x + (1 − p(x)) · α · x
    p(x)    = sigmoid(BatchNorm(x))

    The BN step computes the data-adaptive decision boundary: instead of always
    activating at x=0 (PReLU), the threshold shifts with the batch distribution.
    α (per-channel, init 0) is learned and controls the negative-region slope.
    """

    def build(self, input_shape):
        d = int(input_shape[-1])
        self.alpha = self.add_weight(shape=(d,), initializer="zeros", name="alpha")
        # BN without affine transform — we only want the normalised gate p(x)
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False, name="bn")
        super().build(input_shape)

    def call(self, x, training=None):
        p = tf.sigmoid(self.bn(x, training=training))
        return p * x + (1.0 - p) * self.alpha * x

    def get_config(self):
        return super().get_config()


# ------------------------------------------------------------------
# Activation Unit (attention scorer)
# ------------------------------------------------------------------

class ActivationUnit(tf.keras.layers.Layer):
    """Scores each behavior item's relevance to the target item.

    Input features per position:
        [h_i, e_t, h_i ⊙ e_t, |h_i − e_t|]   — shape (B, T, 4·embed_dim)

    These are passed through a small MLP → raw scalar score.
    No softmax: absolute interest strength is preserved.

    Args:
        hidden_units: Sizes of hidden Dense layers in the attention MLP.
        use_dice:     Use Dice activation (True) or ReLU (False).
    """

    def __init__(self, hidden_units=(64, 16), use_dice=True, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = tuple(hidden_units)
        self.use_dice = use_dice

    def build(self, input_shape):
        self.fcs = []
        self.acts = []
        for i, u in enumerate(self.hidden_units):
            self.fcs.append(tf.keras.layers.Dense(u, name=f"au_fc_{i}"))
            if self.use_dice:
                self.acts.append(Dice(name=f"au_dice_{i}"))
            else:
                self.acts.append(tf.keras.layers.Activation("relu", name=f"au_relu_{i}"))
        self.score_fc = tf.keras.layers.Dense(1, name="au_score")
        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Args:
            inputs: [behavior_emb (B,T,D), target_emb (B,D), pad_mask (B,T) bool]
        Returns:
            scores: (B, T) attention scores; 0.0 at padded positions.
        """
        behavior_emb, target_emb, pad_mask = inputs

        T = tf.shape(behavior_emb)[1]
        # Expand target to (B, T, D) for element-wise operations
        e_t = tf.tile(tf.expand_dims(target_emb, axis=1), [1, T, 1])  # (B, T, D)

        # 4-part interaction features — note absolute difference |h_i − e_t|
        x = tf.concat(
            [behavior_emb, e_t, behavior_emb * e_t, tf.abs(behavior_emb - e_t)],
            axis=-1,
        )  # (B, T, 4D)

        for fc, act in zip(self.fcs, self.acts):
            x = fc(x)
            if self.use_dice:
                x = act(x, training=training)
            else:
                x = act(x)

        scores = tf.squeeze(self.score_fc(x), axis=-1)  # (B, T)

        # Zero out padded positions so they contribute nothing to the weighted sum
        scores = scores * (1.0 - tf.cast(pad_mask, tf.float32))
        return scores

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"hidden_units": self.hidden_units, "use_dice": self.use_dice})
        return cfg


# ------------------------------------------------------------------
# DIN Attention Pooling layer
# ------------------------------------------------------------------

class DINAttentionPooling(tf.keras.layers.Layer):
    """Full DIN attention-weighted pooling.

    Computes:
        scores_i       = ActivationUnit(h_i, e_target)   (no softmax)
        user_interest  = Σ_i scores_i · h_i

    Args:
        hidden_units: Attention MLP hidden sizes (passed to ActivationUnit).
        use_dice:     Use Dice activation in the attention MLP.
    """

    def __init__(self, hidden_units=(64, 16), use_dice=True, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = tuple(hidden_units)
        self.use_dice = use_dice
        self.activation_unit = ActivationUnit(hidden_units, use_dice, name="activation_unit")

    def call(self, inputs, training=None):
        """
        Args:
            inputs: [behavior_emb (B,T,D), target_emb (B,D), item_seq_ids (B,T) int]
        Returns:
            user_interest: (B, D) weighted sum of behavior embeddings.
        """
        behavior_emb, target_emb, item_seq_ids = inputs

        pad_mask = tf.equal(item_seq_ids, 0)  # (B, T), True where padded

        scores = self.activation_unit(
            [behavior_emb, target_emb, pad_mask], training=training
        )  # (B, T)

        # Weighted sum: scores (B, T, 1) · behavior_emb (B, T, D) → sum over T → (B, D)
        user_interest = tf.reduce_sum(
            behavior_emb * tf.expand_dims(scores, axis=-1), axis=1
        )
        return user_interest

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"hidden_units": self.hidden_units, "use_dice": self.use_dice})
        return cfg


# ------------------------------------------------------------------
# DIN wrapper class
# ------------------------------------------------------------------

class DIN:
    """Deep Interest Network for CTR prediction.

    Inputs:
        item_seq:       (N, max_seq_len) int32 — padded behavior sequence. 0 = padding.
        target_item:    (N,) int32             — candidate item ID.
        other_features: (N, other_feat_dim) float32 — optional dense features
                        (user profile, context, etc.).
        y:              (N,) float32 — click labels.

    Args:
        n_items:          Total number of distinct items (max item ID).
        max_seq_len:      Padded sequence length.
        embed_dim:        Embedding dimension shared by all item IDs.
        other_feat_dim:   Dimension of additional dense features (0 = none).
        attention_units:  Hidden layer sizes for the attention MLP.
        dnn_units:        Hidden layer sizes for the main DNN.
        use_dice:         Dice activation in the attention MLP (True) or ReLU (False).
        dropout_rate:     Dropout after each main DNN hidden layer (0 = off).
        l2_reg:           L2 regularization on main DNN kernels.
        use_batch_norm:   BatchNorm before each ReLU in the main DNN.
        learning_rate:    Adam learning rate.
    """

    def __init__(
        self,
        n_items: int,
        max_seq_len: int = 50,
        embed_dim: int = 16,
        other_feat_dim: int = 0,
        attention_units: tuple = (64, 16),
        dnn_units: list = [256, 128, 64],
        use_dice: bool = True,
        dropout_rate: float = 0.0,
        l2_reg: float = 0.0,
        use_batch_norm: bool = False,
        learning_rate: float = 0.001,
    ):
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.other_feat_dim = other_feat_dim
        self.attention_units = tuple(attention_units)
        self.dnn_units = dnn_units
        self.use_dice = use_dice
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

        # ----- Inputs -----
        item_seq_inp    = tf.keras.Input(shape=(self.max_seq_len,), dtype=tf.int32, name="item_seq")
        target_item_inp = tf.keras.Input(shape=(),                  dtype=tf.int32, name="target_item")
        model_inputs = [item_seq_inp, target_item_inp]

        if self.other_feat_dim > 0:
            other_feat_inp = tf.keras.Input(
                shape=(self.other_feat_dim,), dtype=tf.float32, name="other_features"
            )
            model_inputs.append(other_feat_inp)

        # ----- Shared item embedding table -----
        # Index 0 is reserved for padding; item IDs start at 1.
        item_emb_layer = tf.keras.layers.Embedding(
            input_dim=self.n_items + 1,   # +1 for padding index 0
            output_dim=self.embed_dim,
            embeddings_initializer="uniform",
            name="item_embedding",
        )

        behavior_emb = item_emb_layer(item_seq_inp)    # (B, T, D)
        target_emb   = item_emb_layer(target_item_inp) # (B, D)

        # ----- DIN attention pooling -----
        user_interest = DINAttentionPooling(
            hidden_units=self.attention_units,
            use_dice=self.use_dice,
            name="din_pooling",
        )([behavior_emb, target_emb, item_seq_inp])    # (B, D)

        # ----- Concatenate all features -----
        parts = [user_interest, target_emb]
        if self.other_feat_dim > 0:
            parts.append(other_feat_inp)
        x = tf.keras.layers.Concatenate(name="concat")(parts)

        # ----- Main DNN -----
        for i, units in enumerate(self.dnn_units):
            x = self._dense_block(x, units, f"dnn_{i}", regularizer)

        output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

        return tf.keras.Model(inputs=model_inputs, outputs=output)

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

    def _make_inputs(self, item_seq, target_item, other_features=None):
        inputs = [item_seq, target_item]
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
        other_features=None,
        epochs: int = 10,
        batch_size: int = 256,
        validation_split: float = 0.0,
        verbose: int = 1,
    ):
        """Train DIN.

        Args:
            item_seq:       (N, max_seq_len) int32 — padded behavior history.
            target_item:    (N,) int32 — candidate ad item IDs.
            y:              (N,) float32 — click labels.
            other_features: (N, other_feat_dim) float32 — optional dense features.
            epochs:         Training epochs.
            batch_size:     Mini-batch size.
            validation_split: Fraction held out for validation.
            verbose:        Keras verbosity.

        Returns:
            Keras History object.
        """
        if not self._compiled:
            self.compile()
        return self.model.fit(
            self._make_inputs(item_seq, target_item, other_features),
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, item_seq, target_item, other_features=None) -> np.ndarray:
        """Return predicted click probabilities, shape (N, 1)."""
        return self.model.predict(
            self._make_inputs(item_seq, target_item, other_features), verbose=0
        )

    def evaluate(self, item_seq, target_item, y, other_features=None, verbose=1):
        """Evaluate loss and metrics on labelled data."""
        if not self._compiled:
            self.compile()
        return self.model.evaluate(
            self._make_inputs(item_seq, target_item, other_features), y, verbose=verbose
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
                "Dice": Dice,
                "ActivationUnit": ActivationUnit,
                "DINAttentionPooling": DINAttentionPooling,
            },
        )
        self._compiled = True
