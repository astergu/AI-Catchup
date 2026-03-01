"""
Entire Space Multi-Task Model (ESMM) for joint CTR + CVR prediction.

Two key problems with naïve CVR modeling:

1. Sample Selection Bias (SSB)
   Traditional CVR models train only on clicked samples, but inference runs on all
   impressions. The training distribution p(x | click) ≠ inference distribution p(x).

2. Data Sparsity
   Conversion labels are rare (clicks are rare; conversions are rarer still),
   making it hard to learn a good CVR model directly.

Core insight — factorize CTCVR using the probability chain rule:
    P(conversion | impression) = P(click | impression) × P(conversion | click, impression)
    p_ctcvr                    = p_ctr               × p_cvr

Training on the entire impression space with two supervised signals:
    Loss = BCE(y_ctr, p_ctr) + BCE(y_ctcvr, p_ctr × p_cvr)

The CVR tower is NEVER directly supervised. Its gradients come entirely through the
CTCVR product, which is trained on all impressions → no SSB, and the shared
embedding helps with data sparsity.

Reference: Ma et al., 2018 — "Entire Space Multi-Task Model: An Effective Approach
           for Estimating Post-Click Conversion Rate"
           https://arxiv.org/abs/1804.07931
"""

import numpy as np
import tensorflow as tf


class ESMM:
    """Entire Space Multi-Task Model for CTR and CVR prediction.

    Architecture:
                         ┌── CTR Tower (DNN) ──► p_ctr ──┐
        Input ──► Shared ┤                                 ├── p_ctcvr = p_ctr × p_cvr
                         └── CVR Tower (DNN) ──► p_cvr ──┘

    Training loss:
        L = BCE(y_ctr, p_ctr) + BCE(y_ctcvr, p_ctcvr)

    Labels required:
        y_ctr   : 1 if the impression was clicked, else 0.
        y_ctcvr : 1 if the impression was clicked AND converted, else 0.
                  By definition y_ctcvr[i] <= y_ctr[i].

    Args:
        input_dim:        Dimension of the shared feature vector.
        shared_units:     Hidden units in the shared bottom layers (can be empty).
        ctr_tower_units:  Hidden units in the CTR tower.
        cvr_tower_units:  Hidden units in the CVR tower.
        dropout_rate:     Dropout probability after each hidden layer (0 = off).
        l2_reg:           L2 weight regularization coefficient (0 = off).
        use_batch_norm:   If True, apply BatchNorm before each ReLU.
        learning_rate:    Adam learning rate used when fit() auto-compiles.
    """

    def __init__(
        self,
        input_dim: int,
        shared_units: list = [],
        ctr_tower_units: list = [128, 64],
        cvr_tower_units: list = [128, 64],
        dropout_rate: float = 0.0,
        l2_reg: float = 0.0,
        use_batch_norm: bool = False,
        learning_rate: float = 0.001,
    ):
        self.input_dim = input_dim
        self.shared_units = shared_units
        self.ctr_tower_units = ctr_tower_units
        self.cvr_tower_units = cvr_tower_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self._compiled = False

        # Build two views of the same computation graph:
        #   self.model       — full inference model (p_ctr, p_cvr, p_ctcvr)
        #   self._train_model — training model (p_ctr, p_ctcvr), same weights
        self.model, self._train_model = self._build_models()

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

    def _build_tower(self, x, units: list, prefix: str, regularizer):
        for i, n in enumerate(units):
            x = self._dense_block(x, n, f"{prefix}_{i}", regularizer)
        return x

    def _build_models(self):
        regularizer = tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None

        inp = tf.keras.Input(shape=(self.input_dim,), name="input")

        # Shared bottom — common representation before task-specific towers
        shared = self._build_tower(inp, self.shared_units, "shared", regularizer)

        # CTR tower
        ctr_h = self._build_tower(shared, self.ctr_tower_units, "ctr", regularizer)
        p_ctr = tf.keras.layers.Dense(1, activation="sigmoid", name="p_ctr")(ctr_h)

        # CVR tower (no direct supervision — trained via CTCVR signal only)
        cvr_h = self._build_tower(shared, self.cvr_tower_units, "cvr", regularizer)
        p_cvr = tf.keras.layers.Dense(1, activation="sigmoid", name="p_cvr")(cvr_h)

        # CTCVR = CTR × CVR — the product is what we supervise for the CVR branch
        p_ctcvr = tf.keras.layers.Multiply(name="p_ctcvr")([p_ctr, p_cvr])

        # Full model: used for inference (exposes p_cvr)
        full_model = tf.keras.Model(
            inputs=inp,
            outputs={"p_ctr": p_ctr, "p_cvr": p_cvr, "p_ctcvr": p_ctcvr},
        )

        # Training model: only the two supervised outputs (p_ctr, p_ctcvr)
        # Shares all weights with full_model
        train_model = tf.keras.Model(
            inputs=inp,
            outputs={"p_ctr": p_ctr, "p_ctcvr": p_ctcvr},
        )

        return full_model, train_model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compile(self, optimizer=None):
        """Compile the training model.

        Only p_ctr and p_ctcvr receive direct loss signals.
        The CVR tower is updated implicitly through the CTCVR = CTR × CVR product.
        """
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self._train_model.compile(
            optimizer=optimizer,
            loss={
                "p_ctr": "binary_crossentropy",
                "p_ctcvr": "binary_crossentropy",
            },
            loss_weights={"p_ctr": 1.0, "p_ctcvr": 1.0},
            metrics={
                "p_ctr": tf.keras.metrics.AUC(name="ctr_auc"),
                "p_ctcvr": tf.keras.metrics.AUC(name="ctcvr_auc"),
            },
        )
        self._compiled = True

    def fit(
        self,
        X,
        y_ctr,
        y_ctcvr,
        epochs: int = 10,
        batch_size: int = 256,
        validation_split: float = 0.0,
        verbose: int = 1,
    ):
        """Train ESMM on the entire impression space.

        Args:
            X:                Feature matrix for all impressions, shape (N, input_dim).
            y_ctr:            Click labels, shape (N,). 1 = clicked.
            y_ctcvr:          Click-AND-convert labels, shape (N,). 1 = clicked + converted.
                              Constraint: y_ctcvr[i] == 1 implies y_ctr[i] == 1.
            epochs:           Training epochs.
            batch_size:       Mini-batch size.
            validation_split: Fraction held out for validation.
            verbose:          Keras verbosity (0 / 1 / 2).

        Returns:
            Keras History object.
        """
        if not self._compiled:
            self.compile()
        return self._train_model.fit(
            X,
            {"p_ctr": y_ctr, "p_ctcvr": y_ctcvr},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X) -> dict:
        """Return predicted probabilities for all three quantities.

        Returns:
            Dict with keys "p_ctr", "p_cvr", "p_ctcvr", each shape (N, 1).
            p_cvr is the estimated post-click conversion rate — useful for
            ranking or bid optimization, but never directly supervised during
            training (which eliminates sample selection bias).
        """
        return self.model.predict(X, verbose=0)

    def evaluate(self, X, y_ctr, y_ctcvr, verbose: int = 1):
        """Evaluate training loss and AUC metrics on labelled data."""
        if not self._compiled:
            self.compile()
        return self._train_model.evaluate(
            X,
            {"p_ctr": y_ctr, "p_ctcvr": y_ctcvr},
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        """Save the full inference model."""
        self.model.save(filepath)

    def load(self, filepath: str):
        """Load a saved model. Rebuilds the training view from the same weights."""
        self.model = tf.keras.models.load_model(filepath)
        # Reconstruct training model by reusing the same output tensors
        self._train_model = tf.keras.Model(
            inputs=self.model.input,
            outputs={
                "p_ctr": self.model.get_layer("p_ctr").output,
                "p_ctcvr": self.model.get_layer("p_ctcvr").output,
            },
        )
        self._compiled = False
