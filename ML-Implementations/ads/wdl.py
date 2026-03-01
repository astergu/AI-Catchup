"""
Wide & Deep Learning jointly trains wide linear models and deep neural networks
to capture both memorization and generalization.

Wide component: A linear model (y = Wx + b) that memorizes specific feature
               co-occurrences. Trained with FTRL in the original paper.

Deep component: A feedforward DNN (a^(l+1) = f(W^(l) a^(l) + b^(l))) that
               generalizes to unseen feature combinations. Trained with Adagrad.

Joint prediction:
    P(Y=1|X) = σ( W_wide·X  +  W_deep·a_deep  +  b )

Reference: Cheng et al., 2016 — "Wide & Deep Learning for Recommender Systems"
"""

import numpy as np
import tensorflow as tf


class WDL:
    """Wide & Deep model for CTR prediction / binary classification.

    Args:
        wide_input_dim:    Dimension of wide (sparse/cross) features.
        deep_input_dim:    Dimension of deep (dense/embedding) features.
        deep_hidden_units: List of hidden layer sizes for the DNN, e.g. [128, 64, 32].
        dropout_rate:      Dropout probability after each hidden layer (0 = disabled).
        l2_reg:            L2 regularization coefficient on DNN kernels (0 = disabled).
        use_batch_norm:    If True, add BatchNormalization before each ReLU in the DNN.
        learning_rate:     Default learning rate used when fit() compiles the model.
    """

    def __init__(
        self,
        wide_input_dim: int,
        deep_input_dim: int,
        deep_hidden_units: list,
        dropout_rate: float = 0.0,
        l2_reg: float = 0.0,
        use_batch_norm: bool = False,
        learning_rate: float = 0.001,
    ):
        self.wide_input_dim = wide_input_dim
        self.deep_input_dim = deep_input_dim
        self.deep_hidden_units = deep_hidden_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self._compiled = False

        self.model = self._build_model()

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self) -> tf.keras.Model:
        """Build the Wide & Deep architecture using Keras Functional API."""
        regularizer = tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None

        wide_input = tf.keras.Input(shape=(self.wide_input_dim,), name="wide_input")
        deep_input = tf.keras.Input(shape=(self.deep_input_dim,), name="deep_input")

        # Wide component — single linear projection
        wide_output = tf.keras.layers.Dense(1, use_bias=False, name="wide_output")(wide_input)

        # Deep component — DNN with optional BatchNorm + Dropout
        x = deep_input
        for i, units in enumerate(self.deep_hidden_units):
            x = tf.keras.layers.Dense(
                units,
                activation=None,
                kernel_regularizer=regularizer,
                name=f"deep_{i}",
            )(x)
            if self.use_batch_norm:
                x = tf.keras.layers.BatchNormalization(name=f"bn_{i}")(x)
            x = tf.keras.layers.Activation("relu", name=f"relu_{i}")(x)
            if self.dropout_rate > 0:
                x = tf.keras.layers.Dropout(self.dropout_rate, name=f"dropout_{i}")(x)
        deep_output = tf.keras.layers.Dense(1, name="deep_output")(x)

        # Joint logit = wide logit + deep logit + shared bias
        logit = tf.keras.layers.Add(name="joint_logit")([wide_output, deep_output])
        output = tf.keras.layers.Activation("sigmoid", name="output")(logit)

        return tf.keras.Model(inputs=[wide_input, deep_input], outputs=output)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compile(self, optimizer=None):
        """Compile the model with binary cross-entropy loss.

        Calling this explicitly lets you pass a custom optimizer. If omitted,
        fit() calls it automatically with Adam.
        """
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        self._compiled = True

    def fit(self, X_wide, X_deep, y, epochs=10, batch_size=32, validation_split=0.0, verbose=1):
        """Train the model with Adam optimizer (single optimizer, recommended).

        Args:
            X_wide:           Wide feature matrix.
            X_deep:           Deep feature matrix.
            y:                Binary labels.
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
            [X_wide, X_deep],
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
        )

    def fit_with_dual_optimizers(
        self,
        X_wide,
        X_deep,
        y,
        epochs=10,
        batch_size=32,
        wide_lr=0.05,
        deep_lr=0.01,
        verbose=1,
    ) -> dict:
        """Train with separate optimizers as described in the original paper.

        Uses FTRL (with light L1/L2 regularization) for the wide component and
        Adagrad for the deep component. This mirrors the production setup at Google
        where wide features are sparse and benefit from FTRL's per-coordinate
        learning rates.

        NOTE: On dense / synthetic data Adam typically outperforms this setup.
              Dual optimizers shine when wide features are genuinely sparse
              (e.g., high-cardinality one-hot or cross-product features).

        Args:
            X_wide:    Wide feature matrix (numpy array).
            X_deep:    Deep feature matrix (numpy array).
            y:         Binary labels (numpy array).
            epochs:    Training epochs.
            batch_size: Mini-batch size.
            wide_lr:   FTRL learning rate for wide variables.
            deep_lr:   Adagrad learning rate for deep variables.
            verbose:   0 = silent, 1 = per-epoch summary.

        Returns:
            Dict with lists: {"loss": [...], "accuracy": [...]}.
        """
        wide_optimizer = tf.keras.optimizers.Ftrl(
            learning_rate=wide_lr,
            l1_regularization_strength=1e-4,
            l2_regularization_strength=1e-5,
        )
        deep_optimizer = tf.keras.optimizers.Adagrad(learning_rate=deep_lr)

        wide_vars = [v for v in self.model.trainable_variables if "wide_output" in v.name]
        deep_vars = [v for v in self.model.trainable_variables
                     if "deep_" in v.name or "bn_" in v.name or "deep_output" in v.name]

        loss_fn = tf.keras.losses.BinaryCrossentropy()
        n_samples = len(y)
        indices = np.arange(n_samples)
        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            np.random.shuffle(indices)
            epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0

            for start in range(0, n_samples, batch_size):
                idx = indices[start : start + batch_size]
                xw = tf.constant(X_wide[idx], dtype=tf.float32)
                xd = tf.constant(X_deep[idx], dtype=tf.float32)
                yb = tf.constant(y[idx], dtype=tf.float32)

                with tf.GradientTape() as tape:
                    preds = self.model([xw, xd], training=True)
                    loss = loss_fn(yb, preds)

                all_vars = wide_vars + deep_vars
                grads = tape.gradient(loss, all_vars)

                wide_gv = [(g, v) for g, v in zip(grads[:len(wide_vars)], wide_vars) if g is not None]
                deep_gv = [(g, v) for g, v in zip(grads[len(wide_vars):], deep_vars) if g is not None]

                if wide_gv:
                    wide_optimizer.apply_gradients(wide_gv)
                if deep_gv:
                    deep_optimizer.apply_gradients(deep_gv)

                epoch_loss += loss.numpy()
                epoch_acc += tf.reduce_mean(
                    tf.cast(tf.equal(tf.round(preds), yb), tf.float32)
                ).numpy()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            avg_acc = epoch_acc / n_batches
            history["loss"].append(avg_loss)
            history["accuracy"].append(avg_acc)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}, acc: {avg_acc:.4f}")

        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X_wide, X_deep) -> np.ndarray:
        """Return predicted click probabilities."""
        return self.model.predict([X_wide, X_deep], verbose=0)

    def evaluate(self, X_wide, X_deep, y, verbose=1):
        """Evaluate loss and metrics on labelled data."""
        if not self._compiled:
            self.compile()
        return self.model.evaluate([X_wide, X_deep], y, verbose=verbose)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        """Save the Keras model to disk."""
        self.model.save(filepath)

    def load(self, filepath: str):
        """Load a saved Keras model from disk."""
        self.model = tf.keras.models.load_model(filepath)
        self._compiled = True
