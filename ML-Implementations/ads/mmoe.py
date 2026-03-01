"""
Multi-gate Mixture-of-Experts (MMoE) for multi-task learning.

The core problem with naive multi-task learning (hard parameter sharing):
    All tasks share the same bottom network. If tasks conflict (e.g., CTR optimises
    for clicks while CVR optimises for purchases), the shared layers are pulled in
    opposite directions → negative transfer, worse performance than single-task models.

MMoE's solution:
    Replace the single shared bottom with K independent expert networks.
    Each task gets its own gating network that learns a soft selection over experts:

        gate_t(x) = softmax(W_t · x)        # (K,) weights, task-specific
        h_t       = Σ_k gate_t(x)[k] · f_k(x)  # expert mixture for task t
        y_t       = Tower_t(h_t)             # task-specific DNN + sigmoid

    Intuition:
    - Experts specialise: some become general-purpose, others learn task-specific patterns.
    - Gate_CTR and Gate_CVR may select different expert combinations:
        gate_CTR → weights expert knowledge about user click patterns
        gate_CVR → weights expert knowledge about purchase intent
    - Tasks that are correlated share experts naturally; conflicting tasks diverge.

    This differs from ESMM (which enforces p_ctcvr = p_ctr × p_cvr) — MMoE applies
    no probabilistic coupling; each task is supervised independently.

Reference: Ma et al., 2018 — "Modeling Task Relationships in Multi-task Learning
           with Multi-gate Mixture-of-Experts"
           https://dl.acm.org/doi/10.1145/3219819.3220007
"""

import numpy as np
import tensorflow as tf


class GatedMixture(tf.keras.layers.Layer):
    """Softmax-gated mixture of expert outputs.

    Computes:
        gate   = softmax(gate_logits)          # (B, K) — task-specific expert weights
        output = Σ_k gate[k] · expert_k(x)    # (B, D) — weighted sum of expert outputs

    Inputs:
        A list [gate_logits, expert_1, ..., expert_K]
        where gate_logits is (B, K) and each expert is (B, D).

    Output:
        (B, D) — the gated mixture.
    """

    def call(self, inputs):
        gate_logits = inputs[0]       # (B, K)
        expert_outs = inputs[1:]      # K tensors each (B, D)

        gate = tf.nn.softmax(gate_logits, axis=-1)           # (B, K)
        experts_stacked = tf.stack(expert_outs, axis=1)      # (B, K, D)
        gate_exp = tf.expand_dims(gate, axis=-1)             # (B, K, 1)
        return tf.reduce_sum(experts_stacked * gate_exp, axis=1)  # (B, D)


class MMoE:
    """Multi-gate Mixture-of-Experts for multi-task CTR/CVR prediction.

    Architecture:
                              ┌─ Expert_1(x) ─┐
                              ├─ Expert_2(x) ─┤
        Input x ──────────── ├─    ...        ├ ──────────────────────────────┐
                              └─ Expert_K(x) ─┘                               │
                                     │                                         │
               ┌─ gate_1 = softmax(W_1·x) ─► GatedMixture_1 ─► Tower_1 ─► y_1│
               └─ gate_2 = softmax(W_2·x) ─► GatedMixture_2 ─► Tower_2 ─► y_2│
                                                                               │
                           (experts_stacked shared by all gates) ◄─────────────┘

    Args:
        input_dim:      Dimension of the shared input feature vector.
        num_experts:    K — number of expert networks. More experts allow finer
                        specialisation but increase parameters. Typical: 4–8.
        expert_units:   Hidden layer sizes for each expert DNN.
        task_names:     Names of tasks. Determines number of tasks T and names
                        the model outputs, e.g. ["ctr", "cvr"].
        tower_units:    Hidden layer sizes for each task tower (shared architecture,
                        independent weights).
        dropout_rate:   Dropout probability after each hidden layer (0 = off).
        l2_reg:         L2 regularisation on all Dense kernels.
        use_batch_norm: BatchNorm before each ReLU in experts and towers.
        learning_rate:  Adam learning rate used when fit() auto-compiles.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 8,
        expert_units: list = [256, 128],
        task_names: list = ["ctr", "cvr"],
        tower_units: list = [64, 32],
        dropout_rate: float = 0.0,
        l2_reg: float = 0.0,
        use_batch_norm: bool = False,
        learning_rate: float = 0.001,
    ):
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.expert_units = expert_units
        self.task_names = task_names
        self.tower_units = tower_units
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

        # ------ K Expert networks (shared across all tasks) ------
        expert_outputs = []
        for k in range(self.num_experts):
            x = inp
            for i, units in enumerate(self.expert_units):
                x = self._dense_block(x, units, f"expert_{k}_{i}", regularizer)
            expert_outputs.append(x)

        # ------ Task-specific gating + mixing + tower ------
        outputs = {}
        for task in self.task_names:
            # Gate: linear → softmax over K experts
            # (No bias: gate depends only on the input, not a learned offset.)
            gate_logits = tf.keras.layers.Dense(
                self.num_experts, use_bias=False, name=f"gate_{task}"
            )(inp)  # (B, K)

            # Gated mixture: Σ_k softmax(gate)[k] · expert_k
            h = GatedMixture(name=f"mixture_{task}")([gate_logits] + expert_outputs)

            # Task tower
            x = h
            for i, units in enumerate(self.tower_units):
                x = self._dense_block(x, units, f"tower_{task}_{i}", regularizer)

            outputs[task] = tf.keras.layers.Dense(
                1, activation="sigmoid", name=f"output_{task}"
            )(x)

        return tf.keras.Model(inputs=inp, outputs=outputs)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compile(self, optimizer=None, loss_weights: dict = None):
        """Compile the model.

        Args:
            optimizer:    Keras optimizer. Defaults to Adam.
            loss_weights: Per-task loss weights, e.g. {"ctr": 1.0, "cvr": 1.0}.
                          Defaults to equal weighting for all tasks.
        """
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        if loss_weights is None:
            loss_weights = {t: 1.0 for t in self.task_names}

        self.model.compile(
            optimizer=optimizer,
            loss={t: "binary_crossentropy" for t in self.task_names},
            loss_weights={t: loss_weights.get(t, 1.0) for t in self.task_names},
            metrics={t: tf.keras.metrics.AUC(name=f"{t}_auc") for t in self.task_names},
        )
        self._compiled = True

    def fit(
        self,
        X,
        labels: dict,
        epochs: int = 10,
        batch_size: int = 256,
        validation_split: float = 0.0,
        verbose: int = 1,
    ):
        """Train MMoE.

        Args:
            X:          Feature matrix, shape (N, input_dim).
            labels:     Dict mapping task names to label arrays, e.g.
                        {"ctr": y_ctr, "cvr": y_cvr}. Keys must match task_names.
            epochs:     Training epochs.
            batch_size: Mini-batch size.
            validation_split: Fraction held out for validation.
            verbose:    Keras verbosity (0 / 1 / 2).

        Returns:
            Keras History object.
        """
        if not self._compiled:
            self.compile()

        y = {t: labels[t] for t in self.task_names}

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

    def predict(self, X) -> dict:
        """Return predicted probabilities for all tasks.

        Returns:
            Dict mapping task names to (N, 1) probability arrays,
            e.g. {"ctr": array, "cvr": array}.
        """
        raw = self.model.predict(X, verbose=0)
        return {t: raw[t] for t in self.task_names}

    def evaluate(self, X, labels: dict, verbose: int = 1):
        """Evaluate loss and AUC metrics on labelled data."""
        if not self._compiled:
            self.compile()
        y = {t: labels[t] for t in self.task_names}
        return self.model.evaluate(X, y, verbose=verbose)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        self.model.save(filepath)

    def load(self, filepath: str):
        self.model = tf.keras.models.load_model(
            filepath, custom_objects={"GatedMixture": GatedMixture}
        )
        self._compiled = True
