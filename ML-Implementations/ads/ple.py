"""
Progressive Layered Extraction (PLE) for multi-task learning.

The core problem with MMoE:
    All K experts are shared across every task. When tasks conflict, the shared
    experts are simultaneously pulled toward CTR patterns and CVR patterns.
    The gate for task A may suppress experts that task B needs — a subtler form
    of negative transfer that hard parameter sharing makes obvious but MMoE still
    suffers from.

PLE's solution — Customized Gate Control (CGC):
    Separate each extraction layer into task-specific experts + shared experts.

        Task-k gate: softmax over [task-k experts + shared experts]
        h_k = Σ_j gate_k[j] · candidate_j

    Task-k experts only serve task k; no other task can pull them. Shared experts
    are insulated from task-specific experts of other tasks.

    Stacking L such CGC layers gives PLE: each layer progressively refines both
    the task-specific and the shared representations.

        Layer 1 input:  raw features (same for all)
        Layer n input:  task-k output from layer n-1  (for task-k experts)
                        shared output from layer n-1   (for shared experts)
                        ← shared gate at layer n-1 selects over ALL experts
                          (task-specific + shared) to form this shared state

    L = 1 ≡ CGC (no multi-layer extraction). Increasing L adds depth while
    keeping task isolation per layer.

Key equations (one CGC layer for task k):
    task-k candidates  = [e_{k,1}(h_k), ..., e_{k,Kk}(h_k),
                          e_{s,1}(h_s), ..., e_{s,Ks}(h_s)]
    h_k_new = Σ_j softmax(W_k · h_k)[j] · task-k candidates[j]

    shared candidates  = task-1 experts || task-2 experts || ... || shared experts
    h_s_new = Σ_j softmax(W_s · h_s)[j] · shared candidates[j]

Reference: Tang et al., 2020 — "Progressive Layered Extraction (PLE):
           A Novel Multi-Task Learning (MTL) Model for Personalization"
           https://dl.acm.org/doi/10.1145/3383313.3412236
"""

import numpy as np
import tensorflow as tf


class GatedMixture(tf.keras.layers.Layer):
    """Softmax-gated mixture of expert outputs.

    Inputs:
        [gate_logits (B, K), expert_1 (B, D), ..., expert_K (B, D)]
    Output:
        (B, D) — Σ_k softmax(gate_logits)[k] · expert_k
    """

    def call(self, inputs):
        gate_logits = inputs[0]       # (B, K)
        expert_outs = inputs[1:]      # K tensors each (B, D)

        gate = tf.nn.softmax(gate_logits, axis=-1)           # (B, K)
        experts_stacked = tf.stack(expert_outs, axis=1)      # (B, K, D)
        gate_exp = tf.expand_dims(gate, axis=-1)             # (B, K, 1)
        return tf.reduce_sum(experts_stacked * gate_exp, axis=1)  # (B, D)


class PLE:
    """Progressive Layered Extraction for multi-task CTR/CVR prediction.

    Architecture (2 tasks, 2 extraction layers):

        Input x
          │
          ├─ Task-1 experts (x) ─┐
          ├─ Task-2 experts (x)  ├──  Extraction Layer 1
          └─ Shared  experts (x) ┘
               │
               ├─ gate_1(x): mix [task-1 + shared]          → h1¹
               ├─ gate_2(x): mix [task-2 + shared]          → h2¹
               └─ gate_s(x): mix [task-1 + task-2 + shared] → hs¹
                    │
          ├─ Task-1 experts (h1¹) ─┐
          ├─ Task-2 experts (h2¹)   ├── Extraction Layer 2 (last)
          └─ Shared  experts (hs¹) ─┘
               │
               ├─ gate_1(h1¹): mix [task-1 + shared] → h1²
               └─ gate_2(h2¹): mix [task-2 + shared] → h2²
                    │
                    ├─ Tower_1(h1²) → y_1
                    └─ Tower_2(h2²) → y_2

    The shared gate at the LAST extraction layer is omitted (its output has
    no next-layer consumers).

    Args:
        input_dim:             Dimension of the shared input feature vector.
        num_task_experts:      K_k — task-specific expert count per task.
        num_shared_experts:    K_s — shared expert count.
        expert_units:          Hidden layer sizes for each expert DNN.
        num_extraction_layers: L — stacked CGC layers. L=1 ≡ plain CGC.
        task_names:            Task names, e.g. ["ctr", "cvr"].
        tower_units:           Hidden layer sizes for each task tower.
        dropout_rate:          Dropout after each hidden layer (0 = off).
        l2_reg:                L2 regularisation on all Dense kernels.
        use_batch_norm:        BatchNorm before each ReLU.
        learning_rate:         Adam learning rate.
    """

    def __init__(
        self,
        input_dim: int,
        num_task_experts: int = 3,
        num_shared_experts: int = 3,
        expert_units: list = [128, 64],
        num_extraction_layers: int = 2,
        task_names: list = ["ctr", "cvr"],
        tower_units: list = [64, 32],
        dropout_rate: float = 0.0,
        l2_reg: float = 0.0,
        use_batch_norm: bool = False,
        learning_rate: float = 0.001,
    ):
        self.input_dim = input_dim
        self.num_task_experts = num_task_experts
        self.num_shared_experts = num_shared_experts
        self.expert_units = expert_units
        self.num_extraction_layers = num_extraction_layers
        self.task_names = task_names
        self.tower_units = tower_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self._compiled = False

        self.model = self._build_model()

    # ------------------------------------------------------------------
    # Architecture helpers
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

    def _build_expert(self, inp, prefix: str, regularizer):
        """Build one expert DNN; return its output tensor."""
        x = inp
        for i, units in enumerate(self.expert_units):
            x = self._dense_block(x, units, f"{prefix}_{i}", regularizer)
        return x

    def _build_model(self) -> tf.keras.Model:
        regularizer = tf.keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None

        inp = tf.keras.Input(shape=(self.input_dim,), name="input")

        # Running states for each task and for the shared component
        task_states  = {t: inp for t in self.task_names}
        shared_state = inp

        for layer_idx in range(self.num_extraction_layers):
            is_last = (layer_idx == self.num_extraction_layers - 1)

            # ---- Task-specific experts ----
            # Each expert_k_j takes task_states[t] as input
            task_expert_outs = {
                t: [
                    self._build_expert(
                        task_states[t],
                        f"task_{t}_expert_l{layer_idx}_e{j}",
                        regularizer,
                    )
                    for j in range(self.num_task_experts)
                ]
                for t in self.task_names
            }

            # ---- Shared experts ----
            # Each shared expert takes shared_state as input
            shared_expert_outs = [
                self._build_expert(
                    shared_state,
                    f"shared_expert_l{layer_idx}_e{j}",
                    regularizer,
                )
                for j in range(self.num_shared_experts)
            ]

            # ---- Task gating: select over [task-k experts + shared experts] ----
            new_task_states = {}
            for t in self.task_names:
                candidates = task_expert_outs[t] + shared_expert_outs
                n = self.num_task_experts + self.num_shared_experts
                gate_logits = tf.keras.layers.Dense(
                    n, use_bias=False, name=f"gate_{t}_l{layer_idx}"
                )(task_states[t])
                new_task_states[t] = GatedMixture(name=f"mix_{t}_l{layer_idx}")(
                    [gate_logits] + candidates
                )

            # ---- Shared gating: select over ALL experts ----
            # Only needed when there is a next extraction layer to consume it.
            if not is_last:
                all_candidates = []
                for t in self.task_names:
                    all_candidates.extend(task_expert_outs[t])
                all_candidates.extend(shared_expert_outs)
                n_all = len(self.task_names) * self.num_task_experts + self.num_shared_experts
                gate_s_logits = tf.keras.layers.Dense(
                    n_all, use_bias=False, name=f"gate_shared_l{layer_idx}"
                )(shared_state)
                shared_state = GatedMixture(name=f"mix_shared_l{layer_idx}")(
                    [gate_s_logits] + all_candidates
                )

            task_states = new_task_states

        # ---- Task-specific towers → output ----
        outputs = {}
        for t in self.task_names:
            x = task_states[t]
            for i, units in enumerate(self.tower_units):
                x = self._dense_block(x, units, f"tower_{t}_{i}", regularizer)
            outputs[t] = tf.keras.layers.Dense(
                1, activation="sigmoid", name=f"output_{t}"
            )(x)

        return tf.keras.Model(inputs=inp, outputs=outputs)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compile(self, optimizer=None, loss_weights: dict = None):
        """Compile the model.

        Args:
            optimizer:    Keras optimizer. Defaults to Adam.
            loss_weights: Per-task loss weights. Defaults to equal weighting.
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
        """Train PLE.

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
            Dict mapping task names to (N, 1) probability arrays.
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
