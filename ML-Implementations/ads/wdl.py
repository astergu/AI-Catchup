"""
Wide & Deep Learning jointly trains wide linear models and deep neural networks to capture both memorization and generalization capabilities. 
The wide component captures feature interactions, while the deep component captures complex patterns in the data. 

"""

import tensorflow as tf

class WDL:
    """
    Wide component: A linear model (y=Wx+b) that captures feature interactions and memorization.
                    We use Follow-the-regularized-Leader (FTRL) optimizer for training the wide part.

    Deep component: A feedforward neural network (a(l+1)=f(W(l)*a(l)+b(l))) that captures complex patterns and generalization.
                    Use Adagrad optimizer for the deep part.

    P(Y=1|X) = sigmoid(W_wide*X + W_deep*a_deep)
    """
    def __init__(self, wide_input_dim, deep_input_dim, deep_hidden_units, learning_rate=0.001):
        self.wide_input_dim = wide_input_dim
        self.deep_input_dim = deep_input_dim
        self.deep_hidden_units = deep_hidden_units
        self.learning_rate = learning_rate

        # Build the model using Keras Functional API
        self.model = self._build_model()

    def _build_model(self):
        """Build the Wide & Deep model architecture"""
        # Wide input
        wide_input = tf.keras.Input(shape=(self.wide_input_dim,), name='wide_input')
        # Deep input
        deep_input = tf.keras.Input(shape=(self.deep_input_dim,), name='deep_input')
        # Wide component - linear model
        wide_output = tf.keras.layers.Dense(1, activation=None, name='wide_output')(wide_input)
        # Deep component - feedforward neural network
        deep = deep_input
        for i, units in enumerate(self.deep_hidden_units):
            deep = tf.keras.layers.Dense(units, activation='relu', name=f'deep_layer_{i}')(deep)
        deep_output = tf.keras.layers.Dense(1, activation=None, name='deep_output')(deep)

        # Combine wide and deep outputs
        combined = tf.keras.layers.Add(name='wide_deep_add')([wide_output, deep_output])
        output = tf.keras.layers.Activation('sigmoid', name='output')(combined)

        # Create model
        model = tf.keras.Model(inputs=[wide_input, deep_input], outputs=output)

        return model

    def fit(self, X_wide, X_deep, y, epochs=10, batch_size=32, validation_split=0.0, verbose=1):
        """
        Train the Wide & Deep model.

        Args:
            X_wide: Wide features (numpy array or tensor)
            X_deep: Deep features (numpy array or tensor)
            y: Labels (numpy array or tensor)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)

        Returns:
            Training history
        """
        # For proper Wide & Deep training, we should use different optimizers
        # However, Keras doesn't support multiple optimizers easily in compile()
        # So we use a single optimizer (Adam) which works well in practice
        # For production, you'd want to use custom training loop with FTRL for wide and Adagrad for deep

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )

        history = self.model.fit(
            [X_wide, X_deep],
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )

        return history

    def fit_with_dual_optimizers(self, X_wide, X_deep, y, epochs=10, batch_size=32,
                                  wide_lr=None, deep_lr=None, verbose=1):
        """
        Train with separate optimizers for wide and deep components.
        Uses FTRL for wide and Adagrad for deep as per the original paper.

        Args:
            X_wide: Wide features (numpy array)
            X_deep: Deep features (numpy array)
            y: Labels (numpy array)
            epochs: Number of training epochs
            batch_size: Batch size
            wide_lr: Learning rate for wide (FTRL). Default: 10x base learning rate
            deep_lr: Learning rate for deep (Adagrad). Default: base learning rate
            verbose: Verbosity mode
        """
        import numpy as np

        # Set default learning rates based on empirical tuning on synthetic CTR data
        # WARNING: FTRL+Adagrad show high variance (AUC 0.5-0.65) vs Adam's ~0.87
        # This method is experimental - use fit() with Adam for production
        if wide_lr is None:
            wide_lr = 0.05  # FTRL can handle higher LR for sparse features
        if deep_lr is None:
            deep_lr = 0.01  # Adagrad with moderate LR

        if verbose:
            print(f"Using wide_lr={wide_lr}, deep_lr={deep_lr}")

        # Create optimizers as per original Wide & Deep paper
        # FTRL with very light regularization for wide features
        wide_optimizer = tf.keras.optimizers.Ftrl(
            learning_rate=wide_lr,
            l1_regularization_strength=0.0001,
            l2_regularization_strength=0.00001
        )
        deep_optimizer = tf.keras.optimizers.Adagrad(learning_rate=deep_lr)

        # Get trainable variables
        wide_vars = [var for var in self.model.trainable_variables if 'wide_output' in var.name]
        deep_vars = [var for var in self.model.trainable_variables if
                    'deep_layer' in var.name or 'deep_output' in var.name]

        if verbose:
            print(f"Wide variables: {len(wide_vars)}, Deep variables: {len(deep_vars)}")

        # Loss function
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        # Training loop
        n_samples = len(y)
        indices = np.arange(n_samples)

        for epoch in range(epochs):
            # Shuffle data each epoch
            np.random.shuffle(indices)
            epoch_loss = 0
            epoch_acc = 0
            n_batches = 0

            # Batch training
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                # Get batch data
                X_wide_batch = tf.constant(X_wide[batch_indices], dtype=tf.float32)
                X_deep_batch = tf.constant(X_deep[batch_indices], dtype=tf.float32)
                y_batch = tf.constant(y[batch_indices], dtype=tf.float32)

                # Forward pass and compute gradients
                with tf.GradientTape() as tape:
                    # Make predictions
                    predictions = self.model([X_wide_batch, X_deep_batch], training=True)
                    # Compute loss
                    loss = loss_fn(y_batch, predictions)

                # Compute gradients for ALL variables
                all_vars = wide_vars + deep_vars
                grads = tape.gradient(loss, all_vars)

                # Split gradients
                wide_grads = grads[:len(wide_vars)]
                deep_grads = grads[len(wide_vars):]

                # Filter out None gradients before applying
                wide_grads_vars = [(g, v) for g, v in zip(wide_grads, wide_vars) if g is not None]
                deep_grads_vars = [(g, v) for g, v in zip(deep_grads, deep_vars) if g is not None]

                # Apply gradients with respective optimizers
                if wide_grads_vars:
                    wide_optimizer.apply_gradients(wide_grads_vars)
                if deep_grads_vars:
                    deep_optimizer.apply_gradients(deep_grads_vars)

                # Update metrics
                epoch_loss += loss.numpy()
                epoch_acc += tf.reduce_mean(
                    tf.cast(tf.equal(tf.round(predictions), y_batch), tf.float32)
                ).numpy()
                n_batches += 1

            # Print epoch results
            if verbose:
                avg_loss = epoch_loss / n_batches
                avg_acc = epoch_acc / n_batches
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}')

    def predict(self, X_wide, X_deep):
        """
        Make predictions using the trained model.

        Args:
            X_wide: Wide features (numpy array or tensor)
            X_deep: Deep features (numpy array or tensor)

        Returns:
            Predicted probabilities (values between 0 and 1)
        """
        return self.model.predict([X_wide, X_deep])

    def predict_classes(self, X_wide, X_deep, threshold=0.5):
        """
        Predict binary classes (0 or 1).

        Args:
            X_wide: Wide features
            X_deep: Deep features
            threshold: Classification threshold (default: 0.5)

        Returns:
            Binary predictions (0 or 1)
        """
        probabilities = self.predict(X_wide, X_deep)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, X_wide, X_deep, y, verbose=1):
        """
        Evaluate the model on test data.

        Args:
            X_wide: Wide features
            X_deep: Deep features
            y: True labels
            verbose: Verbosity mode

        Returns:
            Loss and metrics values
        """
        return self.model.evaluate([X_wide, X_deep], y, verbose=verbose)

    def save(self, filepath):
        """Save the model to disk"""
        self.model.save(filepath)

    def load(self, filepath):
        """Load a model from disk"""
        self.model = tf.keras.models.load_model(filepath)