import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        # Initialize weights and bias
        m, n = X.shape                      # number of samples, number of features
        self.weights = np.zeros(n)          # (n,)
        self.bias = 0.0                     # scalar

        # Gradient descent
        for _ in range(self.num_iterations):
            # Compute predictions
            y_pred = X @ self.weights + self.bias
            errors = y_pred - y
            
            # Compute gradients, MSE loss: 1/2m * (y_pred - y)^2
            dw = (1 / m) * (X.T @ errors)
            db = (1 / m) * np.sum(errors)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db    

    def predict(self, X):
        return X @ self.weights + self.bias