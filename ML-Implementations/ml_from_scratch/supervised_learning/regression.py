import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

class Regression(object):
    """ 
    Base regression model.

    Parameters:
    ------------------
    n_iterations: float
            The number of training iterations.
    learning_rate: float
            The step length that will be used when updating the weights.
    """
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initialize_weights(self, n_features):
        """ Initialize weights randomly """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y):
        """ Train on training dataset """
        # add bias X_0 = 1
        X = np.insert(X, 0, 1, axis=1)
        self.initialize_weights(n_features=X.shape[1])
        self.training_errors = []

        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            mse = np.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
            self.training_errors.append(mse)
            grad_w = (y_pred - y).dot(X) + self.regularization.grad(self.w)
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred


class LinearRegression(Regression):
    """ Linear Regression model. 
    
    Parameters:
    ---------------------
    n_iterations: float
            The number of training iterations.
    learning_rate: float
            The step length that will be used when updating the weights.
    gradient_descent: boolean
            Whether gradient descent should be used when training.
    """
    def __init__(self, n_iterations=100, learning_rate=0.01, gradient_descent=True):
        self.gradient_descent = gradient_descent
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        if not self.gradient_descent:
            X = np.insert(X, 0, 1, axis=1)
            U, S, V = np.linalg.svd(X.T.dot(X))
            S = np.diag(S)
            x_sq_reg_inv = V.dot(np.linalg.pinv(S)).dot(U.T)
            self.w = x_sq_reg_inv.dot(X.T).dot(y)
        else:
            super(LinearRegression, self).fit(X, y)

def test_linear_regression():
    X, y = make_regression(n_samples=100, n_features=1, noise=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    n_samples, n_features = np.shape(X)

    model = LinearRegression(n_iterations=100)
    model.fit(X_train, y_train)
    
    # Training error plot
    n = len(model.training_errors)
    training, = plt.plot(range(n), model.training_errors, label="Training Error")
    plt.legend(handles=[training])
    plt.title("Error Plot")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Iterations")
    plt.show()

if __name__ == '__main__':
    test_linear_regression()