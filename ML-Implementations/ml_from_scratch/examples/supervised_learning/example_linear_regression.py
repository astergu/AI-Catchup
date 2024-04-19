from sklearn.datasets import make_regression
from supervised_learning import LinearRegression

def example_linear_regression():
    X, y = make_regression(n_samples=100, n_features=1, noise=20)
