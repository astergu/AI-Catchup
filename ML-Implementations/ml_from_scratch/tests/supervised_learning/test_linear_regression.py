import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from ml_from_scratch.supervised_learning import LinearRegression

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
    plt.ylabel('Mean Squared Error')