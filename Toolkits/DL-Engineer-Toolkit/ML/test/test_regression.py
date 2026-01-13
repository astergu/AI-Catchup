import numpy as np
from src.regression import LinearRegression

models = [
    LinearRegression,
]

# Generate some synthetic data
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1)
y = y.ravel()

for model in models:
    m = model(learning_rate=0.01, num_iterations=1000)
    m.fit(X, y)

    y_pred = m.predict(X)

    # Print the results
    print(f"{model.__name__} MSE:", np.mean((y_pred - y) ** 2))