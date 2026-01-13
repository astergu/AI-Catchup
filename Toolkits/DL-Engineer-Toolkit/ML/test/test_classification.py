import numpy as np
from src.classification import LogisticRegression

models = [
    LogisticRegression,
]

X = np.random.rand(100, 1)
y = np.random.randint(0, 2, (100,))

for model in models:
    m = model(learning_rate=0.01, num_iterations=1000)
    m.fit(X, y)

    y_pred = m.predict(X)
    print(f"{model.__name__} Accuracy:", np.mean(y_pred == y))