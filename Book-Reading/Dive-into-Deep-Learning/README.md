# Preliminaries

## Data Preprocessing

### Reading the Dataset

```python
import pandas as pd

data = pd.read_csv(data_file)
```

### Handling Missing Data

To handle missing data, typical methods include `imputation` and `deletion`, where imputation replaces missing values witrh substituted ones, while deletion ignores missing values.

```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2] # assume that columns 0 and 1 are features, column index 2 is the target value
inputs = inputs.fillna(inputs.mean()) # use mean value as the imputation value
```

`NaN` in Categorical or discrete values will be considered as a category.

```python
inputs = pd.get_dummies(inputs, dummy_na=True)
```