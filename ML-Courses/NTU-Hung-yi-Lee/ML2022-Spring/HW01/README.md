# COVID-19 Daily Cases Prediction
- Training data: 2699 samples
- Testing data: 1078 samples
- Evaluation metric: **Mean Squared Error (MSE)**

## Attempts

` Initial Baseline: based on code provided by the course.`

| Model | Modifications | Private MSE | Public MSE |
| ------------ | --------------- | --------------- | ------------- |
| Initial baseline  |  | 1.81579 | 1.75023 | 
| Feature engineering  | 1. Choose 4 features [53, 69, 85, 101] | 1.19474 | 1.09591 | 
| Feature engineering  | 1. Use feature selection `mutual_info_regression` to select top 20 features | 0.91823 | 0.87508 |
| Model Structure | 1. Linear layer size 16 -> 32 | 0.90913 | 0.85993 | 