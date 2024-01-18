# Image Classification

## Data

The images are collected from the food-11 dataset classified into 11 classes.

- **Train**: 9866 labeled images;
- **Validation**: 3430 labeled images;
- **Test**: 3347 images 

## Evaluation Metric

Accuracy

## Suggested baselines

- `Simple`: 0.50099
- `Medium`: 0.73207 
  - Training Augmenetation
  - Train longer
- `Strong`: 0.81872
  - Training Augmentation
  - Model Design
  - Train longer
  - Cross Validation
  - Ensemble
- `Boss`: 0.88446
  - Training Augmentation
  - Model Design
  - Test Time Augmentation
  - Train Longer
  - Cross Validation
  - Ensemble

## Attempts


` Initial Baseline: based on code provided by the course.`

| Model | Modifications | Private MSE | Public MSE |
| ------------ | --------------- | --------------- | ------------- |
| Initial Baseline | | 0.47332 | 0.48306 |
| Train Augmentation |  transforms.RandomRotation(20) | 0.52283 | 0.50199 | 
| Train longer | epochs 3->30 | 0.69227 | 0.71613 |
| Add dropout | Dropout layer (p=0.2) before Linear | 0.70337 | 0.71812 |
| Cross Validation | 5-fold | 0.73751 | 0.75298 |
| Ensemble | | | |