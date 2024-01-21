# Homework 4: Speaker Identification

**Multiclass Classification**

Predict speaker class from given speech of dataset `VoxCeleb2`.

- `Train`: 56666 processed audio features with labels.
- `Test`: 4000 processed audio features without labels.
- `Label`: 600 classes in total, each class represents a speaker.

## Suggested baselines

- `Simple`: 0.60824
- `Medium`: 0.70375
  - Modify the parameters of the transformer modules.
- `Strong`: 0.77750
  - Construct `Conformer`, a variety of Transformer.
- `Boss`: 0.86500
  - Implement `Self-Attention Pooling` & `Additive Margin Softmax`

## Attempts

[https://www.kaggle.com/c/ml2022spring-hw4](https://www.kaggle.com/c/ml2022spring-hw4)

> Somehow submission failed with errors (key finding error), so I will just use the best `Validation Accuracy` as evaluation metric.
>  
> The accuracy listed below can not be compared to the `suggested baselines`.

` Initial Baseline: based on code provided by the course.`

| Model | Modifications | Validation Accuracy |
| ---------- | ---------- | ----------- |
| Initial Baseline |  | 0.6494 |
| Tranformer tuning | d_model=128 | 0.7039  | 
| Transformer tuning | Transformer layers (num_layers=2) | 0.7412 |  
| Transformer tuning | Transformer layers (num_layers=3) | 0.7708 |  
| Transformer tuning | Transformer layer (nhead 2->4) | 0.7936 |  
| Transformer tuning | Transformer layer (dim_feedforward 256->512, num_layers=4) | 0.8137 |  