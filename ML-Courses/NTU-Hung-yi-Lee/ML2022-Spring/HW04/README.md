# Speaker Identification

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

` Initial Baseline: based on code provided by the course.`

| Model | Modifications | Private MSE | Public MSE |
| ------------ | --------------- | --------------- | ------------- |