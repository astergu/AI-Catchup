# Framewise Phoneme Classification on the LibriSpeech Dataset

## Data

- **Train**: 4268 preprocessed audio features with labels (total 2644158 frames)
- **Test**: 1078 preprocessed audio features (total 646268 frames)
- **Label**: 41 classes, each class represents a phoneme

## What is a phoneme?

A unit of speech sound in a language that can serve to distinguish one word from the other.

## Evaluation Metric

Categorical accuracy

## Suggested baselines

- `Simple`: 0.45797 (sample code)
- `Medium`: 0.69747
  - concat n frames, add layers
- `Strong`: 0.75028
  - concat n frames
  - batchnorm
  - dropout
  - add layers
- `Boss`: 0.82324
  - sequence-labeling (using RNN)

## Attempts

[https://www.kaggle.com/competitions/ml2022spring-hw2](https://www.kaggle.com/competitions/ml2022spring-hw2)

` Initial Baseline: based on code provided by the course.`

| Model | Modifications | Private Accuracy | Public Accuracy |
| ------------ | --------------- | --------------- | ------------- |
| Initial baseline |  | 0.46226 | 0.46066 |
| Frame concat | Concat frames = 29 | 0.66413 | 0.66212 | 
| Make network deeper | hidden layers 1->2 | 0.68649 | 0.68523 |  
| Make network wider | hidden_dim 256->512 | 0.69436 | 0.69315 |  
| Learning rate tuning | learning rate 0.0001->0.0002 | 0.7061 | 0.70507 |
| More model tuning | hidden_dim=1024 <br> hidden_layers 2->3 | 0.71653 | 0.71416 |
| Dynamic learning rate (update every epoch) | CosineAnnealingLR | 0.72599 | 0.7233 |
| Make network deeper | hidden layers 3->4 | 0.7268 | 0.72438 |
| Add dropout | Dropout layer (p=0.1) before linear | 0.72849 | 0.72714 |
| Add dropout | Dropout layer (p=0.25) before linear | 0.72921 | 0.72762 |