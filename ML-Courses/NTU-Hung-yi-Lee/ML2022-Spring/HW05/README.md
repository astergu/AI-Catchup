# Homework 5: Machine Translation

The task is to translate English to Traditional Chinese.

**Training datasets**

- Paired data
  - TED2020: TED talks with transcripts translated by a global community of volunteers to more than 100 language.
  - We will use (en, zh-tw) aligned pairs.
- Monolingual data
  - More TED talks in traditional Chinese.

## Evaluation Metric

**BLEU**

- Modified n-gram precision (n=1~4).
- Brevity penalty: penalizes short hypotheses.
- The BLEU socre is the geometric mean of n-gram precision, miltiplied by brevity penalty.

## Training tips

- Tokenize data with sub-word units.
- Label smoothing regularization
- Learning rate scheduling
- Back-translation


## Suggest baselines

- `Simple`: 14.58
  - Train a simple RNN seq2seq to achieve translation.
- `Medium`: 18.04
  - Add learning rate scheduler and train longer.
- `Strong`: 25.20
  - Switch to Transformer and tuning hyperparameter.
- `Boss`: 29.13
  - Apply back-translation.