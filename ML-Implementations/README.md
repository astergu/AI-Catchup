- [Machine Learning Basics](#machine-learning-basics)
  - [Supervised Learning](#supervised-learning)
  - [Unsupervised Learning](#unsupervised-learning)
  - [Data Preprocessing](#data-preprocessing)
  - [Parameter Initialization](#parameter-initialization)
  - [Loss Function](#loss-function)
  - [Activation Function](#activation-function)
  - [Optimizer](#optimizer)
  - [Prevent Overfitting](#prevent-overfitting)
  - [Model Evaluation](#model-evaluation)
    - [Regression Evaluation](#regression-evaluation)
    - [Classification Evaluation](#classification-evaluation)
- [Advanced ML Models](#advanced-ml-models)
  - [CNN](#cnn)
  - [RNN](#rnn)
  - [Seq2Seq](#seq2seq)
- [Machine Learning Applications](#machine-learning-applications)
  - [Natural Language Processing](#natural-language-processing)
    - [Tutorials](#tutorials)
  - [Recommender Systems](#recommender-systems)
    - [Ads Recommendation](#ads-recommendation)
  - [Searching](#searching)
  - [Ranking](#ranking)
- [Alternative Resources](#alternative-resources)

# Machine Learning Basics

## Supervised Learning

- Classification
  - Logic Regression
- Regression
  - [Linear Regression](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/linear_regression/main.py) (implemented by yunjey)
- Decision Trees and Random Forest
- SVM
- K-Nearest Neighbors
- Clustering
- Boosting
- Dimension Reduction (PCA, LDA, Factor Analysis)
- Neural Networks
  - Layers
  - Optimizers
    - Adagrad
  - Batch Normalization

## Unsupervised Learning


## Data Preprocessing

- Mean Subtraction
  - In practice, the mean is calculated only across the training set, and this mean is subtracted from the training, validation, and testing sets.
- Normalization
  - scale every input feature dimension to have similar ranges of magnitudes.
- Whitening
  - convert the data to have an identity covariance matrix.


## Parameter Initialization

- Xavier initialization

## Loss Function

- Classification
  - Negative Log Likelihood Loss
  - Max-margin hinge loss
- Softmax
  - Full Softmax
  - Hierarchical Softmax



## Activation Function

- Non-linear
  - logistic (sigmoid) `(0, 1)`
    - Drawbacks
      - Sigmoids saturate and kill gradients. 
      - Sigmoid outputs are not zero-centered.
    - $f(z)=\frac{1}{1+exp(-z)}$
  - tanh `[-1, 1]`
    - $tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}$
    - its activations saturate, but unlike the sigmoid neuron its output is zero-centered.
    - the tanh neuron is simply a scaled sigmoid neuron: $tanh(x)=2\sigma(2x)-1$
  - hard tanh `[-1, 1]`
    - -1 if x < -1
    - x if -1 <= x <= 1
    - 1 if x > 1
  - ReLU (Rectified Linear Unit) `[0, `$\infty$`)`
    - $ReLU(z) = max(z, 0)$
    - greatly accelerate the convergence of stochastic gradient descent compared to the sigmoid/tanh functions.
    - ReLU can be implemented by simply thresholding a matrix of activations at zero.
    - Unfortunately, ReLU units can be fragile during training and can "die". With a proper setting of the learning rate, this is less frequently an issue.
  - Leaky ReLU
    - $leaky(z)=max(z, k \cdot z)$, where $0<k<1$
  - Swish
    - $swish(x)=x\cdot logistic(x)$

## Optimizer

- Learning rate
  - fixed learning rate
  - annealing
    - exponential decay $\alpha(t)=\alpha_0 e^{-kt}$
    - decrease over time $\alpha(t)=\frac{\alpha_0 \tau}{max(t, \tau)}$
  - Start optimizer with an initial rate, around 0.001
- Momentum
- Gradient Descent
  - Stochastic Gradient Descent
- Adagrad
  - SGD with the learning rate varying for each parameter
  - $\theta_{t, i}=\theta_{t-1,i}-\frac{\alpha}{\sqrt{\sum_{\tau=1}^t g_{\tau,i}^2}}g_{t,i}$ where $g_{t,i}=\frac{\partial}{\partial \theta_i^t}J_t(\theta)$
- RMSProp
  - RMSProp is a variant of AdaGrad that utilizes a moving average of squared gradients
- Adam
  - A variant of RMSProp, with the addition of momentum-like updates
  - A fairly good, safe place to begin in many cases


## Prevent Overfitting

- Regularization
  - L1 Regularization
  - L2 Regularization
- Dropout
  - Prevent feature co-adaptation: A feature cannot only be useful in the presence of particular other features.  
  - `Must typically divide the outputs of each neuron during testing by a certain value?`
  - During training
    - randomly set input to 0 with probability `p` (often $p=0.5$ except $p-0.15$ for input layer)
  - During testing
    - multiply all weights by $1-p$

## Model Evaluation

### Regression Evaluation

- Mean Squared Error (MSE)

### Classification Evaluation


# Advanced ML Models

## CNN

## RNN

## Seq2Seq


# Machine Learning Applications

## Natural Language Processing

- N-gram
- tokenization
- Word2vec 
  - Model
    - Skip-gram
    - CBOW
  - Loss Functions
    - Naive Softmax
    - Hierarchical Softmax
    - Negative Sampling
  - Improvements
    - Subsampling of very frequent words $p=1-\sqrt{\frac{t}{f}}$
    - Subsampling of Frequent Words
    - Learning Phrases, e.g. New York Times
  - [Implementation](./NLP/Word2vec/word2vec_pytorch.py) ([pytorch word embeddings](https://github.com/pytorch/tutorials/blob/main/beginner_source/nlp/word_embeddings_tutorial.py))
  - Limitations
    - Same word but different synonyms [[Improving Word Representations via Global Context and Multiple Word Prototypes]](https://nlp.stanford.edu/pubs/HuangACL12.pdf)
      - Gather fixed size context windows of all occurrences of the word.
      - Each context is represented by a weighted average of the context words' vectors (using idf-weighting)
      - Apply spherical k-means to cluster these context representations.
      - Each word occurrence is re-labeled to its associated cluster and is used to train the world representation for that cluster.
    - Inability to represent idiomatic phrases that are note compositions of the individual words
      - Identify a large number of phrases first
- Part-Of-Speech tagging (POS)
- Chunking
- Named Entity Recognition (NER)
- Semantic Role Labeling (SRL)
- 

### Tutorials

- [https://github.com/pytorch/tutorials/tree/main/beginner_source/nlp](https://github.com/pytorch/tutorials/tree/main/beginner_source/nlp)
- [https://cs231n.github.io/neural-networks-1/](https://cs231n.github.io/neural-networks-1/)

## Recommender Systems

### Ads Recommendation


## Searching
## Ranking

# Alternative Resources

- [ML from scratch](https://github.com/eriklindernoren/ML-From-Scratch)
- [minGPT](https://github.com/karpathy/minGPT)
- [annotated deep learning paper implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)