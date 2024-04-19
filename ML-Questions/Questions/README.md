- [Machine Learning Interview Questions](#machine-learning-interview-questions)
  - [General Questions](#general-questions)
    - [What's the trade-off between `bias` and `variance`?](#whats-the-trade-off-between-bias-and-variance)
    - [What is `gradient descent`?](#what-is-gradient-descent)
    - [Explain `over-fitting` and `under-fitting` and how to combat them?](#explain-over-fitting-and-under-fitting-and-how-to-combat-them)
    - [How do you combat the `curse of dimensionality`?](#how-do-you-combat-the-curse-of-dimensionality)
    - [What is `regularization`, why do we use it, and give some examples of common methods?](#what-is-regularization-why-do-we-use-it-and-give-some-examples-of-common-methods)
  - [Company Interview Questions](#company-interview-questions)
    - [Google](#google)
    - [Meta](#meta)
    - [Amazon](#amazon)
    - [TikTok](#tiktok)
    - [Nvidia](#nvidia)
    - [Uber](#uber)
    - [Apple](#apple)
    - [Tesla](#tesla)
    - [Stripe](#stripe)
    - [Scale.AI](#scaleai)
    - [Snowflake](#snowflake)
    - [Pinterest](#pinterest)
- [References](#references)

# Machine Learning Interview Questions

## General Questions

### What's the trade-off between `bias` and `variance`?

If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data. [[source]](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)

### What is `gradient descent`?

Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost).

Gradient descent is best used when the parameters cannot be calculated analytically (e.g. using linear algebra) and must be searched for by an optimization algorithm. [[source]](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)

### Explain `over-fitting` and `under-fitting` and how to combat them?

ML/DL models essentially learn a relationship between its given inputs(called training features) and objective outputs(called labels). Regardless of the quality of the learned relation(function), its performance on a test set(a collection of data different from the training input) is subject to investigation.

Most ML/DL models have trainable parameters which will be learned to build that input-output relationship. Based on the number of parameters each model has, they can be sorted into more flexible(more parameters) to less flexible(less parameters).

The problem of Underfitting arises when the flexibility of a model(its number of parameters) is not adequate to capture the underlying pattern in a training dataset. Overfitting, on the other hand, arises when the model is too flexible to the underlying pattern. In the later case it is said that the model has “memorized” the training data.

An example of underfitting is estimating a second order polynomial(quadratic function) with a first order polynomial(a simple line). Similarly, estimating a line with a 10th order polynomial would be an example of overfitting. [[source]](https://towardsdatascience.com/overfitting-vs-underfitting-a-complete-example-d05dd7e19765)

### How do you combat the `curse of dimensionality`?

[[source]](https://towardsdatascience.com/why-and-how-to-get-rid-of-the-curse-of-dimensionality-right-with-breast-cancer-dataset-7d528fb5f6c0)

- Feature Selection(manual or via statistical methods)
- Principal Component Analysis (PCA)
- Multidimensional Scaling
- Locally linear embedding

### What is `regularization`, why do we use it, and give some examples of common methods?

A technique that discourages learning a more complex or flexible model, so as to avoid the risk of overfitting. Examples

- Ridge (L2 norm)
- Lasso (L1 norm)

The obvious disadvantage of ridge regression, is model interpretability. It will shrink the coefficients for least important predictors, very close to zero. But it will never make them exactly zero. In other words, the final model will include all predictors. However, in the case of the lasso, the L1 penalty has the effect of forcing some of the coefficient estimates to be exactly equal to zero when the tuning parameter λ is sufficiently large. Therefore, the lasso method also performs variable selection and is said to yield sparse models. 

## Company Interview Questions

These were collected from all the web resources, not necessarily using machine learning or deep learning techniques, could be general system design questions.

### Google

1. 如何给一串corpus做tokenization （LLM相关职位）

### Meta

1. 在线代码比赛
2. [https://www.1point3acres.com/bbs/thread-1046445-1-1.html](https://www.1point3acres.com/bbs/thread-1046445-1-1.html)

### Amazon

[Amazon OA](https://wdxtub.com/interview/14520850399861.html)

1. maintain的题 设计一个system去支持查找商品库存


### TikTok

1. 如何设计transaction system，不涉及三方支付系统，如何保证收/付款一致性。

### Nvidia

1. 在训练的时候用 dropout 但在inference/testing中不用dropout会有什么影响？

### Uber

1. 给定budget，设计一个推荐discount promo的系统。（希望能最大化profit）[https://www.1point3acres.com/bbs/thread-1048175-1-1.html](https://www.1point3acres.com/bbs/thread-1048175-1-1.html)
2. 手写KMeans
3. 手写Cross Validation

### Apple

1. 要求从头（包括相关的数据预处理或者database建立）设计一个healthy food recommendation system。先跟面试官clarify healthy food的定义，后来narrow down到说从高protein的食物入手就行，然后聊能access什么用户信息，什么食物信息，聊需要建立怎样的database方便query，最后基本formulate成information retrieval+ranking的经典结构。Follow-up是如果基于这个系统，设计一个新的系统，是用户到了一个餐厅后，依照餐厅的menu推荐dishes，怎么做。大概思路就是menu上一般会列举每一个dish的食材，然后套用之前已经建好的系统，按照食材推荐高protein的dish。
2. 用说话的音频数据判断是否有帕金森综合症。
3. 设计一个类似于Netflix home page上的其中一个推荐video的panel，要求按照用户的之前看过的视频的semantic similarity推荐，关键点就是，你已经有其他panel是按照类似于CTR推荐的了，这个panel必须着重强调semantic similarity的话你怎么train，怎么保证它和其他panel 不完全一样。

### Tesla

1. kafka design (distributed message queue system)

### Stripe

1. 设计一个仓库系统存储注册与登录的信息

### Scale.AI

1. Search System Design

### Snowflake

[https://www.1point3acres.com/bbs/thread-1046567-1-1.html](https://www.1point3acres.com/bbs/thread-1046567-1-1.html)

### Pinterest

1. 设计pin feed recommendation，先聊数据，后聊各种vision-language modeling相关模型方法

# References

- [Machine Learning Articles](https://github.com/christianversloot/machine-learning-articles)
- [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design)