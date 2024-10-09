# CS229: Machine Learning

[[Course Homepage]](https://cs229.stanford.edu/) [[Youtube Playlist]](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)

| Lecture | Topics | Optional Materials | Implementation
| ---- | ---- | ---- | ---- | 
| Lecture 1 | [Introduction](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=2) | |   |
| Lecture 2 | [Linear Regression and Gradient Descent](https://www.youtube.com/watch?v=4b4MUYve_U8&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=2) | - [Perform linear regression with python and scikit](https://github.com/christianversloot/machine-learning-articles/blob/main/performing-linear-regression-with-python-and-scikit-learn.md) |  - [Linear Regression](../../../ML-Implementations/ml_from_scratch/supervised_learning/regression.py) <br> - [Linear Regression [torch version]](../../../ML-Implementations/pytorch/supervised_learning/regression.py) |
| Lecture 3 | [Locally Weighted & Logistic Regression](https://www.youtube.com/watch?v=het9HFqo1TQ&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=3) | |  |
| Lecture 4 | [Perceptron & Generalized Linear Model](https://www.youtube.com/watch?v=iZTeva0WSTQ&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=4) | |  | 
| Lecture 5 | [GDA & Naive Bayes](https://youtu.be/nt63k3bfXS0?si=4j3pzyrjZjacF9rT) | |  | 
| Lecture 6 | [Support Vector Machine](https://youtu.be/lDwow4aOrtg?si=q1sESlXjDHHqV9CQ) | |  | 
| Lecture 7 | [Kernels](https://youtu.be/8NYoQiRANpg?si=Y9-kNgeuo2xflHgz) | | | 
| Lecture 8 | [Data Splits, Models & Cross-Validation](https://youtu.be/rjbkWSTjHzM?si=NGjBbXsMpp4hJ36U) | | | 
| Lecture 9 | [Appox/Estimation Error & ERM](https://youtu.be/iVOxMcumR4A?si=mIAmyGKL75JeYFc0) | | | 
| Lecture 10 | [Decision Trees and Ensemble Methods](https://youtu.be/wr9gUr-eWdA?si=zZqu3-P1DlFNCi0h) | | |  
| Lecture 11 | Introduction to Neural Networks | | | 
| Lecture 12 | Backprop & Improving Neural Networks | | |  
| Lecture 13 | Debugging ML Models and Error Analysis | | |  
| Lecture 14 | Expectation-Maximization Algorithms | | |  
| Lecture 15 | EM Algorithm & Factor Analysis | | |  
| Lecture 16 | Independent Component Analysis & RL | | |  
| Lecture 17 | MDPs & Value/Policy Iteration | | |  
| Lecture 18 | Continuous State MDP & Model Simulation | | |  
| Lecture 19 | Reward Model & Linear Dynamical System | | |  
| Lecture 20 | RL Debugging and Diagnostics | | |  

## Lecture 1: Introduction

- Overview of machine learning types
  - Supervised Learning
    - Regression
    - Classification
  - Unsupervised Learning
    - Clustering
  - Reinforcement Learning

## Lecture 2: Linear Regression and Gradient Descent

- **Linear Regression**
  - one of the most easy supervised learning models
  - Concepts
    - Hypothesis: $h(x)=\sum_0^{n}\theta_j x_j$, where $x_0=1$
    - $\Theta$: parameters
    - $m$: number of training examples
    - $X$: inputs, also known as features
    - $y$: outputs, also known as target values
    - $(x^{(i)}, y^{(i)})$: the $i$th training example
  - How do you choose parameters $\Theta$?
    - Choose $\Theta$ such that $h_{\theta}(x)\approx y$
    - Squared error: $J(\theta)=\frac{1}{2}\sum_{i=0}^m (h_{\theta}(x^{(i)})-y^{(i)})^2$
- **Gradient Descent**
  - Idea
    - start with some $\theta$
    - Keep changing $\theta$ to reduce $J(\theta)$
      - $\theta_j\colonequals\theta_j - \alpha\frac{\partial}{\partial \theta_j}J(\theta)$, for $j \in {1,2...,n}$, where $\alpha$ is the learning rate
    - Repeat until convergence 
      - $\theta_j\colonequals\theta_j-\alpha\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}$ (after derivative computing)
  - Batch Gradient Descent
    - Do gradient descent on the whole training data as a batch
    - Disadvantage
      - Go through the entire dataset to perform gradient descent can be extremely slow for large datasets
  - **Stochastic Gradient Descent**
    - Instead of scanning through the entire dataset, use just one single example to update parameters
    - Repeat until convergence 
      - For $i=1$ to $m$, update $\theta_j\colonequals\theta_j-\alpha(h_{\theta}(x^{(i)})-y^{(i)})x_j^{(i)}$ for every $j$ 
  - **Normal Equation**
    - works only for linear regression
    - set $\nabla_{\theta}J(\theta)=\vec{0}$
    - $\Theta=(X^{\intercal}X)^{-1}X^{\intercal}y$

## Lecture 3: Locally Weighted & Logistic Regression

- Locally Weighted Regression
  - Parametric learning algorithm vs. Non-parametric learning algorithm
    - Parametric: fit fixed set of parameters ($\theta$) to data.
    - Non-parametric: amount of data/parameters grows (linearly) with size of data
  - Fit $\theta$ to minimize $\sum_{i=1}^m (y^{(i)}-\theta^{\intercal}x^{(i)})^2$, where $w^{(i)}$ is a "weighting" function $w^{(i)}=exp(-\frac{(x^{(i)}-x)^2}{\tau})$
    - If $|x^{(i)}-x|$ is small, then $w^{(i)}\approx 1$
    - If $|x^{(i)}-x|$ is large, then $w^{(i)}\approx 0$
- Probabilistic Interpretation of Linear Regression
  - Why squared error?
    - Assume $y^{(i)}=\theta^{\intercal}x^{(i)}+\epsilon^{(i)}$, where $\epsilon^{(i)}$ is an error, includes unmodelled effects, random noise, etc.
      - $\epsilon^{(i)}\sim \mathcal{N}(0, \sigma^2)$
      - This implies that $y^{(i)}|x^{(i)};\theta \sim \mathcal{N}(\theta^{\intercal}x^{(i)}, \sigma^2)$
      - Likelihood of $\theta$: $L(\theta)=P(\vec{y}|x;\theta)=\prod_{i=1}^m P(y^{(i)}|x^{(i)};\theta)$
      - Log likelihood $l(\theta)=logL(\theta)=mlog\frac{1}{\sqrt{2\pi}\sigma}+\sum_{i=1}^m -\frac{(y^{(i)}-\theta^{\intercal}x^{(i)})^2}{2\sigma^2}$
      - Maximum Likelihood Estimation (MLE): Choose $\theta$ to maximize $l(\theta)$, which is exactly least squared error
- Logistic Regression
  - The most commonly used classification algorithm 
  - One special case of generalized linear models
  - Want $h_{\theta}(x)\in [0,1]$
  - Define "sigmoid" (logistic) function $g(z)=\frac{1}{1+e{-z}}$
  - Thus $h_{\theta}(x)=g(\theta^{\intercal}x)=\frac{1}{1+e^{-\theta^{\intercal}x}}$
  - Since $y\in {0, 1}$, $p(y|x;\theta)=h(x)^y (1-h(x))^{1-y}$
  - Likelihood of parameters $L(\theta)=p(\vec{y}|x;\theta)=\prod_{i=1}^m h_{\theta}(x^{(i)})^{y^{(i)}}(1-h_{\theta}(x^{(i)}))^{1-y^{(i)}}$
  - Log likelihood $l(\theta)=log(L(\theta))=\sum_{i=1}^m y^{(i)}logh_{\theta}(x^{(i)})+(1-y^{(i)})log(1-h_{\theta}(x^{(i)}))$
  - Goal: Choose $\theta$ to maximize $l(\theta)$
  - Use Batch Gradient Ascent to maximize $l(\theta)$
    - $\theta_j\colonequals \theta_j+\alpha\frac{\partial}{\partial \theta_j}l(\theta)=\theta_j+\alpha\sum_{i=1}^m (y^{(i)}-h_{\theta}(x^{(i)}))x_j^{(i)}$
    - $l(\theta)$ has only one global maximum
    - Exactly the same as linear regression
- Newton's method
  - Say you have function $f$, you want to find $\theta, s.t.f(\theta)=0$
  - Update $\theta^{(t+1)}\colonequals \theta^{(t)}-\frac{f(\theta^{(t)})}{f'(\theta^{(t)})}$


## Lecture 4: Perceptron & Generalized Linear Models

- Perceptron
  - Use function $g(z)=1$ if $z \geq 0$, $g(z)=0$ if $z < 0$
  - Update rule for Perceptron and Logistic Regression is the same: $\theta_j \colonequals \theta_j + \alpha(y^{(i)}-h_{\theta}(x)^{(i)})x_j^{(i)}$
- Exponential Familiy
- Generalized Linear Models

## Lecture 5: Gaussian Discriminant Analysis (GDA) & Naive Bayes

- Gaussian Discriminant Analysis (GDA)
  - Generative model
  - Predict $p(x|y)$ with gaussian distribution
- Bayes Rule
  - $p(y|x)=\frac{p(x|y)p(y)}{p(x)}$
- Naive Bayes
  - Given email classification problem, how do you represent as feature vector $x$?
  -  Assume $x_i$ are conditionally independent given $y$
  -  $L(\phi_y, \phi_{j|y})=\prod_{i=1}^m p(x^{(i)},y^{(i)};\phi_y, \phi_{j|y})$
  

## Lecture 6: Support Vector Machines

- Naive Bayes
  - Laplace Smoothing
- Multinomial Event Model
- Support Vector Machine
  - can learn nonlinear relationships
  - Less parameters to fiddle with
  - Optimal margin classifier (separable case)
    - functional margin 
    - geometric margin 

## Lecture 7: Kernels

- SVM = optimal marger classifier + kernel trick
- L1 norm soft margin SVN

## Lecture 8: Data Splits, Models & Cross-Validation

- Bias/Variance
- Regularization
- Train/dev/test splits
- Model selection & Cross validation
  - Cross Validation
    - k-fold CV
    - Leave-one-out CV
- Feature Selection (forward search)

## Lecture 9: Approx/Estimation Error & ERM

- Bias vs. variance
- Approx Estimation
  - Bayes error (Irreducible Error) $\epsilon(g)$
  - Approximation Error $\epsilon(h^*)-\epsilon(g)$
  - Estimation Error $\epsilon(\hat{h})-\epsilon(h^*)$
  - Total Error $\epsilon(\hat{h})=\text{Estimation Error}+\text{Approx Error}+\text{Irreducible Error}=\text{Bias}+\text{Variance}+\text{Irreducible}$
  - **Reduce Variance**
    - More data
    - Decrease number of features
    - Add regularization
  - **Reduce Bias**
    - More complex model
    - Add more features (polynomial features)
    - Less regularization 
- Empirical Risk Minimization (ERM)
  - $\hat{ERM}=argmin_{h\in H} \frac{1}{m}\sum_{i=1}^m 1\{h(x^{(i)})\neq y^{(i)}\}$
- Uniform Convergence
  - Union Bound
  - Hoeffding's Inequality

## Lecture 10: Decision Trees and Ensemble Methods

- Decision Trees
  - Split function $S_p(j,t)=(\{x|x_j<t, x\in R_p\}, \{x|x_j\geq t, x\in R_p\})$ 
  - How to splits
    - Define loss $L(R)$, define $\hat{p_c}$ to be the proportion of examples in $R$ that are of class $c$, thus the loss of misclassification $L_{misclass}=1-\text{max}\hat{p_c}$.
    - $max L(R_p)-(L(R_1)+L(R_2))$, meaning parent loss minus children loss.
    - Cross-entropy loss $L_{cross}=-\sum\hat{p_c}log_2 \hat{p_c}$
    - Gini loss $\sum\hat{p_c}(1-\hat{p_c})$
  - Regression Trees
    - Predict $\hat{y_m}=\frac{\sum_{i\in R_m}y_i}{|R_m|}$
    - Loss $L_{squared}=\frac{\sum_{i\in R_m}(y_i-\hat{y_m})^2}{|R_m|}$
  - Regularization of Decision Trees
    - min leaf size
    - max depth
    - max number of nodes
    - min decrease in loss
    - prunning (misclassification with val set)
  - Runtime
    - $O(nfd)$, where $n$ is number of examples, $f$ is the number of features, $d$ is the max depth of the tree.
  - Pros
    - Easy to explain
    - Interpretable
    - Can deal with categorical variables
    - Generally fast
  - Cons
    - High variance 
    - Bad at additive structure
    - Low predictive accuracy
- Ensemble Methods
  - Ways to ensemble
    - Bagging: Random Forests
    - Boosting: Adaboost, xgboost
- Bagging (Bootstrap Aggregation)
  - Goal: decrease variance
  - Random Forests
    - At each split, consider only a fraction of the total features
- Boosting
  - Goal: decrease bias
  - Determine for classifier $G_m$ a weight $\alpha_m$ proportional to $log(\frac{1-err_m}{err_m})$
  - Each $G_m$ is trained on a re-weighted training set