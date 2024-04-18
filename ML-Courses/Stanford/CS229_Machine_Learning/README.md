# CS229: Machine Learning

[[Course Homepage]](https://cs229.stanford.edu/) [[Youtube Playlist]](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)

| Lecture | Topics | Course Materials | Assignments | 
| ---- | ---- | ---- | ---- |
| Lecture 1 | [Introduction](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=2) | |  |
| Lecture 2 | [Linear Regression and Gradient Descent](https://www.youtube.com/watch?v=4b4MUYve_U8&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=2) | | |

## Lecture 1

- Overview of machine learning types
  - Supervised Learning
    - Regression
    - Classification
  - Unsupervised Learning
    - Clustering
  - Reinforcement Learning

## Lecture 2

- Linear Regression
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
- Gradient Descent
  - Idea
    - start with some $\theta$
    - Keep changing $\theta$ to reduce $J(\theta)$
      - $\theta_j\colonequals\theta_j - \alpha\frac{\partial}{\partial \theta_j}J(\theta)$, for $j \in {1,2...,n}$, where $\alpha$ is the learning rate
    - Repeat until convergence
  - Batch Gradient Descent
    - Do gradient descent on the whole training data as a batch
    - Disadvantage
      - Go through the entire dataset to perform gradient descent can be extremely slow for large datasets