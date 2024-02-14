# CS224n Assignment #3: Dependency Parsing

In this assignment, you will build a neural dependency parser using PyTorch. In Part 1, you will learn about two general neural network techniques (Adam Optimization and Dropout). In Part 2, you will implement and train a dependency parser using the techniques from Part 1, before analyzing a few erroneous dependency parses.

## Machine Learning & Neural Networks

1. **Adam Optimizer** <br>

(1) First, Adam uses a trick called `momentum` by keeping track of `m`, a rolling average of the gradients: <br>
$m_{t+1} \leftarrow \beta_{1}m_{t} + (1 - \beta_{1})\nabla_{\theta_t}J_{minibatch}(\theta_t)$ <br>
$\theta_{t+1} \leftarrow \theta_t - \alpha m_{t+1}$ <br>

where $\beta_1$ is a hyperparameter between 0 and 1 (often set to 0.9). **Briefly explain in 2-4 sentences (you don't need to prove mathematically, just give an intuition) how using `m` stops the updates from varying as much and why this low variance may be helpful to learning, overall**.

> Answer: <br>
>

(2) Adam extends the idea of `momentum` with the trick of `adaptive learning rates` by keeping track of `v`, a rolling average of the magnitudes of the gradients: <br>
$m_{t+1}\leftarrow \beta_1m_t + (1-\beta_1)\nabla_{\theta_t}J_{minibatch}(\theta_t)$ <br>
$v_{t+1}\leftarrow \beta_2v_t + (1-\beta_2)(\nabla_{\theta_t}J_{minibatch}(\theta_t) \odot \nabla_{\theta_t}J_{minibatch}(\theta_t))$ <br>
$\theta_{t+1}\leftarrow \theta_t - \alpha m_{t+1}$ <br>

where $\odot$ and / denote elementwise multiplication and division (so $z\odot z$ is elementwise squaring) and $\beta_2$ is a hyperparameter between 0 and 1 (often set to 0.99). **Since Adam divides the update by $\sqrt{v}$, which of the model parameters will get larger updates? Why might this help with learning?**

> Answer: <br>
>

2. Dropout is a regularization technique. During training, dropout randomly sets units in the hidden layer `h` to zero with probability $p_{drop}$ (dropping different units each minibatch), and then multiplies `h` by a constant $\gamma$. We can write this as: $h_{drop}=\gamma d\odot h$ <br>
where $d \in {0, 1}^{D_h}$ ($D_h$ is the size of $h$) is a mask vector where each entry is 0 with probability $p_{drop}$ and 1 with probability (1-$p_{drop}$). $\gamma$ is chosen such that the expected value of $h_{drop}$ is $h$: <br> 
$\mathbb{E}_{p_{drop}}[h_{drop}]_i=h_i$ <br>
for all $i \in {1, ..., D_h}$.

(1) **What must $\gamma$ equal in terms of $p_{drop}$? Briefly justify your answer or show your math derivation using the equations given above.** <br>

> Answer: <br>
>

(2) **Why should dropout be applied during training? Why should dropout NOT be applied during evaluation?** 

> Answer: <br>
>