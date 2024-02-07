# CS224n Assignment #2: Word2vec

## Writing Questions

1. Prove that the native-softmax loss is the same as the cross-entropy loss between y and $\hat{y}$, i.e.  $-\sum_{w \in Vocab} y_w log(\hat{y}_w) = - log(\hat{y}_o)$

> Answer: <br> 
> <font color='red'> $y_w$ is one-hot vector with a 1 for the true outside word o, and 0 everywhere else. Thus all the sum terms whenever w $\neq$ o are zeros. </font>

2 (1) Compute the partial derivative of $J_{naive-softmax}(v_c, o, U)$ with rescpect to $v_c$. Please write your answer in terms of $y$, $\hat{y}$, $U$, and show your work to receive full credit.

> Answer: <br> 
> <font color='red'> The partial derivative of $J_{naive-softmax}(v_c, o, U)$ with respect to $v_c$:  
> $\frac{\partial J_{naive-softmax}(v_c, o, U)}{\partial v_c}$ <br>
> = $\frac{\partial}{\partial v_c}[-(log(exp(u_o^T v_c)) - log(\sum_{w\in Vocab} exp(u_w^Tv_c)))]$ <br>
> = $\frac{\partial}{\partial v_c}[log(\sum_{w \in Vocab} exp(u_w^Tv_c)) - u_o^Tv_c]$ <br>
> = $\frac{1}{\sum_{w \in Vocab}exp(u_w^Tv_c)}(\sum_{w \in Vocab}exp(u_w^Tv_c)u_w) - u_o$ <br>
> = $\sum_{w \in Vocab}\frac{exp(u_w^Tv_c)}{\sum_{w\in Vocab}exp(u_w^Tv_c)}u_w - u_o$ <br>
> = $\sum_{w\in Vocab}\hat{y_w}u_w - u_o$ <br>
> = $U\hat{y} - u_o$ <br>
> = $U(\hat{y} - y)$ </font>


(2) When is the gradient you computed equal to zero?

> Answer: <font color='red'> </font>


(3) The gradient you found is the difference between two terms. Provide an interpretation of how each of these term improves the word vector when this gradient is subtracted from the word vector $v_c$.

> Answer: <font color='red'> </font>

(4) In many downstream application using word embeddings, L2 normalized vectors (e.g.$u/$)


## Coding: Implementing word2vec

In this part you will implmenet the word2vec model and train your own word vectors with stochastic gradient descent (SGD). Before you begin, first run the following commands within the assignment directory in order to create the appropriate conda virtual environment. You'll probably want to implement and test each part of this section in order, since the questions are cumulative.

```bash
conda env create -f env.yml
conda activate a2
```

Once you are done with the assignment you can deactivate this environment by running:

```bash
conda deactivate
```

1. We will start by implementing methods in `word2vec.py`. You can test a particular method by running `python word2vec.py m` where `m` is the method you would like to test. For example, you can test the sigmoid method by running `python word2vec.py sigmoid`.

(1) Implement the `sigmoid` method, which takes in a vector and applies the sigmoid function to it.

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

(2) Implement the sofxmax loss and gradient in the `naiveSoftmaxLossAndGradient` method.

```python
def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset):
    y_hat = softmax(centerWordVec @ outsideVectors)
    y = np.zeros_like(y_hat)
    y[outsideWordIdx] = 1

    loss = -np.log(y_hat[outsideWordIdx])
    gradCenterVec = (y_hat - y) @ outsideVectors
    gradOutsideVecs = np.outer(y_hat - y, centerWordVec)

    return loss, gradCenterVec, gradOutsideVecs
```

(3) Implement the negative sampling loss and gradient in the `negSamplingLossAndGradient` method.

(4) Implement the skip-gram model in the `skipgram` method.