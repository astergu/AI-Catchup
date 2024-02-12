# CS224n Assignment #2: Word2vec

## Writing Questions

1. Prove that the native-softmax loss is the same as the cross-entropy loss between y and $\hat{y}$, i.e.  $-\sum_{w \in Vocab} y_w log(\hat{y}_w) = - log(\hat{y}_o)$

> Answer: <br> 
> $y_w$ is one-hot vector with a 1 for the true outside word o, and 0 everywhere else. Thus all the sum terms whenever w $\neq$ o are zeros.

2. **Compute Derivates** <br>
(1). Compute the partial derivative of $J_{naive-softmax}(v_c, o, U)$ with rescpect to $v_c$. Please write your answer in terms of $y$, $\hat{y}$, $U$, and show your work to receive full credit.
   

> Answer: <br> 
>  The partial derivative of $J_{naive-softmax}(v_c, o, U)$ with respect to $v_c$:  
> $\frac{\partial J_{naive-softmax}(v_c, o, U)}{\partial v_c}$ <br>
> = $\frac{\partial}{\partial v_c}[-(log(exp(u_o^T v_c)) - log(\sum_{w\in Vocab} exp(u_w^Tv_c)))]$ <br>
> = $\frac{\partial}{\partial v_c}[log(\sum_{w \in Vocab} exp(u_w^Tv_c)) - u_o^Tv_c]$ <br>
> = $\frac{1}{\sum_{w \in Vocab}exp(u_w^Tv_c)}(\sum_{w \in Vocab}exp(u_w^Tv_c)u_w) - u_o$ <br>
> = $\sum_{w \in Vocab}\frac{exp(u_w^Tv_c)}{\sum_{w\in Vocab}exp(u_w^Tv_c)}u_w - u_o$ <br>
> = $\sum_{w\in Vocab}\hat{y_w}u_w - u_o$ <br>
> = $U\hat{y} - u_o$ <br>
> = $U(\hat{y} - y)$ 


(2). When is the gradient you computed equal to zero?
   
> Answer: <br>
> When softmax perfectly classifies some word $w \in Vocab$ (with probability of 1) which has a non-zero embedding and also happens to be the output word, i.e. $w = o$. 

(3). The gradient you found is the difference between two terms. Provide an interpretation of how each of these term improves the word vector when this gradient is subtracted from the word vector $v_c$. 

> Answer:<br>
> The first term in the gradient is the weighted sum of all the other word vectors in the vocabulary. The weights are determined by the probability of each word in the vocabulary given the current context. The second term is simply the current word vector. When this gradient is subtracted from the current word vector, the first term of the gradient $\hat{y}$ improves the word vector by incorporating information about the surrounding context and adjusting the vector to be more representative of that context. The second term $y$ ensures that the vector is still anchored to the original word and doesn't stray too far from its original meaning.

(4). In many downstream application using word embeddings, L2 normalized vectors (e.g. $u/||u||_2$ where $||u||_2=\sqrt{\sum_i u_i^2}$) are used instead of their raw forms (e.g. $u$). Now, suppose you would like to classify phrases as being positive or negative. When would L2 normalization take away useful information for the downstream task? When would it not? *Hint: Consider the case where $u_x=\alpha u_y$ for some words $x \neq y$ and some scalar $\alpha$.* 

> Answer: <br>
> In most case, L2 normalization does not take away useful information for the downstream tasks, but improve the discriminative power of the embeddings by making them more invariant to differences in scale. Follow the hint, consider the case where $u_x=\alpha u_y$ for some words $x \neq y$ and some scalar $\alpha$. In this case, L2 normalization would make the embeddings of these words indistinguishable from each other, as they would have the same direction and hence the same normalized form.

3. Compute the partial derivatives of $J_{naive-softmax}(v_c, o, U)$ with respect to each of the 'outside' word vectors, $u_w$'s. There will be two cases: when $w=o$, the true 'outside' word vector, and $w \neq o$, for all the other words. Please write your answer in terms of $y$, $\hat{y}$, and $v_c$. In this subpart, you may use specific elements within these terms as well (such as $y_1$, $y_2$, ...). Note that $u_w$ is a vector while $y_1$, $y_2$, ... are scalars. 

> Answer: <br>
> 
> $\frac{\partial J(v_c, o, U)}{\partial u_w} = -\frac{\partial u_o^Tv_c}{\partial u_w} + \frac{\partial log\sum_{w\in Vocab}exp(u_w^Tv_c)}{\partial u_w} = -y_wv_c + \hat{y_w}v_c = (\hat{y_w}-y_w)^Tv_c$, where $y_w=1$ (if $w=o$), or $y_w=0$ (if $w \neq o$).

4. Write down the partial derivatives of $J_{naive-softmax}(v_c, o, U)$ with respect to $U$. Please break down your answer in terms of the column vectors $\frac{\partial J(v_c, o, U)}{\partial u_1}$,  $\frac{\partial J(v_c, o, U)}{\partial u_2}$, ..., $\frac{\partial J(v_c, o, U)}{\partial u_{|Vocab|}}$. No derivations are necessary, just an answer in the form of a matrix.

> Answer: <br>
> 
> $\frac{\partial J(v_c, o, U)}{\partial U}=(\hat{y}-y)^Tv_c=[\frac{\partial J(v_c, o, U)}{\partial u_1}, \frac{\partial J(v_c, o, U)}{\partial u_2},..., \frac{\partial J(v_c, o, U)}{\partial u_{|Vocab|}}]$


5. The Leaky ReLU (Leaky Rectified Linear Unit) activation function is given by Equation 4 and Figure 2: $f(x) = max(\alpha x, x)$ where $x$ is a scalar and 0 < $\alpha$ < 1, please compute the derivative of $f(x)$ with respect to $x$. You may ignore the case where the derivative is not defined at 0.

> Answer: <br>
> $\frac{\partial f(x)}{\partial x} = 1$ if x > 0, $\frac{\partial f(x)}{\partial x} = \alpha$ if x < 0.

6. The sigmoid function is given by Equation 5: $\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1}$. Please compute the derivative of $\sigma(x)$ with respect to $x$, where $x$ is a scalar. Please write your answer in terms of $\sigma(x)$.

> Answer:<br>
> $\frac{\partial \sigma(x)}{\partial x}=\frac{e^{-x}}{(1+e^{-x})^2}=\frac{1+e^{-x}-1}{(1+e^{-x})^2}=\frac{1}{1+e^{-x}}-\frac{1}{(1+e^{-x})^2}=\frac{1}{(1+e^{-x})}\cdot(1-\frac{1}{1+e^{-x}})=\sigma(x)(1-\sigma(x))$

7. Now we shall consider the Negative Sampling loss, which is alternative of the Naive Softmax loss. Assume that *K* negative samples (words) are drawn from the vocabulary. For simplicity of notation we shall refer to them as $w_1$, $w_2$, ..., $w_K$, and their outside vectors as $u_{w_1}$, $u_{w_2}$, ..., $u_{w_K}$. For this question, assume that the *K* negative samples are distinct. In other words, $i\neq j$ implies $w_i \neq w_j$ for $i, j \in {1,...,K}$. Note that $o \in {w_1,...,w_K}$. For a center word $c$ and an outside word $o$, the negative sampling loss function is given by: <br>
$J_{neg_sample}(v_c, o, U) = -log(\sigma(u_o^Tv_c)) - \sum_{s=1}^{K} log(\sigma(-u_{w_s}^Tv_c))$

for a sample $w_1,...,w_K$, where $\sigma(\cdot)$ is the sigmoid function.

(1). Please compute the partial derivatives of $J_{neg-sample}$ with respect to $v_c$, with repsect to $u_o$, and with respect to the $s^{th}$ negative sample $u_{w_s}$. Please write your answers in terms of the vectors $v_c$, $u_o$, and $u_{w_s}$, where $s \in [1, K]$. 

> Answer: <br>
> $\frac{\partial J_{neg-sample}(v_c, o, U)}{v_c} = \frac{\partial(-log(\sigma(u_o^Tv_c))-\sum_{s=1}^Klog(\sigma(-u_{w_s}^Tv_c)))}{\partial v_c}=-(1-\sigma(u_o^Tv_c))u_o+\sum_{s=1}^K(1-\sigma(-u_{w_s}^Tv_c))u_{w_s}$ <br>
> $\frac{\partial J_{neg-sample}(v_c, o, U)}{u_o}=\frac{\partial(-log(\sigma(u_o^Tv_c))-\sum_{s=1}^Klog(\sigma(-u_{w_s}^Tv_c)))}{\partial u_o}=-(1-\sigma(u_o^Tv_c))v_c$ <br>
> $\frac{\partial J_{neg-sample}(v_c, o, U)}{u_{w_s}}=\frac{\partial(-log(\sigma(u_o^Tv_c))-\sum_{s=1}^Klog(\sigma(-u_{w_s}^Tv_c)))}{\partial u_{w_s}}=(1-\sigma(-u_{w_s}^Tv_c))v_c$ <br>

(2). In lecture, we learned that an efficient implementation of backpropagation leverages the re-use of previously-computed partial derivatives. Which quantity could you reuse amongst the three partial derivatives calculated above to minimize duplicate computation? Write your answer in terms of $U_{o,{w_1,..,w_K}}=[u_o, -u_{w_1},...,-u_{w_K}]$, a matrix with the outside vectors stacked as columns, and 1, $a(K+1) \times 1$ vector of 1's. Addtional terms and functions (other than $U_{o,{w_1,...,w_K}}$ and 1) can be used in your solution.

> Answer: <br>
> The embeddings of negative samples $U_{o,{w_1,..,w_K}}=[u_o, -u_{w_1},...,-u_{w_K}]$ are shared across multiple terms of the loss function as sigmoid function. Therefore, we can reuse the quantity $\sigma(U^Tv_c)-1=[\sigma(u_o^Tvc)-1, \sigma(-u_{w_1}^Tv_c)-1, ..., \sigma(-u_{w_k}^Tv_c)-1]^T$.

(3). Describe with one sentence why this loss function is much more efficient to compute than the naive-softmax loss.

> Answer: <br>
> Negative sampling loss is much more efficient because it samples a fixed number K of vocabulary that are involved in loss computation along with the true outside word, whereas naive softmax loss involves normalizing the unnormalized probability meaning it has to go through all word vectors in the whole vocabulary. 

8. Now we will repeat the previous exercise, but without the assumption that the $K$ sampled words are distinct. Assume that $K$ negative samples (words) are drawn from the vocabulary. For simplicity of notation we shall refer to them as $w_1,w_2,...,w_K$ and their outside vectors as $u_{w_1},...,u_{w_K}$. In this question, you may not assume that the words are distinct. In other words, $w_i=w_j$ may be true when $i \neq j$ is true. Note that $o \notin {w_1,...,w_K}$. For a center word $c$ and an outside word $o$, the negative sampling loss function is given by: $J_{neg_sample}(v_c, o, U) = -log(\sigma(u_o^Tv_c)) - \sum_{s=1}^{K} log(\sigma(-u_{w_s}^Tv_c))$ for a sample $w_1,...,w_K$, where $\sigma(\cdot)$ is the sigmoid function.<br>
Compute the partial derivative of $J_{neg-sample}$ with respect to a negative sample $u_{w_s}$. Please write your answers in terms of the vectors $v_c$ and $u_{w_s}$, where $s\in [1, K]$. <br>
*Hint: break up sum in the loss function into two sums: a sum over all sampled words equal to $w_s$ and a sum over all sampled words not equal to $w_s$. *

> Answer: <br>
> Firstly breaking up the sum, we get<br>
> $J_{neg-sample}(v_c, o, U)=-log(\sigma(u_o^Tv_c))-\sum_{w_j=w_s}log(\sigma(-u_{w_j}^Tv_c))-\sum_{w_j\neq w_s}log(\sigma(-u_{w_s}^Tv_c))$ <br>
> Now it is obvious that the first and last term vanish by taking the derivative and we are left with $\frac{\partial J_{neg-sample}(v_c, o, U)}{\partial u_{w_s}}=-\sum_{w_j=w_s}(\sigma(-u_{w_s}^Tv_c)-1)v_c$

9. Suppose the center word is $c=w_t$ and the context window is $[w_{t-m},...,w_{t-1},w_t, w_{t+1},...,w_{t+m}]$, when $m$ is the context window size. Recall that for the skip-gram version of `word2vec`, the total loss for the context window is $J_{skip-gram}(v_c, w_{t-m},...,w_{t+m},U)=\sum_{-m\leq j \leq m} J(v_c, w_{t+j}, U)$. Here, $J(v_c, w_{t+j}, U) represents an arbitrary loss term for the center word $c=w_t$ and outside word $w_{t+j}$. $J(v_c, w_{t+j}, U)$ could be $J_{naive-softmax}(v_c, w_{t+j}, U)$ or $J_{neg-sample}(v_c, w_{t+j}, U)$, depending our your implementation.

<br>

Write down three partial derivatives in terms of $\frac{\partial J(v_c, w_{t+j}, U)}{\partial U}$ and $\frac{\partial J(v_c, w_{t+J}, U)}{\partial v_c}$. This is very simple - each solution should be one line.

<br>

> Answer: <br>
> $\frac{\partial J_{skip-gram}(v_c, w_{t-m},...,w_{t+m}, U)}{\partial U}=\sum_{j\neq 0}\frac{\partial J(v_c,w_{t+j},U)}{\partial U}$ <br>
> $\frac{\partial J_{skip-gram}(v_c, w_{t-m},...,w_{t+m}, U)}{\partial v_c}=\sum_{j \neq 0}\frac{\partial J(v_c,w_{t+j},U)}{\partial v_c}$ <br>
> $\frac{\partial J_{skip-gram}(v_c, w_{t-m},...,w_{t+m}, U)}{\partial v_w}=0$ when $w \neq c$ 


## Coding: Implementing word2vec

In this part you will implement the word2vec model and train your own word vectors with stochastic gradient descent (SGD). Before you begin, first run the following commands within the assignment directory in order to create the appropriate conda virtual environment. You'll probably want to implement and test each part of this section in order, since the questions are cumulative.

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

(2) Implement the softmax loss and gradient in the `naiveSoftmaxLossAndGradient` method.

```python
def naiveSoftmaxLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset): # v_c, o, U
    y_hat = softmax(centerWordVec @ outsideVectors)
    y = np.zeros_like(y_hat)
    y[outsideWordIdx] = 1

    loss = -np.log(y_hat[outsideWordIdx])
    gradCenterVec = (y_hat - y) @ outsideVectors
    gradOutsideVecs = np.outer(y_hat - y, centerWordVec)

    return loss, gradCenterVec, gradOutsideVecs
```

(3) Implement the negative sampling loss and gradient in the `negSamplingLossAndGradient` method.

```python
def negSamplingLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset, K=10):
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    uniq, idx, n_reps = np.unique(indices, return_index=True, return_counts=True)
    U = outsideVectors[uniq]
    n_reps[idx==0] *= -1 # u_o, n_reps=1
    U[idx!=0] *= -1 # u_w
    S = sigmoid(centerWordVec @ U.T) # sigmoid((u_w, u_o)*v_c)

    # Find loss and derivatives w.r.t. v_c, U
    loss = -np.sum(np.abs(n_reps) * np.log(S))
    gradCenterVec = np.abs(n_reps) * (1 - S) @ -U
    gradOutsideVecs = np.zeros_like(outsideVectors)
    gradOutsideVecs[uniq] = n_reps[:, None] * np.outer(1 - S, centerWordVec)

    return loss, gradCenterVec, gradOutsideVecs
```
(4) Implement the skip-gram model in the `skipgram` method.

```python
def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE (~8 Lines)
    centerWordIdx = word2Ind[currentCenterWord]
    centerWordVec = centerWordVectors[centerWordIdx]

    for outsideWord in outsideWords:
        outsideWordIdx = word2Ind[outsideWord]
        l, gradVc, gradOutsideVecs = word2vecLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset)
        loss += l
        gradCenterVecs[centerWordIdx] += gradVc
        gradOutsideVectors += gradOutsideVecs
    
    return loss, gradCenterVecs, gradOutsideVectors
```

2. Complete the implementation for your SGD optimizer in the `sgd` method of `sgd.py`. Test your implementation by running `python sgd.py`.

```python
def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False,
        PRINT_EVERY=10):
    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000

    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    exploss = None

    for iter in range(start_iter + 1, iterations + 1):
        # You might want to print the progress every few iterations.

        loss = None
        ### YOUR CODE HERE (~2 lines)
        loss, dx = f(x)
        x -= step * dx
        ### END YOUR CODE

        x = postprocessing(x)
        if iter % PRINT_EVERY == 0:
            if not exploss:
                exploss = loss
            else:
                exploss = .95 * exploss + .05 * loss
            print("iter %d: %f" % (iter, exploss))

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x
```

3. Now we are going to load some real data and train word vectors with everything you just implemented! We are going to use the Stanford Sentiment Treebank (SST) dataset to train word vectors, and later apply them to a simple sentiment analysis task. You will need to fetch the datasets first. To do this, run `sh get_datasets.sh`. There is no additional code to write for this part, just run `python run.py`.

> [!NOTE]
> The training process may take a long time depending on the efficiency of your implementation and the compute power of your machine (**an efficient implementation takes one to two hours**).
> After 40,000 iterations, the script will finish and a visualization of your word vectors will appear. It will also be saved as `word_vectors.png` in your project directory.

[!The result of my sentiment analysis training](./word_vectors.png)