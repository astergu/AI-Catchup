# CS224n Assignment #5: Self-Attention, Transformers, and Pretraining

This assignment is an investigation into Transformer self-attention building blocks, and the effects of pre-training. It covers mathematical properties of Transformers and self-attention through written questions. Further, you'll get experience with practical system-building through repurposing an existing codebase.

## Attention Exploration

Multi-head self-attention is the core modeling component of Transformers. 

Recall that attention can be viewed as an operation on a *query* vector $q \in \mathbb{R}^d$, a set of *value* vectors $\{v_1,...,v_n\}$, $v_i \in \mathbb{R}^d$, and a set of *key* vectors $\{k_1,...,k_n\}$, $k_i \in \mathbb{R}^d$, specified as follows:

$c=\sum_{i=1}^n v_i \alpha_i$

$\alpha_i=\frac{exp(k_i^T q)}{\sum_{j=1}^n exp(k_j^Tq)}$

with $alpha=\{\alpha_1,...,\alpha_n\}$ termed the "attention weights". Observe that the output $c \in \mathbb{R}^d$ is an average over the value vectors weighted with respect to $\alpha$.

1. **Copying in attention**. One advantage of attention is that it's particularly easy to "copy" a value vector to the output *c*. In this problem, we'll motivate why this is the case.
   
a. **Explain** why $\alpha$ can be interpreted as a categorical probability distribution.
   
> There are $n$ $\alpha$ scores corresponding to each term in the sequence. Each score is between 0 and 1 and can be interpreted as a proability. It is a distribution because all scores are normalized, i.e., they sum up to 1.

b. The distribution $\alpha$ is typically relatively "diffuse"; the probability mass is spread out between many different $\alpha_i$. However, this is not always the case. **Describe** (in onr sentence) under what conditions the categorical distribution $\alpha$ puts almost all of its weight on some $\alpha_j$, where $j \in {1,...,n}$ (i.e. $\alpha_j \gg \sum_{i\neq j} \alpha_i$). What must be true about the query *q* and/or the keys ${k_1,...,k_n}$?

> If the key values $k_j$ compared to other key values $k_{i\neq j}$ are large, then the dot product between the key and the query will be large. This will cause softmax to put most of its probability mass onto this large value.

c. Under the conditions you gave in (ii), **describe** the output *c*.

> $j^{th}$ value will have the most weight thus $c$ will be similar to $v_j$, i.e., $c \approx v_j$.

d. **Explain** (in two sentences or fewer) what your answer to (ii) and (iii) means intuitively.

> If the dot product between some $j_{th}$ word's *key* and *query* is very large compared to other word's *keys* and the same *query*, then the *attention output* for that $j^{th}$ word will approach its *value*. It's as if the *value* is "coppied" to the output.

2. **An average of two**. Instead of focusing on just one vector $v_j$, a Transformer model might want to incorporate information from *multiple* source vectors. Consider the case where we instead want to incorporate information from **two** vectors $v_a$ and $v_b$, with corresponding key vectors $k_a$ and $k_b$.

a. How should we combine two $d$-dimensional vectors $v_a$, $v_b$ into one output vector $c$ in a way that preserves information from both vectors? In machine learning, one common way to do so is to take the average: $c=\frac{1}{2}(v_a+v_b)$. It might seem hard to extract information about the original vectors $v_a$ and $v_b$ from the resulting $c$, but under certain conditions one can do so. <br> Suppose that although we don't know $v_a$ or $v_b$, we do know that $v_a$ lies in a subspace $A$ formed by the $m$ basis vectors $\{a_1,a_2,...,a_m\}$, while $v_b$ lies in a subspace $B$ formed by the $p$ basis vectors $\{b_1,b_2,...,b_p\}$. <br> Using the basis vectors $\{a_1,a_2,...,a_m\}$, construct a matrix $M$ such that for arbitrary vectors $v_a\in A$ and $v_b \in B$, we can use $M$ to extract $v_a$ from the sum vector $s=v_a+v_b$. In other words, we want to construct $M$ such that for any $v_a$, $v_b$, $M_s=v_a$ holds for your $M$.

> Assume that $A$ is a matrix of concatenated basis vectors $\{a_1,a_2,...,a_m\}$ and $B$ is a matrix of concatenated basis vectors $b_1,b_2,...,b_p$. Linear combinations of vectors of $v_a$ and $v_b$ can then be expressed as: 
> $v_a=c_1a_1+c_2a_2+...+c_ma_m=Ac$
> $v_b=d_1b_1+d_2b_2+...+d_pb_p=Bd$
> We need to construct such $M$ which, when multiplied with $v_b$, products $0$, and when multiplied with $v_a$, produces the same vector (in terms of its own space):
> $Ms=v_a$
> $Mv_a+Mv_b=v_a$
> It is easy to see that, since $a_j^{\intercal}b_k=0$ for all $j$,$k$, $A^{\intercal}B=0$. And, since $a_i^{\intercal}a_j=0$ whenever $j\neq i$ and since $a_i^{\intercal}a_j=1$ whenever $j=i$ because vectors are normalized, $A^{\intercal}A=I$. If we substitute $M$ with $A^{\intercal}$:
> $A^{\intercal}Ac+A^{\intercal}Bd=Ic+0d=c$
> And we know that in terms of $\mathbb{R}^d$, $v_a$ is just a collection of constants $c$. Thus, $M=A^{\intercal}$.

b. As before, let $v_a$ and $v_b$ be two value vectors corresponding to key vectors $k_a$ and $k_b$, respectively. Assume that (1) all key vectors are orthogonal, so $k_i^{\intercal}k_j=0$ for all $i\neq j$; and (2) all key vectors have norm 1. **Find an expression** for a query vector $q$ such that $c\approx \frac{1}{2}(v_a+v_b)$, and justify your answer.

> Assume that $c$ is approximated as follows:
> $c\approx \frac{1}{2}v_a+\frac{1}{2}v_b$
> This means we want $\alpha_a\approx 0.5$ and $\alpha_b\approx 0.5$, which can be achieved when (whenever $i\neq a$ and $i\neq b$):
> $k_a^{\intercal}q\approx k_b^{\intercal}q \gg k_i^{\intercal}q$
> Like explained in the previous question, if the dot product is big, the probability mass will also be big and we want a balanced mass between $\alpha_a$ and $\alpha_b$. $q$ will be largest for $k_a$ and $k_b$ when it is a large multiplicative of a vector that contains a component in $k_a$ direction and in $k_b$ direction:
> $q=\beta(k_a+k_b)$, where $\beta\gg 0$
> Now, since the keys are orthogonal to each other, it is easy to see that:
> $k_a^{\intercal}q=\beta; k_b^{\intercal}q=\beta; k_i^{\intercal}q=0$, whenever $i\neq a$ and $i\neq b$
> Thus when we exponentiate, only $exp(\beta)$ will matter, because $exp(0)$ will be insignificant to the probability mass. We get that:
> $\alpha_a=\alpha_b=\frac{exp(\beta)}{n-2+2exp(\beta)}\approx\frac{1}{2}$, for $\beta\gg 0$

3. **Drawbacks of single-headed attention**. In the previous part, we saw how it was *possible* for a single-head attention to focus equally on two values. The same concept could easily be extended to any subset of values. In this question we'll see why it's not a *practical* solution. Consider a set of key vectors $\{k_1,...,k_n\}$ that are now randomly sampled, $k_i\sim \mathcal{N}(\mu_i,\Sigma_i)$, where the means $\mu_i\in \mathbb{R}^d$ are known to you, but the covariances $\Sigma_i$ are unknown. Further, assume that the means $\mu_i$ are all perpendicular; $\mu_i^{\intercal}\mu_j=0$ if $i\neq j$, and unit norm, $\|\mu_i\|=1$.
   
a. Assume that the covariance matrices are $\Sigma_i=\alpha I, \forall i\in\{1,2,...,n\}$, for vanishingly small $\alpha$. Design a query $q$ in terms of the $\mu_i$ such that as before, $c\approx \frac{1}{2}(v_a+v_b)$, and provide a brief argument as to why it works.

> Since the variances (diagonal covariance values) for $i \in \{1,2,...,n\}$ are vanishingly small, we can assume each key vector is close to its mean vector:
> $k_i\approx \mu_i$
> Because all the mean vectors are perpendicular, the problem reduces to the previous case when all keys were perpendicular to each other. $q$ can now be expressed as:
> $q=\beta(\mu_a+\mu_b)$, where $\beta \gg 0$

b. Though single-headed attention is resistant to small perturbations in the keys, some types of larger perturbations may pose a bigger issue. Specifically, in some cases, one key vector $k_a$ may be larger or smaller in norm than the others, while still pointing in the same direction as $\mu_a$. As an example, let us consider a covariance for item $a$ as $\Sigma_a=\alpha I + \frac{1}{2}(\mu_a\mu_a^{\intercal})$ for vanishingly small $\alpha$. This causes $k_a$ to point in roughlly the same direction as $\mu_a$, but with large variances in magnitude. Further, let $\Sigma_i=\alpha I$ for all $i\neq a$. 

When you sample $\{k_1,...,k_n\}$ multiple times, and use the $q$ vector that you defined in part a., what do you expect the vector $c$ will look like qualitatively for different samples? Think about how it differs from part (a) and how $c$'s variance would be affected.

> Since $\mu_i^{\intercal}u_i=1$, $k_a$ varies between $(\alpha+0.5)\mu_a$ and $(\alpha+1.5)\mu_a$. All other $k_i$, whenever $i\neq a$, almost don't vary at all. Noting that $\alpha$ is vanishingly small:
> $k_a\approx\gamma\mu_a$, where $\gamma\sim\mathcal{N}(1, 0.5)$
> $k_i\approx\mu_i$, whenever $i\neq a$
> Since $q$ is most similar in drections $k_a$ and $k_b$, we can assume that the dot product between $q$ and any other key vector is 0 (since all key vectors are orthogonal). Thus there are 2 cases to consider (note that means are normalized and orthogonal to each other):
> $k_a^{\intercal}q\approx\gamma\mu_a^{\intercal}\beta(\mu_a+\mu_b)\approx\gamma\beta$, where $\beta\gg 0$
> $k_b^{\intercal}q\approx\mu_b^{\intercal}\beta(\mu_a+\mu_b)\approx\beta$, where $\beta\gg 0$
> We can now directly solve for coefficients $\alpha_a$ and $\alpha_b$, remembering that for large $\beta$ values exp(0) are insignificant (note how $\frac{exp(a)}{exp(a)+exp(b)}=\frac{exp(a)}{exp(a)+exp(b)}\frac{exp(-a)}{exp(-a)}=\frac{1}{1+exp(b-a)}$):
> $\alpha_a\approx\frac{exp(\gamma\beta)}{exp(\gamma\beta)+exp(\beta)}\approx\frac{1}{1+exp(\beta(1-\gamma))}$
> $\alpha_b\approx\frac{exp(\beta)}{exp(\beta)+exp(\gamma\beta)}\approx\frac{1}{1+exp(\beta(\gamma-1))}$
> Since $\gamma$ varies between 0.5 and 1.5, and since $\beta\gg 0$, we have that:
> $\alpha_a\approx\frac{1}{1+\infty}\approx 0; \alpha_b\approx\frac{1}{1+0}\approx 1$; when $\gamma=0.5$
> $\alpha_a\approx\frac{1}{1+0}\approx 1; \alpha_b\approx\frac{1}{1+\infty}\approx 0$; when $\gamma=1.5$
> Since $c\approx\alpha_av_a+\alpha_bv_b$ because other terms are insignificant when $\beta$ is large, we can see that $c$ oscillates between $v_a$ and $v_b$:
> $c\approx v_b$, when $\gamma\rightarrow 0.5$; $c\approx v_a$, when $\gamma\rightarrow 1.5$

4. **Benefits of multi-headed attention**: Now we'll see some of the power of multi-headed attention. We'll consider a simple version of multi-headed attention which is identical to single headed self-attention as we've presented it in this homework, except two query vectors ($q_1$ and $q_2$) are defined, which leads to a pair of vectors ($c_1$ and $c_2$), each the output of single-headed attention given its respective query vector. The final output of the multi-headed attention is their average, $\frac{1}{2}(c_1+c_2)$. Consider a set of key vectors $\{k_1,...,k_n\}$ that are randomly sampled, $k_i\sim \mathcal{N}(\mu_i,\Sigma_i)$, where the means $\mu_i$ are known to you, but the covariances $\Sigma_i$ are unknown. Also as before, assume that the means $\mu_i$ are mutually orthogonal; $\mu_i^{\intercal}\mu_j=0$ if $i\neq j$, and unit norm, $\|\mu_i\|=1$.

a. Assume that the covariance matrices are $\Sigma_i=\alpha I$, for vanishingly small $\alpha$. Design $q_1$ and $q_2$ such that $c$ is approximately equal to $\frac{1}{2}(v_a+v_b)$. Note that $q_1$ and $q_2$ should have different expressions.

> We can design $q_1$ and $q_2$ such that one of them copies $v_a$ and another copies $v_b$. Since all keys are similar to their means and following the explanation in question (1)d, we express the queries are:
> $q_1=\beta\mu_a, q_2=\beta\mu_b$, for $\beta\gg 0$
> This give us (since means are orthogonal):
> $c_1\approx v_a; c_2\approx v_b$
> And since multi-headed attention is just an average of the 2 values, we can see that:
> $c\approx\frac{1}{2}(v_a+v_b)$

b. Assume that the covariance matrices are $\Sigma_a=\alpha I + \frac{1}{2}(\mu_a\mu_a^{\intercal})$ for vanishingly small $\alpha$, and $\Sigma_i=\alpha I$ for all $i\neq a$. Take the query vectors $q_1$ and $q_2$ that you designed in part a. What, qualitatively, do you expect the output $c$ to look like across different samples of the key vectors? Explain briefly in terms of variance in $c_1$ and $c_2$. You can ignore cases in which $k_a^{\intercal}q_i<0$.

> With regards to question (3)b, if we choose $q_1=\beta\mu_a$ and $q_2=\beta\mu_b$, we get that (note that all other key-query dot products will be insignificant):
> $k_a^{\intercal}q_1=\gamma\mu_a^{\intercal}\beta\mu_a=\gamma\beta$, where $\beta\gg 0$
> $k_b^{\intercal}q_2=\gamma\mu_b^{\intercal}\beta\mu_b=\beta$, where $\beta\gg 0$
> We can solve for $\alpha$ values (again, note that all other key-query dot products will be insignificant when $\beta$ is large):
> $\alpha_{a1}\approx\frac{exp(\gamma\beta)}{exp(\gamma\beta)}\approx 1; \alpha_{b2}\approx\frac{exp(\beta)}{exp(\beta)}\approx 1$
> Since we can say that $\alpha_{i1}\approx 0$ for any $i\neq a$ and $\alpha_{i2}\approx 0$ for any $i\neq b$ is easy to see that:
> $c_1\approx v_a, c_2\approx v_b$
> Which means that the final output will always approximately be an average of the values:
> $c\approx\frac{1}{2}(v_a+v_b)$

## Pretrained Transformer models and knowledge access

You'll train a Transformer to perform a task that involves accessing knowledge about the world knowledge which isn't provided via the task's training data.

The code you're provided with is a form of Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT). It's nicer than most research code in that it's relatively simple and transparent. The "GPT" in minGPT refers to the Transformer language model of OpenAI, originally described in [this paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf).

As in previous assignments, you will want to develop on your machine locally, then run training on Azure/Colab. You can use the same conda environment from previous assignments for local development, and the same process for training on a GPU. You'll need around 5 hours for training. **Note that dataset multi-processing can fail on local machines without GPU, so to debug locally, you might have to change num_workers to 0**.

1. **Check out the demo.** <br> In the `mingpt-demo/` folder is a Jupyter notebook `play_char.ipynb` that trains and samples from a Transformer language model. Take a look at it (locally on your computer) to get somewhat familiar with how it defines and trains models. Some of the code you're writing below will be inspired by what you see in this notebook.
2. **Read through `NameDataset` in `src/dataset.py`, our dataset for reading name-birthplace pairs**. <br> The task we'll be working on with our pretrained models is attempting to access the birth place of a notable person, as written in their Wikipedia page. We will think of this as a particularly simple form of question answering:

> Q: Where was [person] born?
> A: [place]

From now on, you'll be working with the `src/` folder. **The code in `mingpt-demo/` won't be changed or evaluated for this assignment.** In `dataset.py`, you'll find the class `NameDataset`, which reads a TSV (tab-separated values) file of name/place pairs and produces examples of the above form that we can feed to our Transformer model.

To get a sense of the examples we'll be working with, if you run the following code, it'll load your `NameDataset` on the training set `birth_places_train.tsv` and print out a few examples.

```bash
python src/dataset.py namedata
```

Note that you do not have to write any code or submit written answers for this part.

3. **Implement finetuning (without pretraining).** <br> 