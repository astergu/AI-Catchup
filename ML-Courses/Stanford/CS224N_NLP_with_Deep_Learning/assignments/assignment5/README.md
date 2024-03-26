# CS224n Assignment #5: Self-Attention, Transformers, and Pretraining

This assignment is an investigation into Transformer self-attention building blocks, and the effects of pre-training. It covers mathematical properties of Transformers and self-attention through written questions. Further, you'll get experience with practical system-building through repurposing an existing codebase.

## Attention Exploration

Multi-head self-attention is the core modeling component of Transformers. 

1. **Copying in attention**. One advantage of attention is that it's particularly easy to "copy" a value vector to the output *c*. In this problem, we'll motivate why this is the case.
   1. **Explain** why $\alpha$ can be interpreted as a categorical probability distribution.
   2. The distribution $\alpha$ is typically relatively "diffuse"; the probability mass is spread out between many different $\alpha_i$. However, this is not always the case. **Describe** (in onr sentence) under what conditions the categorical distribution $\alpha$ puts almost all of its weight on some $\alpha_j$, where $j \in {1,...,n}$ (i.e. $\alpha_j \gg \sum_{i\neq j} \alpha_i$). What must be true about the query *q* and/or the keys ${k_1,...,k_n}$?
   3. Under the conditions you gave in (ii), **describe** the output *c*.
   4. **Explain** (in two sentences or fewer) what your answer to (ii) and (iii) means intuitively.
2. **An average of two**. Instead of focusing on just one vector $v_j$, a Transformer model might want to incorporate information from *multiple* source vectors. Consider the case where we instead want to incorporate information from **two** vectors $v_a$ and $v_b$, with corresponding key vectors $k_a$ and $k_b$.
   1. How should we combine two $d$-dimensional vectors $v_a$, $v_b$ into one output vector $c$ in a way that preserves information from both vectors? In machine learning, one common way to do so is to take the average: $c=\frac{1}{2}(v_a+v_b)$. It might seem hard to extract information about the original vectors $v_a$ and $v_b$ from the resulting $c$, but under certain conditions one can do so. <br> Suppose that although we don't know $v_a$ or $v_b$, we do know that $v_a$ lies in a subspace $A$ formed by the $m$ basis vectors $\{a_1,a_2,...,a_m\}$, while $v_b$ lies in a subspace $B$ formed by the $p$ basis vectors $\{b_1,b_2,...,b_p\}$. <br> Using the basis vectors $\{a_1,a_2,...,a_m\}$, construct a matrix $M$ such that for arbitrary vectors $v_a\in A$ and $v_b \in B$, we can use $M$ to extract $v_a$ from the sum vector $s=v_a+v_b$. In other words, we want to construct $M$ such that for any $v_a$, $v_b$, $M_s=v_a$ holds for your $M$.


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