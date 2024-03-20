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