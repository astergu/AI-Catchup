- [论文阅读](#论文阅读)
  - [Attention is All You Need 论文阅读](#attention-is-all-you-need-论文阅读)
    - [主要工作](#主要工作)
    - [模型架构](#模型架构)
      - [Attention 注意力](#attention-注意力)
        - [Scaled Dot-Product Attention](#scaled-dot-product-attention)
        - [Multi-head Attention 多头注意力](#multi-head-attention-多头注意力)
      - [Layer Norm](#layer-norm)
    - [实验结果](#实验结果)
    - [参考资料](#参考资料)


# 论文阅读
## Attention is All You Need 论文阅读

### 主要工作

- 更简单：提出仅采用注意力机制，而摒弃RNN卷积的Transformer架构
- 更高效：Transformer架构可以并行训练


### 模型架构

![Transformer architecture](./transformer_architecture.png)

模型为Encoder-Decoder架构。

- Encoder：6个相同的模型块。
  - 每一块包含两个子层。
    - Multi-head self-attention
    - Position-wise fully conntected feed-forward network　
  - 对每一个子层，进行残差连接（residual connection），紧接着做layer normalization，即每一个子层的输出为`LayerNorm(x+Sublayer(x))`，每一层的输出维度$d_{model}=512$。
- Decoder：6个相同的模型块。
  - 每一块包含三个子层。
    - Masked Multi-head attention
    - Multi-head self-attention
    - Position-wise fully conntected feed-forward network
  - 每一个子层之后的残差连接和layer normalization与Encoder相同。　


#### Attention 注意力

注意力函数将一个query和一组key-value对映射到一个输出，最终的输出可以被看成是value的加权和，而权重则来自于query和相应key的相似度（compatibility function）。

##### Scaled Dot-Product Attention 

输入由相同维度$d_k$的queries和keys，和维度为$d_v$的values组成，通过计算query和所有keys的电机，再除以$\sqrt{d_k}$，接着通过一个softmax函数得到每个value的权重。

$Attention(Q,K,V)=softmax(\frac{QK^{\intercal}}{\sqrt{d_k}})V$

##### Multi-head Attention 多头注意力 

由于一个attention函数只是value值的加权和，可学习的参数不多，因此引入多头注意力机制，模拟卷积神经网络多输出通道的效果。实现方法，是将queries，keys和values投影到低维的空间$h$次。每一个投影的输出合并到一起，再投影得到最终的输出。

$MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O$, where $head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)$

#### Layer Norm 

在序列模型里，每个样本的长度可能会不一样。对于这种情况，batch norm算出来的均值和方差抖动可能会比较大。另外，在预测时，全局的均值方差对于新遇到的样本可能不具备参考价值。Layer norm针对样本内部进行归一化不受样本长度影响。

### 实验结果

### 参考资料

- [李沐带你读论文](https://www.youtube.com/watch?v=nzqlFIcCSWQ&list=PLFXJ6jwg0qW-7UM8iUTj3qKqdhbQULP5I&index=6)
- 李宏毅机器学习课程