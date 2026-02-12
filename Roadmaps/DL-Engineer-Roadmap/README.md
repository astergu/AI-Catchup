- [DL Engineer Toolkit](#dl-engineer-toolkit)
  - [阶段 0（可并行）：工具与编程基础（贯穿全程）](#阶段-0可并行工具与编程基础贯穿全程)
  - [阶段 1：数学与机器学习基础（0–3 个月）](#阶段-1数学与机器学习基础03-个月)
    - [数学（非常重要）](#数学非常重要)
    - [传统机器学习](#传统机器学习)
  - [阶段 2：深度学习核心（3–6 个月）](#阶段-2深度学习核心36-个月)
    - [神经网络基础](#神经网络基础)
    - [CNN / RNN / Attention](#cnn--rnn--attention)
  - [阶段 3：训练技巧 \& Debug 能力（6–9 个月）](#阶段-3训练技巧--debug-能力69-个月)
  - [阶段 4：高级模型与方向深化（9–15 个月）](#阶段-4高级模型与方向深化915-个月)
    - [方向 A：序列建模 \& 推荐系统（你非常适合）](#方向-a序列建模--推荐系统你非常适合)
    - [方向 B：生成模型](#方向-b生成模型)
    - [方向 C：强化学习（DeepMind 核心）](#方向-c强化学习deepmind-核心)
  - [阶段 5：科研能力（12–18 个月）](#阶段-5科研能力1218-个月)
- [DeepMind风格研究题](#deepmind风格研究题)
  - [题目 1（核心推荐）：多序列用户行为的因果表征学习](#题目-1核心推荐多序列用户行为的因果表征学习)
    - [背景（DeepMind 风格）](#背景deepmind-风格)
    - [你需要回答的问题](#你需要回答的问题)
      - [Q1：模型设计](#q1模型设计)
      - [Q2：因果挑战](#q2因果挑战)
      - [Q3：实验设计](#q3实验设计)
    - [加分项（非常 DeepMind）](#加分项非常-deepmind)
    - [考察点](#考察点)
  - [题目 2（强化学习方向）：长时延奖励下的用户转化建模](#题目-2强化学习方向长时延奖励下的用户转化建模)
    - [背景](#背景)
    - [你需要回答的问题](#你需要回答的问题-1)
      - [Q1：状态、动作、奖励如何定义？](#q1状态动作奖励如何定义)
      - [Q2：如何处理稀疏奖励？](#q2如何处理稀疏奖励)
      - [Q3：为什么 Policy Gradient 比 Q-learning 更适合？](#q3为什么-policy-gradient-比-q-learning-更适合)
      - [Q4：是否可以用 model-based RL？](#q4是否可以用-model-based-rl)
      - [Q5：如何稳定训练？](#q5如何稳定训练)
      - [Q6：如何评价 learned policy？](#q6如何评价-learned-policy)
    - [加分项](#加分项)
    - [考察点](#考察点-1)
  - [题目 3（基础研究取向）：Attention 是否真的需要 Softmax？](#题目-3基础研究取向attention-是否真的需要-softmax)
    - [背景](#背景-1)
    - [任务](#任务)
    - [你需要做的](#你需要做的)
    - [加分项](#加分项-1)
    - [考察点](#考察点-2)

# DL Engineer Toolkit

> 🎯 总目标（12–18 个月）

到计划结束时，你应当能够：
- 独立设计和实现复杂 ML / DL 模型
- 读懂并复现 NeurIPS / ICML / ICLR 论文
- 系统性 debug 训练问题（NaN、梯度爆炸、收敛失败）
- 具备申请 DeepMind / OpenAI / FAIR / 顶级 PhD 或研究型岗位的能力

## 阶段 0（可并行）：工具与编程基础（贯穿全程）

必须掌握

- Python（NumPy / SciPy / Pandas）
- 面向对象设计（尤其是 DL module 设计）
- Linux / shell / git
- GPU 基础（显存、batch size、OOM）

🎯 产出标准
- 能写结构清晰、可复用的 ML 模块
- 能 debug shape / dtype / device 问题

## 阶段 1：数学与机器学习基础（0–3 个月）

### 数学（非常重要）

你需要 理解，而不是死记。

必须掌握

- 线性代数
  - 向量空间、矩阵乘法、特征值/特征向量
  - 投影、正交、SVD
- 概率与统计
  - 随机变量、期望、方差
  - MLE / MAP
  - KL divergence、Entropy
- 优化
  - 梯度下降
  - 凸/非凸优化
  - 学习率影响

推荐：

- Mathematics for Machine Learning
- MIT OCW Linear Algebra

🎯 产出标准

- 能从公式推导出 loss 和 gradient
- 能解释为什么某个优化不收敛

### 传统机器学习

必须掌握

- [Linear](ML-Code/src/regression.py) / [Logistic](ML-Code/src/classification.py) Regression
- SVM（margin、kernel）
- Decision Tree / Random Forest / GBDT
- KNN
- K-means
- **Bias–Variance tradeoff**
  - Metrics: train set error vs. validation set error
    - low training error, large validation error (overfit): high variance
    - large training error, large validation error (underfit): high bias
    - large training error, larger validation error (not enough data): high bias, high variance
    - low training error, low validation error (just right): low bias, low variance
  - Recipe
    - High bias: bigger network, train longer
    - High variance: more data, regularization
- **Regularization**
  - L1 / L2
    - L1: sparsity (meaning a lot of zeros)
    - L2: smoothness (also called weight decay)
  - Dropout
  - Early stopping
- **Feature engineering**
  - 

📘 推荐：

- [Andrew Ng ML](https://www.coursera.org/specializations/machine-learning-introduction)
  - [Youtube video playlist](https://www.youtube.com/watch?v=vStJoetOxJg&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI)
- [Elements of Statistical Learning（Hastie）](https://hastie.su.domains/ElemStatLearn/)

🎯 产出标准

- 能解释为什么某模型 overfit
- 能手写 loss + gradient

## 阶段 2：深度学习核心（3–6 个月）

### 神经网络基础

必须掌握

- Backpropagation
- Activation（ReLU / GELU / SiLU）
- Initialization（Xavier / He）
- Normalization（BatchNorm / LayerNorm）

🎯 产出标准

- 能从头写一个 MLP（不用 Keras Sequential）
- 能解释梯度消失 / 爆炸

### CNN / RNN / Attention

- CNN
  - Convolution、Pooling、Receptive Field

- RNN
  - RNN / LSTM / GRU
  - Sequence masking
  - Truncated BPTT

- Attention & Transformer（重点）
  - Scaled dot-product attention
  - Multi-head attention
  - Positional encoding
  - Encoder / Decoder

推荐：

- [CS231n](https://cs231n.stanford.edu/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Jay Alammar 博客](https://jalammar.github.io/)

🎯 产出标准

- 能手写 Transformer Encoder
- 能正确处理 mask
- 能理解为什么 attention 比 RNN 好

## 阶段 3：训练技巧 & Debug 能力（6–9 个月）

这是你之前问的 NaN、batch size、mask、sequence 的根本能力来源。

必须掌握

- Loss scale / mixed precision
- Gradient clipping
- Learning rate schedule
- Batch size vs convergence
- NaN / Inf debug
- 数值稳定性（log-sum-exp）

🎯 产出标准

- 能快速定位 NaN 来源
- 能解释 batch size 改变为什么影响 loss

## 阶段 4：高级模型与方向深化（9–15 个月）

你可以选 1–2 个方向深入（建议选与你现在工作相关的）：

### 方向 A：序列建模 & 推荐系统（你非常适合）

- DIN / DIEN
- Transformer4Rec
- Self-attention vs Cross-attention
- Pooling / attention pooling
- Long sequence modeling

🎯 项目示例：

- 用户事件序列 → 转化预测模型
- 多序列 + cross attention

### 方向 B：生成模型

- AutoEncoder / VAE
- Diffusion models
- Representation learning

### 方向 C：强化学习（DeepMind 核心）

- Q-learning
- Policy Gradient
- PPO / SAC
- Model-based RL

📘 推荐：

- Sutton & Barto
- Spinning Up in Deep RL

## 阶段 5：科研能力（12–18 个月）

必须掌握

- 读论文（问题 → 方法 → 实验）
- 做消融实验
- 写 research-style 代码
- 结果复现

🎯 产出

- GitHub repo（复现 + 改进）
- arXiv / workshop 论文（哪怕是小改进）

# DeepMind风格研究题

## 题目 1（核心推荐）：多序列用户行为的因果表征学习

### 背景（DeepMind 风格）

你有用户的多种行为序列（app install、purchase、view、click），目标是预测 是否发生转化。

现实问题：
- 行为序列长度不一致
- 不同行为对转化的因果影响不同
- 简单 attention 会“记住相关性”，而不是因果性

🎯 任务
- 设计一个模型，使其能够：
    - 利用多种事件序列
    - 学到 对转化真正有因果贡献的表示
    - 在存在“虚假相关”的情况下仍保持稳定预测

### 你需要回答的问题

（这正是 DeepMind 面试会问的）

#### Q1：模型设计

你会如何建模多序列？

self-attention vs cross-attention 如何选择？

是否需要共享 embedding？

#### Q2：因果挑战

哪些行为只是“相关但非因果”？

如何设计模型或训练目标减少 spurious correlation？

#### Q3：实验设计

你如何验证模型学到了因果，而不是相关？

如果 offline metrics 提升但 online 下降，你如何解释？


### 加分项（非常 DeepMind）

- 引入 counterfactual masking
- 使用 temporal intervention（打乱时间顺序）
- 引入辅助 loss（如 predicting masked event）

### 考察点

- 表示学习
- 因果推理直觉
- attention 机制理解
- 实验设计能力

## 题目 2（强化学习方向）：长时延奖励下的用户转化建模

### 背景

转化可能发生在用户行为序列结束 几小时甚至几天之后。

传统 supervised learning：
- 只看是否转化
- 忽略行为与奖励的时间结构

🎯 任务
- 将转化预测问题重新建模为 强化学习问题。

### 你需要回答的问题

#### Q1：状态、动作、奖励如何定义？

#### Q2：如何处理稀疏奖励？

#### Q3：为什么 Policy Gradient 比 Q-learning 更适合？

#### Q4：是否可以用 model-based RL？

#### Q5：如何稳定训练？

#### Q6：如何评价 learned policy？


### 加分项

- Credit assignment（TD(λ)）
- Use hindsight replay
- Offline RL

### 考察点

- RL 建模能力
- 抽象现实问题能力
- 理论与工程结合

## 题目 3（基础研究取向）：Attention 是否真的需要 Softmax？

### 背景

Transformer 的核心是：

Attention(Q,K,V) = softmax(QK^T)V

但：

- softmax 带来数值不稳定
- attention 分布过于平滑

### 任务

研究并回答：

- 是否存在不使用 softmax 的 attention 机制？
- 如果存在，它有什么优点？

### 你需要做的

- 提出一种替代机制（如 linear attention）
- 分析理论性质（归一化、稳定性）
- 实验验证性能

### 加分项

- 分析复杂度 O(N²) → O(N)
- 长序列建模实验
- 对比收敛速度

### 考察点

- 数学直觉
- 算法创新能力
- 理论 + 实验结合