# CS224N: Natural Language Processing with Deep Learning

[Course Homepage](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/) [[course videos]](https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)

**Scope**

- The foundations of the effective modern methods for deep learning applied to NLP
  - Basics first, then key methods used in NLP in 2023: `Word vectors`, `feed-forward networks`, `recurrent networks`, `attention`, `encoder-decoder models`, `transformers`, `large pre-trained language models`, etc.
- A big picture understanding of human languages and the difficulties in understanding and producing them via computers
- An understanding of and **ability to build systems** (in Pytorch) for some of the major problems in NLP
  - Word meaning, dependency parsing, machine translation, question answering

<br>

**A few uses of NLP**

- *Machine Translation*.  
  - Perhaps one of the earliest and most successful applications and driving uses of natural language processing.
- *Question answering and information retrieval*.  
  - In NLP, question answering has tended to be related to information-seeking questions. 
- *Summarization and analysis of text*.  
  - There are myriad reasons to want to understand (1) what people are talking about and (2) what they think about those things. 
- *speech(or sign)-to-text*.  
  - The process of automatic transcription of spoken or signed language (audio or video) to textual representations is a massive and useful application.

<br>

**High-Level Plan for Assignments**

- Ass1 is hopefully an easy on ramp - A Jupyter/IPython Notebook
- Ass2 is pure Python (numpy) but expects you to do (multivariate) calculus, so you really understand the basics
- Ass3 introduces PyTorch, building a feed-forward network for dependency parsing
- Ass4 and Ass5 use PyTorch on a GPU (Microsoft Azure)
- For Final Project
  - Do the default project, which is a question answering system
  - Propose a custom final project

<br> 

Lecture | Topics | Course Materials | Assignments |
| --------- | --------- | --------- | --------- |
| **Lecture 1** | **Word Vectors** <br> [[slides]](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture01-wordvecs1.pdf) [[notes]](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/readings/cs224n_winter2023_lecture1_notes_draft.pdf)  <br><br>  **Gensim word vectors** <br> [[code]](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/materials/Gensim%20word%20vector%20visualization.html) | Suggested Readings: <br> 1. [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781.pdf) (original word2vec paper) <br> 2. [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) (negative sampling paper) | Assignment 1 <br> [[code]](./assignments/assignment1/exploring_word_vectors_22_23.ipynb) |
| **Lecture 2** | **Word Vectors, Word Window CLassification, Language Models** <br> [[slides]](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture02-wordvecs2.pdf) [[notes]](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/readings/cs224n-2019-notes02-wordvecs2.pdf)  | Suggested Readings: <br> 1. [GloVe: Global Vectors for Word Representation](http://nlp.stanford.edu/pubs/glove.pdf) (original GloVe paper) <br> 2. [Improving Distributional Similarity with Lessons Learned from Word Embeddings](http://www.aclweb.org/anthology/Q15-1016) <br> 3. [Evaluation methods for unsupervised word embeddings](http://www.aclweb.org/anthology/D15-1036) <br><br> Additional Readings: <br> 1. [A Latent Variable Model Approach to PMI-based Word Embeddings](http://aclweb.org/anthology/Q16-1028) <br> 2. [Linear Algebraic Structure of Word Senses, with Applications to Polysemy](https://transacl.org/ojs/index.php/tacl/article/viewFile/1346/320) <br> [On the Dimensionality of Word Embedding](https://papers.nips.cc/paper/7368-on-the-dimensionality-of-word-embedding.pdf)  |  |
| **Lecture 3** | **Backprop and Neural Networks** <br> [[slides]](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture03-neuralnets.pdf) [[notes]](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/readings/cs224n-2019-notes03-neuralnets.pdf) | Suggested Readings: <br> 1. [matrix calculus notes](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/readings/gradient-notes.pdf) <br> 2. [Review of differential calculus](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/readings/review-differential-calculus.pdf) <br> 3. [CS231n notes on network architectures](http://cs231n.github.io/neural-networks-1/) :thumbsup: <br> 4. [CS231n notes on backprop](http://cs231n.github.io/optimization-2/) :thumbsup: <br> 5. [Derivatives, Backpropagation, and Vectorization](http://cs231n.stanford.edu/handouts/derivatives.pdf) <br> 6. [Learning Representations by Backpropagating Errors](http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf) (seminal Rumelhart et al. backpropagation paper) <br><br> Additional Readings: <br> 1. [Yes you should understand backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b) <br> 2. [Natural Language Processing (Almost) from Scratch](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf) | Assignment 2 <br> [[code]](./assignments/assignment2/) [[handout]](./assignments/assignment2/assignment2.pdf) |
| **Lecture 4** | **Dependency Parsing** <br> [[slides](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture04-dep-parsing.pdf)] [[notes]](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/readings/cs224n-2019-notes04-dependencyparsing.pdf) <br> [[slides (annotated)]](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2021-lecture04-dep-parsing-annotated.pdf) | Suggested Readings: <br> 1. [Incrementality in Deterministic Dependency Parsing](https://www.aclweb.org/anthology/W/W04/W04-0308.pdf) <br> 2. [A Fast and Accurate Dependency Parser using Neural Networks](https://www.emnlp2014.org/papers/pdf/EMNLP2014082.pdf) <br> 3. [Dependency Parsing](http://www.morganclaypool.com/doi/abs/10.2200/S00169ED1V01Y200901HLT002) <br> 4. [Globally Normalized Transition-Based Neural Networks](https://arxiv.org/pdf/1603.06042.pdf) <br> 5. [Universal Stanford Dependencies: A cross-linguistic typology](http://nlp.stanford.edu/~manning/papers/USD_LREC14_UD_revision.pdf) <br> 6. [Universal Dependencies website](http://universaldependencies.org/) <br> 7. [Jurafsky & Martin Chapter 14](https://web.stanford.edu/~jurafsky/slp3/14.pdf)|  | 
| **Lecture 5** | **Recurrent Neural Netowkrs and Language Models** <br> [[slides]](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture05-rnnlm.pdf) [[notes]](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/readings/cs224n-2019-notes05-LM_RNN.pdf) | Suggested Readings: <br> 1. [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf) (textbook chapter) <br> 2. [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) (blog post overview) <br> 3. [Sequence Modeling: Recurrent and Recursive Neural Nets](http://www.deeplearningbook.org/contents/rnn.html) (Sections 10.1 and 10.2) <br> 4. [On Chomsky and the Two Cultures of Statistical Learning](http://norvig.com/chomsky.html) <br> 5. [Sequence Modeling: Recurrent and Recursive Neural Nets](http://www.deeplearningbook.org/contents/rnn.html) (Sections 10.3, 10.5, 10.7-10.12) <br>  6. [Learning long-term dependencies with gradient descent is difficult](http://ai.dinfo.unifi.it/paolo//ps/tnn-94-gradient.pdf) (one of the original vanishing gradient papers) <br> 7. [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/pdf/1211.5063.pdf)(proof of vanishing gradient problem) <br> 8. [Vanishing Gradients Jupyter Notebook](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/lectures/vanishing_grad_example.html) (demo for feedforward networks) <br> 9. [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (blog post overview) | Assignment 3 <br> [[code]](./assignments/assignment3/) [[handout]](./assignments/assignment3/a3_handout.pdf) |

## Lecture 1: Introduction and Word Vectors

### Representating words by their context

- **Distributional semantics**: A word's meaning is given by the words that frequently appear close-by

## Word Vectors

We will build a dense vector for each word, chosen so that it is aimilar to vectors of words that appear in similar contexts.

> `word vectors` are also called `word embeddings` or `word representations`. They are a `distributed` representations.

## Word2vec

`Word2vec` (Mikolov et al. 2013) is a framework for learning word vectors.

### Idea

- We have a large corpus ("body") of text
- Every word in a fixed vocabulary is represented by a `vector`
- Go through each position `t` in the text, which as a center word `c` and context ("outside") word `o`
- Use the `simiilarity of the word vectors` for `c` and `o` to `calculate the probabilty` of `o` given `c` (or vice versa)
- `Keep adjusting the word vectors` to maximize this probability

## Lecture 2: Word Vectors, Word Senses, and Neural Network Classifiers

Two model variants:

1. **Skip-gram (SG)**
   Predict context ("outside") words (position independent) given center word
2. **Continuous Bas of Words (CBOW)**
   Predict center word from (bag of) context words

Additional efficiency in training: **Negative sampling**


## Lecture 3: Backpropagation

## Lecture 4: Syntactic Structure and Dependency Parsing

Dependency syntax postulates that syntactic structure consists of relations between lexical items, normally binary asymmetric relations called `dependencies`. An arrow connects a `head` (governor, superior, regent) with a `dependent` (modifier, inferior, subordinate). Usually, dependencies form a tree (a connected, acyclic, single-root graph).

### Dependency Parsing

A sentence is parsed by choosing for each word what other word (including ROOT) it is a dependent of.

Usually some constructs:
- Only one word is a dependent of ROOT
- Don't want cycles A->B, B->A

Final issue is whether arrows can cross (be non-projective) or not.

### Methods of Dependency Parsing

1. Dynamic programming
2. Graph algorithms
3. Constraint Satisfaction
4. Transition-based parsing or deterministic dependency parsing

### A neural dependency parse [Chen and Manning 2014]

- **[First Win]**: Distributed Representations
  - Represent each word as a `d`-dimensional dense vector (i.e., word embedding)
  - Meanwhile, `part-of-speach tags (POS)` and `dependency labels` are also represented as `d`-dimensional vectors.
-  **[Second win]**: Deep learning classifiers are non-linear classifiers
   -  A `softmax classifier` assigns classes $y \in C$ based on inputs $x \in \mathbb{R}^d$ via the probability: $p(y|x)=\frac{exp(w_yx)}{\sum exp(w_cx)}$. We train the weight matrix $W \in \mathbb{R}^{c\times d}$ to minimize the neg.log loss: $\sum_i{-logp(y_i|x_i)}$
   -  Traditional ML classifiers (including Na$\dot{i}$ve Bayes, SVMs, logistic regression and softmax classifier) are not very powerful classifiers, they only give linear decision boundaries.
   -  Neural networks can learn much more complex functions with nonlinear decision boundaries.