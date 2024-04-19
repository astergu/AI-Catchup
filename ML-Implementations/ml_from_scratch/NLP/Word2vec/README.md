![word2vec diagram](./image/skip_gram_net_arch.png)

> The input to this network is a one-hot vector representing the input word, and the label is also a one-hot vector representing the target word, however, the network’s output is a probability distribution of target words, not necessarily a one-hot vector like the labels.

The rows of the hidden layer weight matrix, are actually the word vectors (word embeddings) we want.

Here is a high-level illustration of the architecture:

![output weight matrix](./image/output_weights_function.png)

> How the negative samples are chosen?

The negative samples are chosen using a unigram distribution.

Essentially, the probability for selecting a word as a negative sample is related to its frequency, with more frequent words being more likely to be selected as negative samples.

Specifically, each word is given a weight equal to it’s frequency (word count) raised to the 3/4 power. The probability for a selecting a word is just it’s weight divided by the sum of weights for all words.

> Skip-gram vs. CBOW

In models using large corpora and a high number of dimensions, the skip-gram model yields the highest overall accuracy, and consistently produces the highest accuracy on semantic relationships, as well as yielding the highest syntactic accuracy in most cases. However, the CBOW is less computationally expensive and yields similar accuracy results.

> Sub-sampling

Some frequent words often provide little information. Words with frequency above a certain threshold (e.g ‘a’, ‘an’ and ‘that’) may be subsampled to increase training speed and performance. Also, common word pairs or phrases may be treated as single “words” to increase training speed.

> Context window

The size of the context window determines how many words before and after a given word would be included as context words of the given word. According to the authors’ note, the recommended value is 10 for skip-gram and 5 for CBOW.


# References

- [The Illustrated Word2Vec](http://jalammar.github.io/illustrated-word2vec/)
- [Word2Vec Explained](https://israelg99.github.io/2017-03-23-Word2Vec-Explained/)
- [Word2vec with PyTorch: Implementing the Original Paper](https://towardsdatascience.com/word2vec-with-pytorch-implementing-original-paper-2cd7040120b0)