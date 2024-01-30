> Before you start, make sure you read the `README.txt` in the same directory as this notebook for important setup information.

# Part 1: Count-Based Word Vectors

## Question 1.1: Implement `distinct_words`

Write a method to work out the distinct words (word types) that occur in the corpus. 

```python
def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): sorted list of distinct words across the corpus
            n_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    n_corpus_words = -1

    corpus_words = sorted(set([w for c in corpus for w in c]))
    n_corpus_words = len(corpus_words)

    return corpus_words, n_corpus_words
```

## Question 1.2: Implement `computer_co_occurence_matrix`

Write a method that constructs a co-occurrence matrix for a certain window-size (with a default of 4), considering words before and after the word in the center of the window.

```python
def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.
              
              For example, if we take the document "<START> All that glitters is not gold <END>" with window size of 4,
              "All" will co-occur with "<START>", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (a symmetric numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, n_words = distinct_words(corpus)
    M = None
    word2ind = {}

    word2ind = dict(zip(words, range(n_words)))  
    M = np.zeros((n_words, n_words))

    for doc in corpus:
        for i in range(len(doc)): # current word
            for j in range(max(i - window_size, 0), min(i + window_size + 1, len(doc))): # context word
                if j == i:
                    continue
                M[word2ind[doc[i]]][word2ind[doc[j]]] += 1

    return M, word2ind
```

## Question 1.3: Implement `reduce_to_k_dim`

Construct a method that performs dimensionality reduction on the matrix to produce k-dimensional embeddings. Use SVD to take the top k components and produce a new matrix of k-dimensional embeddings.

**Note**: All of numpy, scipy, and scikit-learn (sklearn) provide some implementation of SVD, but only scipy and sklearn provide an implementation of Truncated SVD, and only sklearn provides an efficient randomized algorithm for calculating large-scale Truncated SVD. So please use [sklearn.decomposition.TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).

```python
def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
    
        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """    
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    
    svd = TruncatedSVD(n_components=k)
    svd.fit(M)
    M_reduced = svd.transform(M)

    print("Done.")
    return M_reduced
```

## Question 1.4: Implement `plot_embeddings`

Here you will write a function to plot a set of 2D vectors in 2D space. For graphs, we will use Matplotlib (plt).

```python
def plot_embeddings(M_reduced, word2ind, words):
    """ Plot in a scatterplot the embeddings of the words specified in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2ind.
        Include a label next to each point.
        
        Params:
            M_reduced (numpy matrix of shape (number of unique words in the corpus , 2)): matrix of 2-dimensioal word embeddings
            word2ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to visualize
    """
    word_embeddings = [M_reduced[word2ind[w]] for w in words]
    for word, emb in zip(words, word_embeddings):
        plt.scatter(emb[0], emb[1], marker='x', color='red')
        plt.text(emb[0], emb[1], word, fontsize=8)

    plt.show()
```

## Question 1.5: Co-Occurrence Plot Analysis

We will compute the co-occurrence matrix with fixed window of 4 (the default window size), over the Reuters "gold" corpus. Then we will use TruncatedSVD to compute 2-dimensional embeddings of each word. TruncatedSVD returns U*S, so we need to normalize the returned vectors, so that all the vectors will appear around the unit circle (therefore closeness is directional closeness).

```python
reuters_corpus = read_corpus()
M_co_occurrence, word2ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)

# Rescale (normalize) the rows to make them each of unit-length
M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting

words = ['value', 'gold', 'platinum', 'reserves', 'silver', 'metals', 'copper', 'belgium', 'australia', 'china', 'grammes', "mine"]

plot_embeddings(M_normalized, word2ind_co_occurrence, words)
```

### <font color='blue'> a. Find at least groups of words that cluster together in 2-dimensional embedding space. Given an explanation for each cluster you observe. </font>

`australia` and `belgium` are in the same group, because they are both countries, so they tend to occur together in news. `gold` and `mine` are in the sample group, because they are co-related and tends to appear in same sentences. 

### <font color='blue'> b. What doesn't cluster together you might think should have? Describe at least two examples. </font>

`china` as a country, doesn't cluster together with other countries like `australia` and `belgium`. `silver` as a metal, doesn't cluster together with other metals.


# Part 2: Prediction-Based Word Vectors

More recently prediction-based word vectors have demonstrated better performance, such as word2vec and GloVe (which also utilizes the benefit of counts). Here, we shall explore the embeddings produced by GloVe.

## Question 2.1: Glove Plot Analysis


### <font color='blue'> a. What is one way the plot is different from the one generated earlier from the co-occurrence matrix? What is one way it's similar? </font>

In this plot, `china` is close to `reserves`. However, `grammes` is far from other nodes as before.


### <font color='blue'> b. What is a possible cause for the difference? </font>

In news, `reserves` may not occur together frequently with `china`, but may appear immediately after (around) `china`. 

## Question 2.2: Words with Multiple Meanings

Polysemes and homonyms are words that have more than one meaning (see this wiki page to learn more about the difference between polysemes and homonyms ). Find a word with at least two different meanings such that the top-10 most similar words (according to cosine similarity) contain related words from both meanings. 


```python
wv_from_bin.most_similar("good")
```

> [('better', 0.8141133785247803),
 ('really', 0.8016481995582581),
 ('always', 0.7913187146186829),
 ('sure', 0.7792829871177673),
 ('you', 0.7747212052345276),
 ('very', 0.7718809843063354),
 ('things', 0.7658981084823608),
 ('well', 0.7652329802513123),
 ('think', 0.7623050808906555),
 ('we', 0.7586264610290527)]

### <font color='blue'> Please state the word you discover and the multiple meanings that occur in the top 10. Why do you think many of the polysemous or homonymic words you tried didn't work (i.e. the top-10 most similar words only contain one of the meanings of the words)? </font>

`similar` here doesn't mean polysemes or homonyms, just mean the words tend to appear in similar contexts.

## Question 2.3: Synonyms & Antonyms

You should use the the wv_from_bin.distance(w1, w2) function here in order to compute the cosine distance between two words. Please give a possible explanation for why this counter-intuitive result may have happened.

```python
w1 = "good" 
w2 = "bad"
w3 = "wonderful"
w1_w2_dist = wv_from_bin.distance(w1, w2)
w1_w3_dist = wv_from_bin.distance(w1, w3)

print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))
```

### <font color='blue'>  </font>