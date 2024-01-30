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

### <font color='red' size='3'> a. Find at least groups of words that cluster together in 2-dimensional embedding space. Given an explanation for each cluster you observe. </font>

`australia` and `belgium` are in the same group, because they are both countries, so they tend to occur together in news. `gold` and `mine` are in the sample group, because they are co-related and tends to appear in same sentences. 

### <font color='red' size='3'> b. What doesn't cluster together you might think should have? Describe at least two examples. </font>

`china` as a country, doesn't cluster together with other countries like `australia` and `belgium`. `silver` as a metal, doesn't cluster together with other metals.


# Part 2: Prediction-Based Word Vectors

More recently prediction-based word vectors have demonstrated better performance, such as word2vec and GloVe (which also utilizes the benefit of counts). Here, we shall explore the embeddings produced by GloVe.

## Question 2.1: Glove Plot Analysis


### <font color='red' size='3'> a. What is one way the plot is different from the one generated earlier from the co-occurrence matrix? What is one way it's similar? </font>

In this plot, `china` is close to `reserves`. However, `grammes` is far from other nodes as before.


### <font color='red' size='3'> b. What is a possible cause for the difference? </font>

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

### <font color='red' size='3'> Please state the word you discover and the multiple meanings that occur in the top 10. Why do you think many of the polysemous or homonymic words you tried didn't work (i.e. the top-10 most similar words only contain one of the meanings of the words)? </font>

`similar` here doesn't mean polysemes or homonyms, just mean the words tend to appear in similar contexts.

## Question 2.3: Synonyms & Antonyms

You should use the the `wv_from_bin.distance(w1, w2)` function here in order to compute the cosine distance between two words. 

```python
w1 = "good" 
w2 = "bad"
w3 = "better"
w1_w2_dist = wv_from_bin.distance(w1, w2)
w1_w3_dist = wv_from_bin.distance(w1, w3)

print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))
```

> Synonyms good, bad have cosine distance: 0.28903740644454956
> Antonyms good, better have cosine distance: 0.18588662147521973

### <font color='red' size='3'> Please give a possible explanation for why this counter-intuitive result may have happened. </font>

`good` and `bad` are both adjectives that can be used to describe situations in news, but `better` may be used in different comparative contexts. 

## Question 2.4: Analogies with Word Vectors

```python
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'grandfather'], negative=['man']))
```

> [('grandmother', 0.7608445286750793),
 ('granddaughter', 0.7200808525085449),
 ('daughter', 0.7168302536010742),
 ('mother', 0.7151536345481873),
 ('niece', 0.7005682587623596),
 ('father', 0.6659888029098511),
 ('aunt', 0.6623408794403076),
 ('grandson', 0.6618767380714417),
 ('grandparents', 0.6446609497070312),
 ('wife', 0.6445354223251343)]

### <font color='red' size='3'> Using only vectors `m`, `g`, `w`, and the vector arithmetic operators `+` and `-` in your answer, to what expression are we maximizing `x`'s cosine similarity? </font>

`man` + `grandfather` - `woman`

## Question 2.5: Finding Analogies

### <font color='red' size='3'> a. For the previous example, it's clear that "grandmother" completes the analogy. But give an intuitive explanation as to why the most_similar function gives us words like "granddaughter", "daughter", or "mother? </font>



### <font color='red' size='3'> b. Find an example of analogy that holds according to these vectors (i.e. the intended word is ranked top). In your solution please state the full analogy in the form x:y :: a:b. If you believe the analogy is complicated, explain why the analogy holds in one or two sentences. </font>

```python
x, y, a, b = ["china", "chinese", "america", "american"]
assert wv_from_bin.most_similar(positive=[a, y], negative=[x])[0][0] == b
```

## Question 2.6: Incorrect Analogy

### <font color='red' size='3'> a. Below, we expect to see the intended analogy "hand : glove :: foot : sock", but we see an unexpected result instead. Give a potential reason as to why this particular analogy turned out the way it did? </font>

```python
pprint.pprint(wv_from_bin.most_similar(positive=['foot', 'glove'], negative=['hand']))
```

> [('45,000-square', 0.4922032058238983),
 ('15,000-square', 0.4649604558944702),
 ('10,000-square', 0.45447564125061035),
 ('6,000-square', 0.44975772500038147),
 ('3,500-square', 0.4441334009170532),
 ('700-square', 0.44257497787475586),
 ('50,000-square', 0.43563973903656006),
 ('3,000-square', 0.43486514687538147),
 ('30,000-square', 0.4330596923828125),
 ('footed', 0.43236875534057617)]


In news, `foot` is more frequently talked as the meaning of footage in real estate, rather than `foot` that humans have.

### <font color='red' size='3'> b. Find another example of analogy that does not hold according to these vectors. In your solution, state the intended analogy in the form x:y :: a:b, and state the incorrect value of b according to the word vectors (in the previous example, this would be '45,000-square').  </font>

```python
x, y, a, b = ["square", "four", "triangle", "three"]
pprint.pprint(wv_from_bin.most_similar(positive=[a, y], negative=[x]))
```

## Question 2.7: Guided Analysis of Bias in Word Vectors

It's important to be cognizant of the biases (gender, race, sexual orientation etc.) implicit in our word embeddings. Bias can be dangerous because it can reinforce stereotypes through applications that employ these models.

```python
pprint.pprint(wv_from_bin.most_similar(positive=['man', 'profession'], negative=['woman']))
print()
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'profession'], negative=['man']))
```

### <font color='red' size='3'> Point out the difference between the list of female-associated words and the list of male-associated words, and explain how it is reflecting gender bias. </font>

Obviously, man's professions are more related to business, however, women's professions are related to service roles, such as nursing, teacher, etc.


## Question 2.8: Independent Analysis of Bias in Word Vectors

Use the `most_similar` function to find another pair of analogies that demonstrates some bias is exhibited by the vectors. Please briefly explain the example of bias that you discover.

```python
A = "young"
B = "old"
word = "best"
pprint.pprint(wv_from_bin.most_similar(positive=[A, word], negative=[B]))
print()
pprint.pprint(wv_from_bin.most_similar(positive=[B, word], negative=[A]))
```

> [('talented', 0.5989054441452026),
 ('talent', 0.5488317012786865),
 ('award', 0.5478774309158325),
 ('actors', 0.5343650579452515),
 ('good', 0.533554196357727),
 ('success', 0.5324625968933105),
 ('well', 0.5321840047836304),
 ('better', 0.5309637188911438),
 ('most', 0.5245237350463867),
 ('performers', 0.5155372023582458)]

 > [('history', 0.5702727437019348),
 ('record', 0.5318861603736877),
 ('one', 0.5158449411392212),
 ('time', 0.5126581192016602),
 ('winning', 0.5001153349876404),
 ('35-year', 0.4995863735675812),
 ('same', 0.4874439835548401),
 ('only', 0.48707669973373413),
 ('previous', 0.4835243821144104),
 ('good', 0.48213526606559753)]


## Question 2.9: Thinking About Bias

### <font color='red' size='3'> a. Give one explanation of how bias gets into the word vectors. Briefly describe a real-world example that demonstrates this source of bias. </font>


### <font color='red' size='3'> b. What is one method you can use to mitigate bias exhibited by word vectors? Briefly describe a real-world example that demonstrates this method. </font>
