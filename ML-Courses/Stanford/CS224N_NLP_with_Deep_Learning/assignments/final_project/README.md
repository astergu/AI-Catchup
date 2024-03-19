# Default Final Project: miniBERT and Downstream Tasks

## Overview

- Implement some of the most important components of the BERT model so that you can better understand its architecture.
- Examine how to find-tune BERT's contextualized embeddings to simultaneously perform well on multiple sentence-level tasks (**sentiment analysis**, **paraphrase detection**, and **semantic textual similarity**)

### Bidirectional Encoder Representations from Transformers: BERT

Bidirectional Encoder Representations from Transformers or BERT is a transformer-based model that generates contextual word representations. 

### Sentiment Analysis

A basic task in understanding a given text is classifying its polarity (i.e., whether the expressed opinion in a text is positive, negative, or neutral). Sentiment analysis can be utilized to determine individual feelings towards a particular products, politicians, or within news reports.

As a concrete dataset example, the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/treebank.html) consists of 11,855 single sentences extracted from movie reviews. The dataset was parsed with the [Stanford parser](https://nlp.stanford.edu/software/lex-parser.shtml) and includes a total of 215,154 unique phrases from those parse trees, each annotated by 3 human judges. Each phrase has a label of negative, somewhat negative, somewhat positive, or positive. 

### Paraphrase Detection

Paraphrase Detection is the task of finding paraphrases of texts in a large corpus of passages. Paraphrases are "rewordings of something written or spoken by someone else"; paraphrase detection thus essentially seeks to determine whether particular words or phrases convey the same semantic meaning. 

As a concrete dataset example, the website Quora, often receives questions that are duplicates of other questions. To better redirect users and prevent unnecessary work, Quora released a dataset that labeled whether different questions were paraphrases of each other.

### Semantic Textual Similary (STS)

The semantic textual similary (STS) task seeks to capture the notion that some texts are more similar than others. STS seeks to measure the degree of semantic equaivalence. STS differs from paraphrasing in it is not a yes or no decision; rather STS allows for degress of similarity. 

### This Project

The goal of this project is for you to first implement some of the key aspects of the original BERT model including multi-head self-attention as well as a Transformer layer. Subsequently, you will utilize your completed BERT model to perform sentiment analysis on the Stanford Sentiment Treebank dataset as well as another dataset of movie reviews. Finally, in the latter half of this project, you will fine-tune and otherwise extend the BERT model to create sentence embeddings that can perform well across a wide range of down-stream tasks.

## Getting Started

For this project, you will need a machine with GPUs to train your models efficiently.

Once you are on an appropriate machine, clone the project GitHub repository with the following command.

```bash
git clone https://github.com/gpoesia/minbert-default-final-project
```

This repository contains the starter code and a minimalist implementation of the BERT model (min-BERT) that we will be using.  

### Code overview

The repository `minbert-default-final-project` contains the following files:

- `base-bert.py`: A base BERT implementation that can load pre-trained weights against which you can test your own implementation.
- `bert.py`: This file contains the BERT model. There are several sections of this implementation that need to be completed.
- `config.py`: This is where the configuration class is defined. You won't need to modify this file in this assignment.
- `classifier.py`: A classifier pipeline for running sentiment analysis. There are several sections of this implementations that need to be completed.
- `multitask_classifier.py`: A classifier pipeline for the second half of the project where you will train your minBERT implementation to simultaneously perform sentiment analysis, paraphrase detection, and semantic textual similarity tasks.
- `datasets.py`: A dataset handling script for the second half of this project.
- `evaluation.py`: A evaluation handling script for the second half of this project.
- `optimizer.py`: An implementation of the Adam Optimizer. The `step()` function of the Adam optimizer needs to be completed.
- `optimizer_test.py`: A test for your completed Adam optimizer.
- `optimizer_test.npy`: A numpy file containing weights for use in the `optimizer_test.py`.
- `sanity_check.py`: A test for your completed `bert.py` file.
- `sanity_check.data`: A data file for use in the `sanity_check.py`.
- `tokenizer.py`: This is where `BertTokenizer` is implemented. You won't need to modify this file in this assignment.
- `utils.py`: Utility functions and classes.

In addition, there are two directories:

- `data/`. This directory contains the `train`, `dev`, and `test` splits of `sst` and `CFIMDB` datasets as .csv files that you will be using in the first half of this projects. This directory also contains the `train`, `dev`, and `test` splits for later datasets that you will be using in the second half of this project.
- `predictions/`. This directory will contain the outputted predictions of your models on each of the provided datasets.

### Setup

Once you are on an appropriate machine and have cloned the project repository, it's time to run the setup commands.

- Make sure you have Anaconda or Miniconda installed.
- cd into `minbert-default-final-project` and run `source setup.sh`
  - This creates a conda environment called `cs224n_dfp`.
  - In addition to the defaults installed, you may also have to install the following packages: `zipp-3.11.0`, `idna-3.4`, and `chardet-4.0.0`.
  - For the first part of this assignment, you are only allowed to use libraries that are installed by setup.sh, no other exteral libraries are allowed (e.g., transformers).
  - Do not change any of the existing command options (including defaults) or add any new required parameters
- Run `conda activate cs224n_dfp`
  - This activates the `cs224n_dfp` environment.
  - Remember to do this each time you work on your code.
  
## Implementing minBERT

We have provided you with several of the building blocks for implementing minBERT. In this section, we will describe the baseline BERT model as well as the sections of it that you must implement.