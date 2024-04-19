"""
A quick refresher on the Word2Vec architecture as defined by Mikolov et al:

    Three layers: input, hidden and output.
    Input and output are the size of the vocabulary. Hidden is smaller.
    Fully connected with linear activations.

There are two variants of this architecture:

    CBOW (continuous bag-of-words): context word is input, center word is output.
    Skip-gram: center word is input, context word is output.
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from datasets import load_dataset # Use Huggingface data

# Define a simple dataset class
class TextDataset(Dataset):
    def __init__(self, text):
        self.text = text
        self.vocab = set(text.split())
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.data = self.tokenize(text)
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def tokenize(self, text):
        return [self.word2idx[word] for word in text.split()]

# Define the Skip-gram model
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_word):
        embed = self.embedding(center_word)
        return self.output(embed)

# Define the Negative Sampling loss function
class NegativeSamplingLoss(nn.Module):
    def __init__(self, num_words, num_negative_samples):
        super(NegativeSamplingLoss, self).__init__()
        self.num_words = num_words
        self.num_negative_samples = num_negative_samples

    def forward(self, output, target):
        batch_size = output.size(0)
        target = target.view(-1, 1)
        noise_words = torch.randint(0, self.num_words, (batch_size, self.num_negative_samples), dtype=torch.long)
        noise_words = noise_words.to(output.device)
        
        target_logits = output.gather(1, target)
        noise_logits = output.gather(1, noise_words)
        
        positive_loss = torch.log(torch.sigmoid(target_logits))
        negative_loss = torch.log(torch.sigmoid(-noise_logits)).sum(dim=1)
        
        loss = -(positive_loss + negative_loss).mean()
        return loss

# Training function
def train_word2vec_model(dataset, embedding_dim, num_negative_samples, num_epochs, batch_size, learning_rate):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SkipGram(dataset.vocab_size, embedding_dim)
    criterion = NegativeSamplingLoss(dataset.vocab_size, num_negative_samples)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for data in dataloader:
            optimizer.zero_grad()
            center_word = data.to(torch.long)
            output = model(center_word)
            loss = criterion(output, center_word)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader)}")

    return model, dataset.word2idx, dataset.idx2word

# Testing function
def test_word2vec_model(model, word2idx, idx2word, word, k=5):
    word_idx = word2idx[word]
    embedding_weights = model.embedding.weight.detach()
    word_embedding = embedding_weights[word_idx]
    similarity_scores = torch.matmul(embedding_weights, word_embedding)
    top_k_similar = torch.topk(similarity_scores, k=k+1, largest=True)[1]
    similar_words = [idx2word[idx.item()] for idx in top_k_similar]
    return similar_words[1:]

# Example usage
text = "natural language processing and word embeddings are important for deep learning"
embedding_dim = 50
num_negative_samples = 5
num_epochs = 100
batch_size = 32
learning_rate = 0.001

model, word2idx, idx2word = train_word2vec_model(text, embedding_dim, num_negative_samples, num_epochs, batch_size, learning_rate)
print(word2idx)
similar_words = test_word2vec_model(model, word2idx, idx2word, word="language")
print("Words similar to 'language':", similar_words)

"""
# Load the hate speech dataset
dataset = load_dataset("tweets_hate_speech_detection")
# Display dataset information
print(dataset)
# Access the train split
train_data = dataset["train"]

# Example usage
print("Number of examples in train split:", len(train_data))
print("First example:")
print(train_data[0])

# Use the text from the dataset as test data for Word2Vec
# Assuming you concatenate all the tweets into one long string
tweets_text = " ".join(train_data["tweet"])
dataset = TextDataset(text)
"""

import torchtext

text_dataset = torchtext.datasets.WikiText2(root="./data")
model, word2idx, idx2word = train_word2vec_model(text_dataset, embedding_dim, num_negative_samples, num_epochs, batch_size, learning_rate)
print(word2idx)
similar_words = test_word2vec_model(model, word2idx, idx2word, word="father")
print("Words similar to 'father':", similar_words)