# CS336 Assignment 1 (basics): Build a Transformer LM 

## Assignment Overview

In this assignment, you will build all the components needed to train a standard Transformer language model (LM) from scratch and train some models.

**What you will implement**
- Byte-pair encoding (BPE) tokenizer
- Transformer language model (LM)
- The cross-entropy loss function and the AdamW optimizer
- The training loop, with support for serializing and loading model and optimizer state

**What you will run**
- Train a BPE tokenizer on the TinyStories dataset.
- Run your trained tokenizer on the dataset to convert it into a sequence of integer IDs.
- Train a Transformer LM on the TinyStories dataset.
- Generate samples and evaluate perplexity using the trained Transformer LM.
- Train models on OpenWebText and submit your attained perplexities to a leaderboard.

**What you can use**

We expect you to build these components from scratch. In particular, you may *not* use any definitions from `torch.nn`, `torch.nn.functional`, or `torch.optim` except for the following:

- `torch.nn.Parameter`
- Container classes in `torch.nn` (e.g., `Module`, `ModuleList`, `Sequential`, etc.)
- The `torch.optim.Optimizer` base class

**What the code looks like**

All the assignment code is available in [official-assignment1-basics](./official-assignment1-basics/).

- `cs336_basics/*`: This is where you write your code. Note that there's no code in here --- you can do whatever you want from scratch!
- `adapters.py`: There is a set of functionality that your code must have. For each piece of functionality (e.g., scaled dot product attention), fill out its implementation (e.g., `run_scaled_dot_product_attention`) by simply invoking your code. Note: your changes to `adapters.py` should not contain any substantive logic; this is glue code.
- `test_*.py`: This contains all the tests that you must pass (e.g., `test_scaled_dot_product_attention`), which will invoke the hooks defined in `adapters.py`. Don't edit the test files.

**How to submit**

You will submit the following files to Gradescope:

- `writeup.pdf`: Answer all the written questions. Please typeset your reponses.
- `code.zip`: Contains all the code you've written.

**Where to get datasets**

This assignment will use two pre-processed datasets: TinyStories [Eldan and Li, 2023] and OpenWebText [Gokaslan et al., 2019]. Both datasets are single, large plaintext files. If you are doing the assignment with the class, you can find these files at `/data` of any non-head node machine.

If you are following along at home, you can download these files with the commands inside the `README.md`.


### Low-Resource/Downscaling Tip

With the staff solution code, we can train an LM to generate reasonably fluent text on an Apple M3 Max chip with 36 GB RAM, in under 5 minutes on Metal GPU (MPS) and about 30 minutes using the CPU. 

## Byte-Pair Encoding (BPE) Tokenizer

In the first part of the assignment, we will train and implement a byte-level byte-pair encoding (BPE) tokenizer [Sennrich et al., 2016, Wang et al., 2019]. In particular, we will represent arbitray (Unicode) strings as a sequence of bytes and train our BPE tokenizer on this byte sequence. Later, we will use this tokenizer to encode text (a string) into tokens (a sequence of integers) for language modeling.

### The Unicode Standard

Unicode is a text encoding standard that maps characters to integer `code points`. As of Unicode 16.0 (released in September 2024), the standard defines 154,998 characters across 168 scripts.


#### Problem (unicode1): Understanding Unicode (1 point)

1. What Unicode Character does `chr(0)` return?
2. How does this character's string representation (`__repr__()`) differ from its printed representation?
3. What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
```
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

### Unicode Encodings

