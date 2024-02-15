# CS224n Assignment #3: Dependency Parsing

In this assignment, you will build a neural dependency parser using PyTorch. In Part 1, you will learn about two general neural network techniques (Adam Optimization and Dropout). In Part 2, you will implement and train a dependency parser using the techniques from Part 1, before analyzing a few erroneous dependency parses.

## Machine Learning & Neural Networks

1. **Adam Optimizer** <br>

Adam Optimization uses a more sophisticated update rule with two additional steps. <br>

(1) First, Adam uses a trick called `momentum` by keeping track of `m`, a rolling average of the gradients: <br>
$m_{t+1} \leftarrow \beta_{1}m_{t} + (1 - \beta_{1})\nabla_{\theta_t}J_{minibatch}(\theta_t)$ <br>
$\theta_{t+1} \leftarrow \theta_t - \alpha m_{t+1}$ <br>

where $\beta_1$ is a hyperparameter between 0 and 1 (often set to 0.9). **Briefly explain in 2-4 sentences (you don't need to prove mathematically, just give an intuition) how using `m` stops the updates from varying as much and why this low variance may be helpful to learning, overall**.

> Answer: <br>
> Momentum allows the optimizer to build up velocity in direction with consistent gradients, and also help the optimizer generalize better by smoothing out the updates and preventing it from getting stuck in sharp, narrow valleys. In detail, at every iteration, there is a decaying average of negative weight gradients which causes the update step not to be instantaneous but rather depend by some amount on previous updates. If the loss curve is very narrow, there is a possibility to overshoot, but it may be a better idea to overshoot such sharp local minima to prevent overfitting. So momentum prefers flat minima 

(2) Adam extends the idea of `momentum` with the trick of `adaptive learning rates` by keeping track of `v`, a rolling average of the magnitudes of the gradients: <br>
$m_{t+1}\leftarrow \beta_1m_t + (1-\beta_1)\nabla_{\theta_t}J_{minibatch}(\theta_t)$ <br>
$v_{t+1}\leftarrow \beta_2v_t + (1-\beta_2)(\nabla_{\theta_t}J_{minibatch}(\theta_t) \odot \nabla_{\theta_t}J_{minibatch}(\theta_t))$ <br>
$\theta_{t+1}\leftarrow \theta_t - \alpha m_{t+1}$ <br>

where $\odot$ and / denote elementwise multiplication and division (so $z\odot z$ is elementwise squaring) and $\beta_2$ is a hyperparameter between 0 and 1 (often set to 0.99). **Since Adam divides the update by $\sqrt{v}$, which of the model parameters will get larger updates? Why might this help with learning?**

> Answer: <br>
> Model parameters with small gradients will get larger updates, because if an accumulated sqaure norm is very small, then dividing learning rate by what is small will cause larger values in corresponding gradient axes, thus larger step sizes. This is useful for lerning because sometimes the gradient descent might stuck performing barely noticeable updates in directions with tiny gradient values. It should also be noted that big gradients result in smaller step sizes thus the square norm normalizes the gradient step direction-wise. 

2. Dropout is a regularization technique. During training, dropout randomly sets units in the hidden layer `h` to zero with probability $p_{drop}$ (dropping different units each minibatch), and then multiplies `h` by a constant $\gamma$. We can write this as: $h_{drop}=\gamma d\odot h$ <br>
where $d \in {0, 1}^{D_h}$ ($D_h$ is the size of $h$) is a mask vector where each entry is 0 with probability $p_{drop}$ and 1 with probability (1-$p_{drop}$). $\gamma$ is chosen such that the expected value of $h_{drop}$ is $h$: <br> 
$\mathbb{E}_{p_{drop}}[h_{drop}]_i=h_i$ <br>
for all $i \in {1, ..., D_h}$.

(1) **What must $\gamma$ equal in terms of $p_{drop}$? Briefly justify your answer or show your math derivation using the equations given above.** <br>

> Answer: <br>
> We need to scale outputs by $\gamma$ to ensure that the scale of outputs at test time is identical to the expected outputs at training time. If we don't multiply by $\gamma$, we can see that during training the expected value of any output entry is: <br>
> $\mathbb{E}_{p_{drop}}[h_{drop}]_i=p_{drop}0+(1-p_{drop})h_i=(1-p_{drop})h_i$ <br>
> Thus, during testing when nothing is dropped we'd have to multiply the output vector by ($1-p_{drop}$) to match the expectation during training. To keep the testing unchanged, we can just apply `inverse dropout` and multiply the output during training by the inverse of the value we'd otherwise multiply during prediction. We can derive that: <br>
> $\gamma=\frac{1}{1-p_{drop}}$

(2) **Why should dropout be applied during training? Why should dropout NOT be applied during evaluation?** 

> Answer: <br>
> Dropout increases network robustness by making it not to rely too much on some specific neurons. During evaluation we want to use all the information from the trained neurons. It can be interpreted as evaluating an averaged prediction across the exponentially-sized ensemble of all the possible binary masks.

## Neural Transition-Based Dependency Parsing

In this section, you'll be implementing a neural-network based dependency parser with the goal of maximizing performance on the UAS (Unlabeled Attachment Score) metric.

Before you begin, please follow the `README.txt` to install all the needed dependencies for the assignment.

A dependency parser analyzes the grammatical structure of a sentence, establishing relationships between `head` words, and words which modify those heads. There are multiple types of dependency parsers, including transition-based parsers, graph-based parsers, and feature-based parsers. Your implementation will be a `transition-based` parser, which incrementally builds up a parse one step at a time. At every step it maintains a `partial parse`, which is represented as follows: 
- A `stack` of words that are currently being processed.
- A `buffer` of words yet to be processed.
- A list of `dependencies` predicted by the parser.

Initially, the stack only contains `ROOT`, the dependencies list is empty, and the buffer contains all words of the sentence in order. At each step, the parser applies a `transitition` to the partial parse until its buffer is empty and the stack size is 1. The following transitions can be applied: 
- `SHIFT`: removes the first word from the buffer and pushes it onto the stack.
- `LEFT-ARC`: marks the second (second most recently added) item on the stack as a dependent of the first item and removes the second item from the stack, adding a `first_word` $\rightarrow$ `second_word` dependency to the dependency list.
- `RIGHT-ARC`: marks the first (most recently added) item on the stack as a dependent of the second item and removes the first item from the stack, adding a `second_word` $\rightarrow$ `first_word` dependency to the dependency list.

On each step, your parser will decide among the three transitions using a neural network classifier.

1. Go through the sequence of transitions needed for parsing the sentence "*I attended lectures in the NLP class*". The dependency tree for the sentence is show below. At each step, give the configuration of the stack and buffer, as well as what transition was applied this step and what new dependency was added (if any). The first four steps are provided below as an example.

| Stack | Buffer | New dependency | Transitition | 
| ----- | ------ | ------------- | ----------- |
| [ROOT] | [I, attended, lectures, in, the, NLP, class] | | Initial Configuration | 
| [ROOT, I] | [attended, lectures, in, the, NLP, class] | | SHIFT |
| [ROOT, I, attended] | [lectures, in, the, NLP, class] | | SHIFT |
| [ROOT, attended] | [lectures, in, the, NLP, class] | attended $\rightarrow$ I | LEFT-ARC |

> Solution: <br>
> | Stack | Buffer | New dependency | Transitition | 
> | ----- | ------ | ------------- | ----------- |
> | [ROOT, attended, lectures] | [in, the, NLP, class] | | SHIFT |
> | [ROOT, attended] | [in, the, NLP, class] | attended $\rightarrow$ lectures | RIGHT-ARC | 
> | [ROOT, attended, in] | [the, NLP, class] | | SHIFT |
> | [ROOT, attended, in, the] | [NLP, class] | | SHIFT | 
> | [ROOT, attended, in, the, NLP] | [class] | | SHIFT |
> | [ROOT, attended, in, the, NLP, class] | [] | | SHIFT |
> | [ROOT, attended, in, the, class] | [] | class $\rightarrow$ NLP | LEFT-ARC |
> | [ROOT, attended, in, class] | [] | class $\rightarrow$ the | LEFT-ARC |
> | [ROOT, attended, class] | [] | class $\rightarrow$ in | LEFT-ARC |
> | [ROOT, attended] | [] | attended $\rightarrow$ class | RIGHT-ARC |
> | [ROOT] | [] | ROOT $\rightarrow$ attended | RIGHT-ARC |


2. A sentence containing `n` words will be parsed in how many steps (in terms of `n`)? Briefly explain in 1-2 sentences why.

> Answer: <br>
> The number of steps required to parse a sentence of `n` words is `2n` because a transition-based parser pushes `n` words to the stack with SHIFT and then removes those `n` words with the LEFT-ARC or RIGHT-ARC.

3. Implement the `__init__` and `parse_step` functions in the `PartialParse` class in `parser_transitions.py`. This implements the transition mechanics your parser will use. You can run basic (non-exhaustive) tests by runing `python parser_transitions.py part_c`.

```python
class PartialParse(object):
    def __init__(self, sentence):
        self.sentence = sentence

        self.stack = ["ROOT"]
        self.buffer = sentence.copy()
        self.dependencies = []
    
    def parse_step(self, transition):
        if transition == "S":
            self.stack.append(self.buffer.pop(0))
        elif transition == "LA":
            self.dependencies.append((self.stack[-1], self.stack.pop(-2)))
        elif transition == "RA":
            self.dependencies.append((self.stack[-2], self.stack.pop(-1)))
```

4. Our network will predict which transition should be applied next to a partial parse. We could use it to parse a single sentence by applying predicted transitions until the parse is complete. However, neural networks run much more efficiently when making predictions about batches of data at a time (i.e., predicting the next transition for any different partial parses simultaneously). We can parse sentences in minibatches with the following algorithm.

Implement this algorithm in the `minibatch_parse` function in `parser_transitions.py`. You can run basic (non-exhaustive) tests by running `python parser_transitions.py part_d`. 

```python
def minibatch_parse(sentences, model, batch_size):
    dependencies = []

    partial_parses = [PartialParse(sentence) for sentence in sentences]
    unfinished_parses = partial_parses[:]
    while unfinished_parses:
        batch = unfinished_parses[:batch_size]
        transitions = model.predict(batch)
        for i, transition in enumerate(transitions):
            batch[i].parse_step(transition)
            if len(batch[i].buffer) == 0 and len(batch[i].stack) == 1:
                unfinished_parses.remove(batch[i])
    
    dependencies = [partial_parse.dependencies for partial_parse in partial_parses]

    return dependencies
```

5. We are now going to train a neural network to predict, given the state of the stack, buffer, and dependencies, which transition should be applied next.

In `parser_model.py` you will find skeleton code to implement this simple neural network using PyTorch. Complete the `__init__`, `embedding_lookup` and `forward` functions to implement the model. Then complete the `train_for_epoch` and `train` functions within the `run.py` file.

```python
class ParserModel(nn.Module):
    def __init__(self, embeddings, n_features=36,
        hidden_size=200, n_classes=3, dropout_prob=0.5):
        """ Initialize the parser model.

        @param embeddings (ndarray): word embeddings (num_words, embedding_size)
        @param n_features (int): number of input features
        @param hidden_size (int): number of hidden units
        @param n_classes (int): number of output classes
        @param dropout_prob (float): dropout probability
        """
        super(ParserModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.embeddings = nn.Parameter(torch.tensor(embeddings))

        self.embed_to_hidden_weight = nn.Parameter(torch.empty(self.n_features * self.embed_size, self.hidden_size))
        nn.init.xavier_uniform_(self.embed_to_hidden_weight)
        self.embed_to_hidden_bias = nn.Parameter(torch.empty(self.hidden_size))
        nn.init.uniform_(self.embed_to_hidden_bias)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.hidden_to_logits_weight = nn.Parameter(torch.empty(self.hidden_size, self.n_classes))
        nn.init.xavier_uniform_(self.hidden_to_logits_weight)
        self.hidden_logits_bias = nn.Parameter(torch.empty(self.n_classes))
        nn.init.uniform_(self.hidden_logits_bias)
    
    def embedding_lookup(self, w):
        """ Utilize `w` to select embeddings from embedding matrix `self.embeddings`
            @param w (Tensor): input tensor of word indices (batch_size, n_features)

            @return x (Tensor): tensor of embeddings for words represented in w
                                (batch_size, n_features * embed_size)
        """
        x = torch.index_select(self.embeddings, 0, w.flatten()).view(w.size(0), -1)

        return x
    
    def forward(self, w):
        """ Run the model forward.

            Note that we will not apply the softmax function here because it is included in the loss function nn.CrossEntropyLoss

            PyTorch Notes:
                - Every nn.Module object (PyTorch model) has a `forward` function.
                - When you apply your nn.Module to an input tensor `w` this function is applied to the tensor.
                    For example, if you created an instance of your ParserModel and applied it to some `w` as follows,
                    the `forward` function would called on `w` and the result would be stored in the `output` variable:
                        model = ParserModel()
                        output = model(w) # this calls the forward function
                - For more details checkout: https://pytorch.org/docs/stable/nn.html#torch.nn.Module.forward

        @param w (Tensor): input tensor of tokens (batch_size, n_features)

        @return logits (Tensor): tensor of predictions (output after applying the layers of the network)
                                 without applying softmax (batch_size, n_classes)
        """
        x = self.embedding_lookup(w)
        x = F.relu(torch.matmul(x, self.embed_to_hidden_weight) + self.embed_to_hidden_bias)
        x = self.dropout(x)
        logits = torch.matmul(x, self.hidden_to_logits_weight) + self.hidden_logits_bias

        return logits
```

Finally execute `python run.py` to train your model and compute predictions on test data from Penn Treebank.

```python
def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size):
    """ Train the neural dependency parser for single epoch.

    Note: In PyTorch we can signify train versus test and automatically have
    the Dropout Layer applied and removed, accordingly, by specifying
    whether we are training, `model.train()`, or evaluating, `model.eval()`

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param optimizer (nn.Optimizer): Adam Optimizer
    @param loss_func (nn.CrossEntropyLoss): Cross Entropy Loss Function
    @param batch_size (int): batch size

    @return dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
    """
    parser.model.train() # Places model in "train" mode, i.e. apply dropout layer
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            optimizer.zero_grad()   # remove any baggage in the optimizer
            loss = 0. # store loss for this batch here
            train_x = torch.from_numpy(train_x).long()
            train_y = torch.from_numpy(train_y.nonzero()[1]).long()

            ### YOUR CODE HERE (~4-10 lines)
            ### TODO:
            ###      1) Run train_x forward through model to produce `logits`
            ###      2) Use the `loss_func` parameter to apply the PyTorch CrossEntropyLoss function.
            ###         This will take `logits` and `train_y` as inputs. It will output the CrossEntropyLoss
            ###         between softmax(`logits`) and `train_y`. Remember that softmax(`logits`)
            ###         are the predictions (y^ from the PDF).
            ###      3) Backprop losses
            ###      4) Take step with the optimizer
            ### Please see the following docs for support:
            ###     Optimizer Step: https://pytorch.org/docs/stable/optim.html#optimizer-step
            outputs = parser.model(train_x)
            loss = loss_func(outputs, train_y)
            loss.backward()
            optimizer.step()

            ### END YOUR CODE
            prog.update(1)
            loss_meter.update(loss.item())

    print ("Average Train Loss: {}".format(loss_meter.avg))

    print("Evaluating on dev set",)
    parser.model.eval() # Places model in "eval" mode, i.e. don't apply dropout layer
    dev_UAS, _ = parser.parse(dev_data)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return dev_UAS

def train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005):
    """ Train the neural dependency parser.

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param output_path (str): Path to which model weights and results are written.
    @param batch_size (int): Number of examples in a single batch
    @param n_epochs (int): Number of training epochs
    @param lr (float): Learning rate
    """
    best_dev_UAS = 0


    ### YOUR CODE HERE (~2-7 lines)
    ### TODO:
    ###      1) Construct Adam Optimizer in variable `optimizer`
    ###      2) Construct the Cross Entropy Loss Function in variable `loss_func` with `mean`
    ###         reduction (default)
    ###
    ### Hint: Use `parser.model.parameters()` to pass optimizer
    ###       necessary parameters to tune.
    ### Please see the following docs for support:
    ###     Adam Optimizer: https://pytorch.org/docs/stable/optim.html
    ###     Cross Entropy Loss: https://pytorch.org/docs/stable/nn.html#crossentropyloss
    optimizer = optim.Adam(parser.model.parameters())
    loss_func = nn.CrossEntropyLoss(reduction='mean') # mean reduction is default

    ### END YOUR CODE

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_UAS = train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size)
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            print("New best dev UAS! Saving model.")
            torch.save(parser.model.state_dict(), output_path)
        print("")
```

> After training, my test UAS hit `89.09`.