# CS224n Assignment #4: Neural Machine Translation 

> Note: Heavily inspired by the https://github.com/pcyin/pytorch_basic_nmt repository

This assignment is split into two sections: `Neural Machine Translation with RNNs` and `Analyzing NMT Systems`. The first is primarily coding and implementation focused, whereas the second entirely consists of written, analysis questions. Note that the NMT system is more complicated than the neural networks we have previously constructed within this class and takes about `2 hours to train on GPU`. 


## Neural Machine Translation with RNNS

In Machine Translation, our goal is to convert a sentence from the *source* language (e.g. Mandrain Chinese) to the *target* language (e.g. English). In this assignment, we will implement a sequence-to-sequence (Seq2Seq) network with attention, to build a Neural Machine Translation (NMT) system. In this section, we describe the **training procedure** for the proposed NMT system, which uses a Bidirectional LSTM Encoder and a Unidirectional LSTM Decoder.

![Seq2Seq Model with Multiplicative Attention](./image/seq2seq_model.png)