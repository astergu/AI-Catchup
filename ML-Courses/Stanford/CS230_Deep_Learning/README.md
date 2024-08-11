# CS 230: Deep Learning

[Course Homepage](https://cs230.stanford.edu)

| Lecture | Topics | Extra Materials | Assignments |
| ---- | ---- | ---- | ---- |
| Lecture 1 | Introduction <br> [[video]](https://www.youtube.com/watch?v=PySo_6S4ZAg&list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb) [[slides]](./slides/lecture_1.pdf) |  | |
| Lecture 2 | Deep Learning Intuition <br> [[video]]() [[slides]](./slides/lecture_2.pdf) | [Coursera: Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning) <br> - C1M1: Introduction to deep learning ([slides](./slides/C1M1.pdf)) <br> - C1M2: Neural Network Basics ([slides](./slides/C1M2.pdf)) | |
| Lecture 3 | Adversarial Attacks <br> [[video]]() [[slides]](./slides/lecture_3.pdf) | Optional Reading: <br> - [Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572.pdf) <br> [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf) <br> - [Conditional GAN](https://arxiv.org/pdf/1611.07004.pdf) <br> - [Super-Resolution](https://arxiv.org/pdf/1609.04802.pdf) <br> - [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf) <br><br> Coursera courses: <br> - C1M3: Shallow Neural Network ([slides](./slides/C1M3.pdf)) <br> - C1M4: Deep Neural Networks ([slides](./slides/C1M4.pdf))  | |
| | | | |
| | | | |
| Lecture 3 | Full-Cycle Deep Learning Projects <br> [[video]](https://www.youtube.com/watch?v=JUJNGv_sb4Y&list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb&index=3&pp=iAQB) | | |
| | | | |
| | | | |
| | | | |


## Class Notes

### Lecture 2: Deep Learning Intuition

- Machine Learning pipeline
  - Data
  - Input
  - Output
  - Architecture
  - Loss
- Example machine learning applications
  - A school wants to use `Face Identification` for recognizing students in facilities (dining hall, gym, pool...)
    - `K-Nearest Neighbors`
  - You want to use `Face Clustering` to group pictures of the same people on your smart phone 
    - `K-Means Algorithm`
  - Art generation
    - inspect its style & content
  - Trigger word detection: given a 10sec audio speech, detect the word "activate".


### Lecture 3: Adversarial Attacks

- Attacking a network with adversarial examples
  - Goal: Given a network pretrained on ImageNet, find an input image that is a cat but will be classify as an iguana.
- Defenses against adversarial examples
  - Types of attacks
    - Non-targeted attackes
    - Targeted attackes
  - Knowledge of the attacker
    - White-box 
    - Black-box
  - Solutions
    - Solution 1: Create a SafetyNet
    - Solution 2: Train on correctly labelled adversarial examples
    - Solution 3: Adversarial training / Adversarial logit pairing
- Why are nerual networks vulnerable to adversarial examples? 
  - Generative Adversarial Networks (GANs)
    - Goal: collect a lot of data, use it to train a model to generate similar data from scratch

### Lecture 3:  Full-Cycle Deep Learning Projects

- Steps
  - Select Project
  - Get Data
  - Design Model
  - Train Model
  - Test Model
  - Deploy
  - Monitor