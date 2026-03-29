# CS231n: Deep Learning for Computer Vision

- [Schedule](https://cs231n.stanford.edu/schedule.html)
- [Lecture Notes](https://cs231n.github.io/)
- [Youtube Playlist](https://www.youtube.com/watch?v=2fq9wYslV0A&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16)

<table border="1" style="width:100%; border-collapse: collapse;">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th>Lecture</th>
      <th>Course Materials</th>
      <th>Assignments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://www.youtube.com/watch?v=2fq9wYslV0A&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&index=1&pp=iAQB&themeRefresh=1">Lecture 1: Introduction</a> <br> <a href="./slides/lecture_1_part_1.pdf">[slides 1]</a> <a href="./slides/lecture_1_part_2.pdf">[slides 2]</a> </td>
      <td></td>
      <td>—</td>
    </tr>
    <tr style="background-color: #e8f4f8;">
      <td colspan="3" style="text-align:center;"><strong>Deep Learning Basics</strong></td>
    </tr>
    <tr>
      <td><a href="https://www.youtube.com/watch?v=pdqofxJeBN8&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&index=2">Lecture 2: Image Classification with Linear Classifiers</a> <br> The data-driven approach <br> K-nearest neighbor <br> Linear Classifiers <br> Algebraic/Visual/Geometric viewpoints <br> Softmax loss <br> <a href="https://cs231n.stanford.edu/slides/2025/lecture_2.pdf">[slides]</a></td>
      <td><a href="https://cs231n.github.io/classification/">Image Classification</a> <br> <a href="https://cs231n.github.io/linear-classify/">Linear Classification</a> </td>
      <td></td>
    </tr>
    <tr>
      <td>Lecture 3: Regularization and Optimization <br> <a href="https://cs231n.stanford.edu/slides/2025/lecture_3.pdf">[slides]</a> </td>
      <td><a href="https://cs231n.github.io/optimization-1/">Optimization</a> </td>
      <td></td>
    </tr>
    <tr>
      <td>Lecture 4: Neural Networks and Backpropagation <br> <a href="https://cs231n.stanford.edu/slides/2025/lecture_4.pdf">[slides]</a></td>
      <td><a href="http://cs231n.github.io/optimization-2">Backprop</a> <br> <a href="https://cs231n.stanford.edu/handouts/linear-backprop.pdf">Linear backprop example</a> <br> Suggested Readings: <br> 1. <a href="https://distill.pub/2017/momentum/">Why Momentum Really Works</a> <br>2. <a href="https://cs231n.stanford.edu/handouts/derivatives.pdf">Derivatives notes</a> <br> 3. <a href="https://cs231n.stanford.edu/papers/lecun-98b.pdf">Efficient backprop</a> <br> 4. More backprop references: <br> <a href="http://colah.github.io/posts/2015-08-Backprop/">[1]</a> <a href="http://neuralnetworksanddeeplearning.com/chap2.html">[2]</a> <a href="https://www.youtube.com/watch?v=q0pm3BrIUFo">[3]</a> </td>
      <td></td>
    </tr>
    <tr>
        <td>Backprop Review Session <br> <a href="https://cs231n.stanford.edu/slides/2025/section_2.pdf">[slides]</a> <a href="https://colab.research.google.com/drive/1yjxfAugU5JrbgCb1TcCbXDMCKE_G_P-e">[Colab]</a> </td>
        <td></td>
        <td></td>
    </tr>
    <tr style="background-color: #e8f4f8;">
      <td colspan="3" style="text-align:center;"><strong>Perceiving and Understanding the Visual World</strong></td>
    </tr>
    <tr>
      <td>Lecture 5: Image Classification with CNNs <br> History <br> Higher-level representations, image features <br> Convolution and pooling <br> <a href="https://cs231n.stanford.edu/slides/2025/lecture_5.pdf">[slides]</a></td>
      <td><a href="http://cs231n.github.io/convolutional-networks">Convolutional Networks</a> </td>
    </tr>
  </tbody>
</table>


## Notes

### Lecture 1: Introduction

- 2012 to Present: Deep Learning explosion
- Overview
  - Deep Learing Basics
  - Perceiving and Understanding the Visual World
  - Generative and Interactive Visual Intelligence
  - Human-Centered Applications and Implications

### Lecture 2: Image Classification with Linear Classifiers

- `Image Classification`: a core task in computer vision
  - challenges
    - viewpoint variation
    - illumination
    - blackground clutter
    - occlusion
    - deformation
    - intraclass variation
  - An image classifier
    - Machine Learning: Data-Driven Approach
      - collect a dataset of images and labels
      - use machine learning algorithms to train a classifier
      - evaluate the classifier on new images
- `Nearest Neighbor Classifier`
  - distance matric: 
    - `L1 distance` $d_1(I_1, I_2)=\sum|I_1-I_2|$
    - `L2 distance` $d_2(I_1, I_2)=\sqrt{\sum(I_1-I_2)^2}$
  - [visualization tool](http://vision.stanford.edu/teaching/cs231n-demos/knn)
  - A good implementation: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
  - Johnson et al, "Billion-scale similarity search with GPUs"
  - Hyperparameters
    - What is the best value of $k$ to use?
    - What is the best distance to use?
  - Setting Hyperparameters
    - cross-validation: split data into folds, try each fold as validation and average the results.
    - useful for small datasets, but not used frequently in deep learning
- Example dataset: `CIFAR10`
  - 10 classes, 50,000 training images, 10,000 testing images 
- k-nearest neighbor with pixel distance never used.
  - distance metrics on pixels are not informative
- Linear Classifier
- 

### Lecture 3: 

## Papers

- AlexNet