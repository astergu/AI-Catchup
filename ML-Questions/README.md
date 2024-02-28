- [ML System Design](#ml-system-design)
  - [The 9-Step ML System Design Formula](#the-9-step-ml-system-design-formula)
    - [Step 1: Problem Formulation](#step-1-problem-formulation)
    - [Step 2: Metrics (Offline and Online)](#step-2-metrics-offline-and-online)
    - [Step 3: Architectural Components (MVP Logic)](#step-3-architectural-components-mvp-logic)
    - [Step 4: Data Collection and Preparation](#step-4-data-collection-and-preparation)
    - [Step 5: Feature Engineering](#step-5-feature-engineering)
    - [Step 6: Model Development and Offline Evaluation](#step-6-model-development-and-offline-evaluation)
    - [Step 7: Prediction Service](#step-7-prediction-service)
    - [Step 8: Online Testing and Model Deployment](#step-8-online-testing-and-model-deployment)
    - [Step 9: Scaling, Monitoring, and Updates](#step-9-scaling-monitoring-and-updates)
- [ML System Design Sample Questions](#ml-system-design-sample-questions)
  - [Recommendation Systems](#recommendation-systems)
  - [Search systems (retrieval, ranking)](#search-systems-retrieval-ranking)
  - [Ranking systems](#ranking-systems)
  - [NLP](#nlp)
  - [Computer Vision](#computer-vision)
  - [AV](#av)
  - [Other](#other)
- [ML System Design Topics](#ml-system-design-topics)
  - [Recommendation Systems](#recommendation-systems-1)
  - [Search and Ranking (Ads, newsfeed, etc)](#search-and-ranking-ads-newsfeed-etc)
  - [NLP](#nlp-1)
  - [Computer Vision](#computer-vision-1)
- [ML at Big Tech Companies](#ml-at-big-tech-companies)
- [More Resources](#more-resources)
- [References](#references)

# ML System Design

![ML system design](./image/ml_system_components.png)

- In an ML system design interview you are exposed to open ended questions with no single correct answer.
- The goal of ML system design interview is evaluate your your ability to zoom out and design a production-level ML system that can be deployed as a service within a company's ML infrastructure.

## The 9-Step ML System Design Formula

In order to design a solid ML system for real world applications, it is important to follow a design flow.

| | |
| ---- | ---- | 
| Step 1 | [Problem Formulation](#step-1-problem-formulation) |
| Step 2 | [Metrics (Offline and Online)](#step-2-metrics-offline-and-online) |
| Step 3 | [Architecture Components (MVP Logic)](#step-3-architectural-components-mvp-logic) |
| Step 4 | [Data Collection and Preparation](#step-4-data-collection-and-preparation) | 
| Step 5 | [Feature Engineering](#step-5-feature-engineering) | 
| Step 6 | [Model Development and Offline Evaluation](#step-6-model-development-and-offline-evaluation) |
| Step 7 | [Prediction Service](#step-7-prediction-service) | 
| Step 8 | [Online Testing and Deployment](#step-8-online-testing-and-model-deployment) | 
| Step 9 | [Scaling, Monitoring and Updates](#step-9-scaling-monitoring-and-updates) |

### Step 1: Problem Formulation

- Clarifying questions
- Use case(s) and business goal
- Requirements
  - Scope (features needed), scale, and personalization
  - Performance: prediction latency, scale of prediction
  - Constraints
  - Data: sources and availability
- Assumptions
- Translate an abstract problem into an ML problem
  - ML objective,
  - ML I/O,
  - ML category (e.g. binary classification, multi-classification, unsupervised learning, etc)
- Do we need ML to solve this problem?
  - Trade off between impact and cost
  - Costs: Data collection, data annotation, compute
  - if Yes, we choose an ML system to design. If No, follow a general system design flow.
            
### Step 2: Metrics (Offline and Online)

- Offline metrics (e.g. classification, relevance metrics)
  - Classification metrics
    - Precision, Recall, F1, ROC AUC, P/R AUC, mAP, log-loss, etc
      - Imbalanced data
    - Retrieval and ranking metrics
      - Precision@k, Recall@k (do not consider ranking quality)
      - mAP, MRR, nDCG
    - Regression metrics: MSE, MAE,
      - Problem specific metrics
        - Language: BLEU, BLEURT, GLUE, ROUGE, etc
        - ads: CPE, etc
      - Latency
      - Computational cost (in particular for on-device)
- Online metrics
  - CTR
  - Task/session success/failure rate,
  - Task/session total (e.g. watch) times,
  - Engagement rate (like rate, comment rate)
  - Conversion rate
  - Revenue lift
  - Reciprocal rank of first click, etc,
  - Counter metrics: direct negative feedback (hide, report)
- Trade-offs b/w metrics

### Step 3: Architectural Components (MVP Logic)

- High level architecture and main components
  - Non-ML components:
    - user, app server, DBs, KGs, etc and their interactions
  - ML components:
    - Modeling modules (e.g. candidate generator, ranker, ect)
    - Train data generator
    - ...
- Modular architecture design
  - Model 1 architecture (e.g. candidate generation)
  - Model 2 architecture (e.g. ranker, filter)
  - ...

### Step 4: Data Collection and Preparation

- Data needs
  - target variable
  - big actors in signals (e.g. users, items, etc)
  - type (e.g. image, text, video, etc) and volume
- Data Sources
  - availability and cost
  - implicit (logging), explicit (e.g. user survey)
- Data storage
  - ML Data types
    - structured
      - numerical (discrete, continuous)
      - categorical (ordinal, nominal),
      - unstructured(e.g. image, text, video, audio)
- Labelling (for supervised)
  - Labeling methods
    - Natural labels (extracted from data e.g. clicks, likes, purchase, etc)
      - Missing negative labels (not clicking is not a negative label)
        - Negative sampling
      - Explicit user feedback
        - Human annotation (super costly, slow, privacy issues)
  - Handling lack of labels
  - Programmatic labeling methods (noisy, pros: cost, privacy, adaptive)
    - Semi-supervised methods (from an initial smaller set of labels e.g. perturbation based)
    - Weak supervision (encode heuristics e.g. keywords, regex, db, output of other ML models)
  - Transfer learning
    - pre-train on cheap large data (e.g. GPT-3),
    - zero-shot or fine-tune for downstream task
  - Active learning
  - Labeling cost and trade-offs
- Data augmentation
- Data Generation Pipeline
  - Data collection/ingestion (offline, online)
  - Feature generation (next)
  - Feature transform
  - Label generation
  - Joiner

### Step 5: Feature Engineering

- Choosing features
  - Define big actors (e.g. user, item, document, query, ad, context),
  - Define actor specific features (current, historic)
    - Example user features: user profile, user history, user interests
    - Example text features: n-grams (uni,bi), intent, topic, frequency, length, embeddings
  - Define cross features (e.g. user-item, or query-document features)
    - Example query-document features: tf-idf
    - Example user-item features: user-video watch history, user search history, user-ad interactions(view, like)
  - Privacy constraints
- Feature representation
  - One hot encoding
  - Embeddings
    - e.g. for text, image, graphs, users (how), stores, etc
    - how to generate/learn?
    - pre-compute and store
  - Encoding categorical features (one hot, ordinal, count, etc)
  - Positional embeddings
  - Scaling/Normalization (for numerical features)
- Preprocessing features
  - Needed for unstructured data
    - Text: Tokenize (Normalize, pre-tokenize, tokenizer model (ch/word/subword level), post-process (add special tokens))
    - Images: Resize, normalize
    - Video: Decode frames, sample, resize, scale and normalize
- Missing Values
- Feature importance
- Featurizer (raw data -> features)
- Static (from feature store) vs dynamic (computed online) features


### Step 6: Model Development and Offline Evaluation

- Model selection (MVP)
  - Heuristics -> simple model -> more complex model -> ensemble of models
    - Pros and cons, and decision
    - Note: Always start as simple as possible (KISS) and iterate over
  - Typical modeling choices
    - Logistic Regression
    - Decision tree variants
      - GBDT (XGBoost) and RF
    - Neural networks
      - FeedForward
      - CNN
      - RNN
      - Transformers
    - Decision Factors
      - Complexity of the task
      - Data: Type of data (structured, unstructured), amount of data, complexity of data
      - Training speed
      - Inference requirements: compute, latency, memory
      - Continual learning
      - Interpretability
- Dataset
  - Sampling
    - Non-probabilistic sampling
    - Probabilistic sampling methods
      - random, stratified, reservoir, importance sampling
  - Data splits (train, dev, test)
    - Portions
    - Splitting time-correlated data (split by time)
      - seasonality, trend
    - Data leakage:
      - scale after split,
      - use only train split for stats, scaling, and missing vals
  - Class Imbalance
    - Resampling
    - weighted loss fcn
    - combining classes
- Model training
  - Loss functions
    - MSE, Binary/Categorical CE, MAE, Huber loss, Hinge loss, Contrastive loss, etc
  - Optimizers
    - SGD, AdaGrad, RMSProp, Adam, etc
  - Model training
    - Training from scratch or fine-tune
  - Model validation
  - Debugging
  - Offline vs online training
  - Model offline evaluation
  - Hyper parameter tuning
    - Grid search
  - Iterate over MVP model
    - Model Selection
    - Data augmentation
    - Model update frequency
  - Model calibration


### Step 7: Prediction Service

- Data processing and verification
- Web app and serving system
- Prediction service
- Batch vs Online prediction
  - Batch: periodic, pre-computed and stored, retrieved as needed - high throughput
  - Online: predict as request arrives - low latency
  - Hybrid: e.g. Netflix: batch for titles, online for rows
- Nearest Neighbor Service
  - Approximate NN
    - Tree based, LSH, Clustering based
- ML on the Edge (on-device AI)
  - Network connection/latency, privacy, cheap
  - Memory, compute power, energy constraints
  - Model Compression
    - Quantization
    - Pruning
    - Knowledge distillation
    - Factorization

### Step 8: Online Testing and Model Deployment

- A/B Experiments
  - How to A/B test?
    - what portion of users?
    - control and test groups
    - null hypothesis
- Bandits
- Shadow deployment
- Canary release

### Step 9: Scaling, Monitoring, and Updates

- Scaling for increased demand (same as in distributed systems)
  - Scaling general SW system (distributed servers, load balancer, sharding, replication, caching, etc)
    - Train data / KB partitioning
  - Scaling ML system
    - Distributed ML
      - Data parallelism (for training)
      - Model parallelism (for training, inference)
        - Asynchronous SGD
        - Synchronous SGD
    - Distributed training
      - Data parallel DT, RPC based DT
    - Scaling data collection
      - MT for 1000 languages
      - NLLB
    - Monitoring, failure tolerance, updating (below)
    - Auto ML (soft: HP tuning, hard: arch search (NAS))
- Monitoring
  - Logging
    - Features, predictions, metrics, events
  - Monitoring metrics
    - SW system metrics
    - ML metrics (accuracy related, predictions, features)
      - Online and offline metric dashboards
    - Monitoring data distribution shifts
      - Types: Covariate, label and concept shifts
      - Detection (stats, hypothesis testing)
      - Correction
- System failures
  - SW system failure
    - dependency, deployment, hardware, downtime
  - ML system failure
    - data distribution difference (test vs online)
    - feedback loops
    - edge cases (e.g. invalid/junk input)
    - data distribution changes
  - Alarms
    - failures (data pipeline, training, deployment), low metrics, etc
- Updates: Continual training
  - Model updates
    - train from scratch or a base model
      - how often? daily, weekly, monthly, etc
    - Auto update models
    - Active learning
    - Human in the loop ML

# ML System Design Sample Questions

Below are the most common ML system design questions for ML engineering interviews.

## Recommendation Systems

- Video/Movie recommendation(Netflix, Youtube)
- Friends / follower recommendation(Facebook, Twitter, LinkedIn)
- Event recommendation system (Eventbrite)
- Game recommendation
- Replacement product recommendation (Instacart)
- Rental recommendation (Airbnb)
- Place recommendation

## Search systems (retrieval, ranking)

- Document search
  - Text query search (full text, semantic),
  - Image/Video search,
  - Multimodal search (MM Query)


## Ranking systems

- Newsfeed system (ranking)
- Ads serving system (retrieval, ranking)
  - Ads click prediction (ranking)


## NLP

- Named entity linking system (NLP tagging, reasoning)
- Autocompletion / typeahead suggestion system
- Sentiment analysis system
- Language identification system
- Chatbot system
- Question answering system


## Computer Vision

- Image blurring system
- OCR/Text recognition system


## AV

- Self-driving car
  - Perception, Prediction, and Planning
  - Pedestrian jaywalking detection
- Ride matching system


## Other

- Proximity service / Yelp
- Food delivery time approximation
- Harmful content / Spam detection system
  - Multimodal harmful content detection
  - Fraud detection system
- Healthcare diagnosis system


# ML System Design Topics

## Recommendation Systems

- Candidate generation
  - Collaborative Filtering (CF)
    - User based, item based
    - Matrix factorization
    - Two-tower approach
  - Content based filtering
- Ranking
- Learning to rank (LTR)
  - point-wise (simplest), pairwise, list-wise


## Search and Ranking (Ads, newsfeed, etc)

- Search systems
  - Query search (keyword search, semantic search)
  - Visual search
  - Video search
  - Two stage model
    - document selection
    - document ranking
- Ranking
  - Newsfeed ranking system
  - Ads ranking system
  - Ranking as classification
  - Multi-stage ranking + blender + filter


## NLP

- Feature engineering
  - Preprocessing (tokenization)
- Text Embeddings
  - Word2Vec, GloVe, Elmo, BERT
- NLP Tasks
  - Text classification
    - Sentiment analysis
    - Topic modeling
  - Sequence tagging
    - Named entity recognition
    - Part of speech tagging
      - POS HMM
      - Viterbi algorithm, beam search
  - Text generation
    - Language modeling
      - N-grams vs deep learning models (trade-offs)
      - Decoding
  - Sequence 2 Sequence models
    - Machine Translation
      - Seq2seq models, NMT, Transformers
  - Question Answering
  - [Adv] Dialog and chatbots
    - CMU lecture on chatbots
    - CMU lecture on spoken dialogue systems
- Speech Recognition Systems
  - Feature extraction, MFCCs
  - Acoustic modeling
    - HMMs for AM
    - CTC algorithm (advanced)


## Computer Vision

- Image classification
  - VGG, ResNET
- Object detection
  - Two stage models (R-CNN, Fast R-CNN, Faster R-CNN)
  - One stage models (YOLO, SSD)
  - Vision Transformer (ViT)
  - NMS algorithm
- Object Tracking

# ML at Big Tech Companies

# More Resources

For more insight on different components above you can check out the following resources):

- [Full Stack Deep Learning course](https://fall2019.fullstackdeeplearning.com/)
- [Production Level Deep Learning](https://github.com/alirezadir/Production-Level-Deep-Learning)
- [Machine Learning Systems Design](https://github.com/chiphuyen/machine-learning-systems-design)
- [Stanford course on ML system design](https://stanford-cs329s.github.io/) [[video]](https://www.youtube.com/watch?v=AZNTqytOhXk&t=12771s)
- Book: [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.amazon.com/Designing-Machine-Learning-Systems-Production-Ready/dp/1098107969)

# References

- [https://github.com/alirezadir/Machine-Learning-Interviews](https://github.com/alirezadir/Machine-Learning-Interviews)
- [https://github.com/jayinai/ml-interview](https://github.com/jayinai/ml-interview)
- [https://github.com/khangich/machine-learning-interview](https://github.com/khangich/machine-learning-interview)
- [https://github.com/andrewekhalel/MLQuestions](https://github.com/andrewekhalel/MLQuestions)
- [https://github.com/aaronwangy/Data-Science-Cheatsheet](https://github.com/aaronwangy/Data-Science-Cheatsheet)