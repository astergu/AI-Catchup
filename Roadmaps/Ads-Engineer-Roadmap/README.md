# Ads Engineer/Scientist Roadmap

Comprehensive guide to becoming an Advertising Engineer or Scientist at major tech companies (Google, Meta, Amazon, TikTok, etc.).

---

## Table of Contents
- [Role Overview](#role-overview)
- [Core Qualifications](#core-qualifications)
- [Technical Skills](#technical-skills)
- [Machine Learning Expertise](#machine-learning-expertise)
- [Domain Knowledge](#domain-knowledge)
- [System Design & Engineering](#system-design--engineering)
- [Soft Skills](#soft-skills)
- [Career Progression](#career-progression)

---

## Role Overview

**Ads Engineers/Scientists** build and optimize large-scale machine learning systems for computational advertising. They work on:
- Click-Through Rate (CTR) prediction
- Conversion Rate (CVR) prediction
- Bid optimization and auction mechanisms
- Ad ranking and retrieval systems
- Budget pacing and delivery optimization
- Real-time bidding (RTB) systems
- Ad fraud detection
- Personalization and targeting

---

## Core Qualifications

### Education
- **Minimum**: Bachelor's degree in Computer Science, Engineering, Mathematics, Statistics, or related field
- **Preferred**: Master's or Ph.D. in Machine Learning, Computer Science, Statistics, or Operations Research
- **Alternative**: Strong track record of building production ML systems with demonstrated impact

### Experience
- **Entry Level (L3/E3)**: 0-2 years, strong ML fundamentals and coding skills
- **Mid Level (L4/E4)**: 2-5 years building ML models in production
- **Senior (L5/E5)**: 5-8 years with significant impact on business metrics
- **Staff+ (L6+)**: 8+ years, technical leadership, system-level thinking

---

## Technical Skills

### 1. Programming & Software Engineering

#### Languages (Must-Have)
- **Python**: Primary language for ML development
  - NumPy, Pandas, Scikit-learn
  - TensorFlow, PyTorch, JAX
  - Fast prototyping and experimentation
- **SQL**: Data analysis and feature engineering
  - Complex queries, window functions
  - Query optimization
  - Hive, Presto, Spark SQL

#### Languages (Nice-to-Have)
- **C++**: Performance-critical components, serving systems
- **Java/Scala**: Distributed systems, Spark jobs
- **Go**: Microservices, high-throughput serving
- **JavaScript**: Ad creative rendering, client-side tracking

#### Software Engineering Fundamentals
- Data structures and algorithms (arrays, trees, graphs, hash tables, heaps)
- Time and space complexity analysis (Big O notation)
- Object-oriented programming (OOP) principles
- Design patterns (Factory, Strategy, Observer, etc.)
- Version control (Git, code reviews)
- Testing (unit tests, integration tests, A/B testing)
- Debugging and profiling

### 2. Data Engineering & Processing

#### Big Data Technologies
- **Apache Spark**: Large-scale data processing
  - PySpark, Spark SQL, DataFrames
  - Optimizing Spark jobs (partitioning, caching, broadcast joins)
- **Hadoop Ecosystem**: HDFS, MapReduce, Hive
- **Stream Processing**: Kafka, Flink, Storm
- **Data Warehouses**: BigQuery, Snowflake, Redshift

#### ETL/ELT Pipelines
- Airflow, Luigi for workflow orchestration
- Feature pipelines (batch and real-time)
- Data quality and monitoring
- Schema evolution and migration

### 3. Databases & Storage

#### SQL Databases
- PostgreSQL, MySQL for transactional data
- Query optimization and indexing
- Normalization and schema design

#### NoSQL Databases
- **Key-Value**: Redis, Memcached (caching, feature stores)
- **Document**: MongoDB (user profiles, configurations)
- **Wide-Column**: Cassandra, HBase (time-series data, event logs)
- **Graph**: Neo4j (social graphs, knowledge graphs)

#### Data Lakes & Warehouses
- S3, GCS for raw data storage
- Parquet, ORC, Avro for columnar storage
- Data partitioning and bucketing strategies

---

## Machine Learning Expertise

### 1. ML Fundamentals

#### Supervised Learning
- **Regression**: Linear, Ridge, Lasso, Elastic Net
- **Classification**: Logistic Regression, SVM, Decision Trees
- **Ensemble Methods**: Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost)
- **Neural Networks**: Feedforward, backpropagation, activation functions

#### Unsupervised Learning
- **Clustering**: K-means, DBSCAN, Hierarchical
- **Dimensionality Reduction**: PCA, t-SNE, UMAP
- **Anomaly Detection**: Isolation Forest, One-Class SVM

#### Model Evaluation
- Metrics: Accuracy, Precision, Recall, F1, AUC-ROC, Log Loss
- Cross-validation, train/validation/test splits
- Calibration (Platt scaling, isotonic regression)
- Online evaluation (A/B testing, multi-armed bandits)

### 2. Deep Learning

#### Architectures
- **Feedforward Networks**: MLPs for tabular data
- **Embedding Layers**: For categorical features (user_id, item_id)
- **CNNs**: Image ads, visual features
- **RNNs/LSTMs/GRU**: Sequential user behavior modeling
- **Transformers/Attention**: User interest modeling, BERT for text ads

#### Specialized Architectures for Ads
- [**Wide & Deep**](../../ML-Implementations/ads/wdl.py): Joint training for memorization + generalization
- **DeepFM**: Factorization machines + deep learning
- [**DCN (Deep & Cross Network)**](../../ML-Implementations/ads/dcn.py): Explicit feature crosses
- [**DIN (Deep Interest Network)**](../../ML-Implementations/ads/din.py): Attention over user behavior
- [**DIEN**](../../ML-Implementations/ads/dien.py): Interest evolution modeling
- [**MMoE**](../../ML-Implementations/ads/mmoe.py)/[**PLE**](../../ML-Implementations/ads/ple.py): Multi-task learning (CTR + CVR)
- **AutoInt**: Self-attention for feature interactions

#### Training Techniques
- Optimizers: SGD, [Adam](../../ML-Implementations/optimizers/adam.py), AdaGrad, FTRL
- Regularization: L1/L2, Dropout, Batch Normalization
- Learning rate scheduling and warm-up
- Gradient clipping and mixed precision training
- Handling class imbalance (focal loss, weighted loss)

### 3. Feature Engineering

#### Feature Types
- **Categorical**: User ID, Item ID, Category, Location
  - One-hot encoding, hashing, embeddings
  - Handling high cardinality
- **Numerical**: Price, CTR history, Time features
  - Normalization (min-max, z-score)
  - Binning and discretization
- **Sequential**: User behavior history, ad impression sequences
  - Recency, frequency analysis
  - Session-based features

#### Feature Crosses & Interactions
- Manual crosses (user_gender × item_category)
- Automatic feature learning (DeepFM, DCN)
- Polynomial features
- Statistical features (mean, std, quantiles)

#### Feature Selection
- Importance ranking (SHAP, permutation importance)
- Correlation analysis
- Forward/backward selection
- Regularization-based selection (Lasso)

### 4. Optimization & Ranking

#### Learning to Rank (LTR)
- **Pointwise**: Predict relevance score independently
- **Pairwise**: RankNet, LambdaRank (compare pairs)
- **Listwise**: ListNet, ListMLE (optimize entire list)

#### Multi-Objective Optimization
- Balancing CTR, CVR, revenue, user experience
- Pareto optimization
- Scalarization and weighted objectives

#### Causal Inference
- Counterfactual reasoning
- Treatment effect estimation
- Propensity scoring
- Instrumental variables

---

## Domain Knowledge

### 1. Computational Advertising Fundamentals

#### Ad Types & Formats
- **Search Ads**: Text ads on search engines (Google Ads)
- **Display Ads**: Banner ads, native ads (GDN, Facebook Audience Network)
- **Video Ads**: Pre-roll, mid-roll, rewarded video (YouTube, TikTok)
- **Shopping Ads**: Product listings (Google Shopping, Amazon)
- **Social Ads**: Feed ads, stories, reels (Instagram, Facebook, LinkedIn)
- **App Install Ads**: User acquisition campaigns

#### Auction Mechanisms
- **First-Price Auction**: Highest bid wins, pays their bid
- **Second-Price Auction**: Highest bid wins, pays second-highest bid (Vickrey)
- **Generalized Second-Price (GSP)**: Used by Google AdWords
- **VCG (Vickrey-Clarke-Groves)**: Truthful mechanism
- **Header Bidding**: Programmatic auctions

#### Pricing Models
- **CPM (Cost Per Mille)**: Cost per 1000 impressions
- **CPC (Cost Per Click)**: Pay per click
- **CPA/CPI (Cost Per Action/Install)**: Pay per conversion
- **CPV (Cost Per View)**: Video ads pricing
- **eCPM (Effective CPM)**: Normalized revenue metric

### 2. Key Prediction Tasks

#### CTR Prediction (Click-Through Rate)
- Binary classification: click or no click
- Typical CTR: 1-10% (varies by platform)
- Metrics: AUC-ROC, Log Loss, Calibration
- Challenges: Data imbalance, position bias

#### CVR Prediction (Conversion Rate)
- Binary classification: conversion or no conversion
- Much sparser than CTR (0.1-5%)
- **Sample Selection Bias**: Only observe conversions after clicks
- Solutions: ESMM (Entire Space Multi-task Model), delayed feedback modeling

#### pCTR/pCVR (Predicted Probabilities)
- Calibrated probability estimation
- Used in eCPM calculation: `eCPM = pCTR × pCVR × bid × 1000`

#### Bid Optimization
- Budget allocation across campaigns
- Real-time bid adjustment
- ROI/ROAS maximization

### 3. Metrics & KPIs

#### User Engagement
- **CTR**: Clicks / Impressions
- **Dwell Time**: Time spent on ad/landing page
- **Bounce Rate**: Users leaving immediately
- **Viewability**: Ad was actually seen

#### Revenue Metrics
- **Revenue**: Total ad revenue
- **eCPM**: Effective revenue per 1000 impressions
- **ARPU (Average Revenue Per User)**
- **Fill Rate**: Ads served / ad requests

#### Advertiser Metrics
- **ROI (Return on Investment)**: (Revenue - Cost) / Cost
- **ROAS (Return on Ad Spend)**: Revenue / Ad Spend
- **CPA (Cost Per Acquisition)**
- **LTV (Lifetime Value)**

#### Platform Health
- **User Satisfaction**: Surveys, retention
- **Ad Load**: % of screen with ads
- **Diversity**: Variety of ads shown
- **Quality Score**: Relevance, landing page quality

### 4. Privacy & Compliance

#### Privacy Regulations
- **GDPR** (Europe): User consent, right to be forgotten
- **CCPA** (California): Data disclosure, opt-out
- **COPPA**: Children's privacy protection

#### Privacy-Preserving Technologies
- **Differential Privacy**: Adding noise to protect individual data
- **Federated Learning**: Training without centralizing data
- **On-Device Learning**: Models run on user devices
- **Privacy Sandbox**: Chrome's cookie replacement (Topics API, FLEDGE)

#### Ad Policies
- Prohibited content (illegal, harmful, deceptive)
- Restricted content (alcohol, gambling, politics)
- Editorial standards and quality guidelines

---

## System Design & Engineering

### 1. Large-Scale ML Systems

#### Offline Training Pipeline
- **Data Collection**: Event logs, impressions, clicks, conversions
- **Feature Engineering**: Batch feature computation (Spark)
- **Model Training**: Distributed training (TensorFlow, PyTorch distributed)
- **Model Validation**: Offline metrics, backtesting
- **Model Registry**: Versioning, metadata tracking (MLflow, Kubeflow)

#### Online Serving Pipeline
- **Feature Store**: Real-time + batch features (Feast, Tecton)
- **Model Serving**: Low-latency inference (TensorFlow Serving, TorchServe, Triton)
- **Caching**: Redis for hot features/predictions
- **Load Balancing**: Distribute traffic across servers
- **Monitoring**: Latency, throughput, error rates

#### Real-Time ML
- **Online Learning**: Incremental model updates
- **Streaming Features**: Kafka → Flink → Feature Store
- **Low-Latency Inference**: <10ms p99 latency
- **Model Staleness**: Handling delayed labels

### 2. Ad Serving Architecture

#### Request Flow
1. **User Request**: User loads page/app
2. **Ad Request**: Client requests ads from ad server
3. **Retrieval**: Candidate generation (100s-1000s ads)
4. **Ranking**: ML models score candidates
5. **Auction**: Bid ranking, winner determination
6. **Rendering**: Serve ad creative to user
7. **Logging**: Record impression, clicks, conversions

#### Components
- **Ad Server**: Orchestrates request flow
- **Candidate Generator**: Retrieve relevant ads (collaborative filtering, embedding-based retrieval)
- **Ranker**: ML model for CTR/CVR prediction
- **Auction**: Bid evaluation and winner selection
- **Pacing**: Budget management and delivery smoothing
- **Creative Server**: Serve ad images/videos
- **Pixel/SDK**: Track user events

#### Performance Requirements
- **Latency**: <50ms end-to-end (p99)
- **Throughput**: 100K+ QPS (queries per second)
- **Availability**: 99.99% uptime
- **Scalability**: Handle traffic spikes (Super Bowl, Black Friday)

### 3. Experimentation & A/B Testing

#### A/B Testing Framework
- **Randomization**: User-level, request-level
- **Stratification**: Ensure balanced groups
- **Traffic Splitting**: Control vs treatment (50/50, 90/10)
- **Guardrail Metrics**: Prevent degradation

#### Statistical Methods
- **Hypothesis Testing**: t-test, chi-square, Mann-Whitney U
- **Sample Size Calculation**: Power analysis
- **Multiple Testing Correction**: Bonferroni, FDR
- **Sequential Testing**: Early stopping, always-valid p-values

#### Advanced Experimentation
- **Multi-Armed Bandits**: Explore-exploit tradeoff (ε-greedy, UCB, Thompson Sampling)
- **Contextual Bandits**: Personalized exploration
- **Interleaving**: Compare ranking algorithms
- **Switchback Experiments**: Time-based randomization for marketplace effects

### 4. Monitoring & Debugging

#### Model Monitoring
- **Performance Drift**: AUC, log loss over time
- **Data Drift**: Feature distribution shifts
- **Prediction Drift**: Output distribution changes
- **Label Drift**: Ground truth changes

#### System Monitoring
- **Latency**: p50, p95, p99 response times
- **Error Rates**: 4xx, 5xx errors
- **Resource Utilization**: CPU, memory, GPU
- **Data Freshness**: Feature lag, model staleness

#### Debugging Techniques
- **Shadow Mode**: Run new model alongside production
- **Canary Deployment**: Gradual rollout (1% → 10% → 50% → 100%)
- **Feature Attribution**: SHAP, LIME for explainability
- **Ablation Studies**: Remove features/components to measure impact

---

## Soft Skills

### 1. Communication
- **Cross-Functional Collaboration**: Work with product, sales, business teams
- **Technical Writing**: Design docs, model cards, runbooks
- **Presentations**: Explain complex ML to non-technical stakeholders
- **Mentorship**: Guide junior engineers/scientists

### 2. Business Acumen
- **Metric-Driven**: Connect ML work to business impact (revenue, user growth)
- **Trade-Offs**: Balance technical elegance with business needs
- **Prioritization**: Focus on high-impact projects
- **Advertiser Perspective**: Understand customer pain points

### 3. Problem-Solving
- **First Principles Thinking**: Break down complex problems
- **Iterative Approach**: Start simple, add complexity as needed
- **Debugging Mindset**: Systematic root cause analysis
- **Experimentation**: Data-driven decision making

### 4. Continuous Learning
- **Research Papers**: Stay current with ML/ads research (NeurIPS, KDD, RecSys, WWW)
- **Industry Trends**: Follow Google Ads, Meta Ads, Amazon Ads innovations
- **Open Source**: Contribute to/learn from projects
- **Side Projects**: Build and deploy ML systems

---

## Career Progression

### Entry Level (L3/E3): Ads ML Engineer I
**Focus**: Learn systems, ship features, build foundations
- Implement feature engineering pipelines
- Train and evaluate baseline models
- Run A/B tests with guidance
- Debug production issues
- **Typical Companies**: Google, Meta, Amazon, Snap, Pinterest

### Mid Level (L4/E4): Ads ML Engineer II
**Focus**: Ownership of models/systems, measurable impact
- Own end-to-end ML pipelines
- Improve CTR/CVR models (1-5% lift)
- Design and run experiments independently
- Mentor junior engineers
- **Typical Companies**: All major tech, unicorn startups (Uber, Airbnb, DoorDash)

### Senior (L5/E5): Senior Ads ML Engineer/Scientist
**Focus**: Technical leadership, cross-team impact
- Lead major projects (new ranking systems, multi-task models)
- Drive 5-10%+ metric improvements
- Design system architecture
- Influence product roadmap
- **Typical Companies**: All major tech, technical lead at startups

### Staff+ (L6/L7): Staff/Principal Ads ML Engineer/Scientist
**Focus**: Org-level impact, technical vision
- Define technical strategy for ads org
- Novel research/systems with industry impact
- Mentor senior engineers, build teams
- External visibility (papers, talks, open source)
- **Typical Companies**: FAANG, large tech companies, VP/Head at startups

### Executive (L8+): Director/VP of Ads ML
**Focus**: Business + technical strategy, org building
- Own ads ML org (100+ people)
- Set long-term technical vision
- Partner with exec team on business strategy
- Represent company in industry (conferences, partnerships)

---

## Learning Resources

### Books
- **Ads & Auctions**: "Internet Advertising and the Generalized Second-Price Auction" (Edelman et al.)
- **ML for Ads**: "Real-Time Bidding" (Jun Wang et al.)
- **Deep Learning**: "Deep Learning" (Goodfellow, Bengio, Courville)
- **System Design**: "Designing Data-Intensive Applications" (Martin Kleppmann)

### Courses
- **Stanford CS329S**: Machine Learning Systems Design
- **Stanford CS246**: Mining Massive Datasets
- **Coursera**: Recommender Systems Specialization
- **Fast.ai**: Practical Deep Learning

### Papers (Must-Read)
- Wide & Deep Learning (Google, 2016)
- Deep & Cross Network (Google, 2017)
- Deep Interest Network (Alibaba, 2018)
- Multi-gate Mixture of Experts (Google, 2018)
- Entire Space Multi-Task Model (Alibaba, 2018)

### Conferences
- **RecSys**: ACM Conference on Recommender Systems
- **KDD**: Knowledge Discovery and Data Mining
- **CIKM**: Conference on Information and Knowledge Management
- **WWW**: The Web Conference
- **AdKDD**: Workshop on Targeting and Ranking for Online Advertising

### Industry Blogs
- Google Ads Research Blog
- Meta Engineering Blog (Ads section)
- Amazon Ads Blog
- Uber Engineering Blog
- DoorDash Engineering Blog

---

## Interview Preparation

### Coding (LeetCode Style)
- **Difficulty**: Medium to Hard
- **Topics**: Arrays, strings, hash tables, trees, graphs, dynamic programming
- **Companies**: All tech companies test coding fundamentals
- **Practice**: 100-200 LeetCode problems

### ML System Design
- Design a CTR prediction system
- Build an ad ranking pipeline
- Create a real-time bidding system
- Multi-task learning for CTR + CVR
- **Resources**: "Machine Learning System Design Interview" (Ali Aminian)

### ML Algorithms/Theory
- Explain gradient boosting vs random forest
- How does FTRL optimizer work?
- Derive logistic regression loss function
- Handle class imbalance in CTR prediction
- Calibration techniques for probability estimates

### Behavioral/Leadership
- Tell me about a time you improved a metric by X%
- How do you prioritize ML projects?
- Describe a failed experiment and what you learned
- How do you communicate technical work to non-technical partners?

### Domain-Specific
- How would you measure ad quality?
- Explain the difference between CTR and CVR prediction
- How do you handle delayed conversions?
- What's sample selection bias in ads ML?
- Design an auction mechanism for video ads

---

## Summary

Becoming an Ads Engineer/Scientist requires:
1. ✅ **Strong ML fundamentals** - Deep learning, optimization, evaluation
2. ✅ **Engineering excellence** - Scalable systems, low-latency serving
3. ✅ **Domain expertise** - Auctions, CTR/CVR, advertiser needs
4. ✅ **Business impact** - Metric-driven, revenue-focused
5. ✅ **Continuous learning** - Research, experimentation, iteration

**Start Here**:
1. Master Python + ML frameworks (TensorFlow/PyTorch)
2. Build end-to-end ML projects (Kaggle, personal projects)
3. Learn SQL and big data tools (Spark)
4. Understand ads fundamentals (CTR, auctions, eCPM)
5. Read key papers (Wide & Deep, DCN, DIN)
6. Practice coding (LeetCode) + ML system design
7. Apply to entry-level ML roles at ad-tech companies

Good luck! 🚀
