# ML Engineer Learning Path - 6 Month Sprint (UPDATED)

## Windows 11 + WSL, Crypto + Nutrition Focus

---

## Phase 1: Foundations (Weeks 1-3)

**Goal**: Get comfortable with Python data manipulation and ML thinking  
**Timeline**: Move fast, these are prerequisites not portfolio pieces

### Core Skills

- **Python fundamentals for ML**: virtual environments, imports, package structure
- **NumPy & Pandas deep dive**: indexing, groupby, merges, reshaping, handling missing data
- **Statistics**: distributions, hypothesis testing, correlation, p-values, Bayesian thinking
  - *Concepts*: Mean/median/mode, standard deviation, normal distribution, sampling

### Quick Learning Projects (Not Portfolio)

1. **EDA on Crypto historical data** (2-3 hours)
   - Fetch 5 years of Bitcoin data from CoinGecko
   - Calculate returns, volatility, drawdowns
   - Visualize trends, correlations
   - *Concepts touched*: Data loading, cleaning, exploratory analysis, basic statistics

2. **EDA on Nutrition dataset** (2-3 hours)
   - Load 5,000 foods from USDA FoodData Central
   - Analyze macro/micronutrient distributions
   - Find correlations between nutrients
   - *Concepts touched*: API integration, data normalization, statistical relationships

### Deliverables

- GitHub repo with clean Python structure (setup.py, requirements.txt, notebooks)
- Two small Jupyter notebooks showing EDA work
- **Portfolio status**: Skip these, they're warmup

---

## Phase 2: ML Fundamentals (Weeks 4-7)

**Goal**: Build your first real ML models using Scikit-learn  
**Timeline**: 4 weeks, starting portfolio work

### Core Frameworks & Concepts

- **Scikit-learn mastery**:
  - *Concepts*: train/test split, cross-validation, hyperparameter tuning, overfitting
  - Classification: logistic regression, decision trees, random forests, SVM
  - Regression: linear regression, ridge/lasso, ensemble methods
  - Clustering: K-means, hierarchical clustering
  - Feature engineering: scaling, encoding, polynomial features, feature selection

- **Model evaluation**:
  - *Concepts*: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, MSE/RMSE/MAE

### **START: Crypto Mega-Project - Part 1**

**"Crypto Market Intelligence System"** (PORTFOLIO PIECE)

**Week 4-5: Crypto Price Prediction & Market Classification**

- Fetch Bitcoin + Ethereum daily data (5 years) from CoinGecko
- Feature engineering:
  - Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands
  - Price-based: returns, volatility, momentum
  - Volume-based: volume ratio, price-volume trend
  - *Concepts*: Feature scaling (StandardScaler, MinMaxScaler), feature selection (correlation, variance threshold)
  
- Build multiple Scikit-learn models:
  - **Regression**: Predict next-day closing price (Linear Regression, Ridge, Lasso, Random Forest)
  - **Classification**: Predict up/down movement (Logistic Regression, Decision Tree, Random Forest)
  - **Clustering**: Group cryptocurrencies by market behavior (K-means on normalized returns)
  - *Concepts*: Train/test split, k-fold cross-validation, hyperparameter grid search, regularization

- Model evaluation & comparison
  - *Concepts*: ROC-AUC, precision-recall tradeoff, learning curves, overfitting detection

**Deliverables for Week 4-5**:

- Jupyter notebook with full pipeline: data → features → models → evaluation
- Model performance comparison table
- Feature importance visualization
- **Portfolio status**: YES - Polish this, add to GitHub

---

## Phase 3: Deep Learning Basics (Weeks 8-10)

**Goal**: Understand neural networks, move beyond Scikit-learn  
**Timeline**: 3 weeks

### Core Frameworks & Concepts

- **PyTorch fundamentals**:
  - *Concepts*: Tensors, autograd, gradients, backpropagation
  - Neural network architecture: layers, activation functions, loss functions
  - Training loop: forward pass, loss calculation, backward pass, optimization
  - *Algorithms*: SGD, Adam optimizer, ReLU, sigmoid, softmax

- **Time-series specific**:
  - *Concepts*: Stationarity, autocorrelation, lag features, rolling windows
  - *Architectures*: LSTM, GRU (recurrent neural networks)

### **CONTINUE: Crypto Mega-Project - Part 2**

**"Crypto Market Intelligence System"** (PORTFOLIO PIECE)

**Week 8-9: Time-Series Forecasting with Neural Networks**

- Build LSTM model to predict Bitcoin price movement
  - *Concepts*: Sequence-to-sequence, hidden state, time-step selection
  - Data preparation: rolling windows, normalization for neural networks
  - Model architecture: 2-3 LSTM layers + dense layers
  - *Algorithms*: LSTM, Adam optimizer, dropout regularization
  
- Compare LSTM vs Scikit-learn baseline
  - *Concepts*: Underfitting vs overfitting, learning curves, early stopping

**Week 9-10: Sentiment Analysis on Crypto News**

- Scrape/collect crypto news headlines (Twitter, Reddit, news APIs)
- Build basic sentiment classifier using PyTorch
  - *Concepts*: Text preprocessing (tokenization, stopwords), embedding layers
  - Architecture: Embedding → LSTM/GRU → Dense classifier
  - *Algorithms*: Embedding layers, RNN, cross-entropy loss

- Combine sentiment with price data
  - *Concepts*: Feature engineering across modalities, lagged features

**Deliverables for Week 8-10**:

- Jupyter notebook with LSTM implementation
- Sentiment analysis notebook with PyTorch
- Comparison: LSTM vs traditional models performance
- Time-series visualization with predictions
- **Portfolio status**: YES - Keep all this, it's impressive

---

## Phase 4: Recommendation Systems & Advanced Models (Weeks 11-13)

**Goal**: Build real recommendation engines, introduce advanced ML concepts  
**Timeline**: 3 weeks

### Recommendation Systems Concepts

- *Concepts*: Collaborative filtering, content-based filtering, hybrid approaches
- *Algorithms*: Matrix factorization, cosine similarity, user/item embeddings, ranking
- *Evaluation*: Precision@K, recall@K, NDCG, coverage, diversity

### **CONTINUE: Crypto Mega-Project - Part 3**

**"Crypto Market Intelligence System"** (PORTFOLIO PIECE)

**Week 11-12: Token Recommendation Engine**

- Build multiple recommendation approaches:
  - **Content-based**: Recommend similar cryptocurrencies based on technical features (correlation, market cap trends, volatility profile)
    - *Algorithms*: Cosine similarity, KNN
  - **Collaborative filtering**: Recommend tokens based on "users" (traders) with similar portfolio behavior
    - *Algorithms*: User-user CF, item-item CF, matrix factorization
  - **Hybrid**: Combine both approaches
    - *Concepts*: Ensemble methods for recommendations

- Input: A user's current holdings → Output: Top-5 tokens to buy
- Evaluate recommendations:
  - *Concepts*: Precision@K, recall@K, coverage, diversity metrics

**Week 12-13: Feature Engineering & Advanced Preprocessing**

- Advanced feature engineering:
  - *Concepts*: Polynomial features, interaction terms, domain-specific features
  - On-chain metrics if available (addresses, transaction volume, etc.)
  - Temporal features: seasonality, trend, cyclical encoding

- Experiment with different feature sets and their impact on all models
  - *Concepts*: Ablation studies, feature importance

**Deliverables for Week 11-13**:

- Recommendation engine notebook with all 3 approaches
- Performance comparison table
- Example recommendations with explanations
- **Portfolio status**: YES - This is your flagship project so far

---

## Phase 5: Production & Deployment (Weeks 14-16)

**Goal**: Ship real code, not just notebooks  
**Timeline**: 3 weeks

### Core Technologies

- **Docker**: Images, containers, volumes, docker-compose
- **FastAPI**: REST endpoints, request validation, error handling
- **Model serving**: Loading models, inference, caching
- **CI/CD**: GitHub Actions, automated testing, model versioning

### **FINALIZE: Crypto Mega-Project - Part 4 + START: Nutrition Mega-Project**

**Week 14: Containerize & Serve Crypto System**

- Refactor Crypto notebooks into production Python modules
  - *Concepts*: Code organization, error handling, logging
  
- Build FastAPI application:
  - Endpoint 1: `POST /predict` - Predict next-day price movement
  - Endpoint 2: `POST /recommend` - Get token recommendations
  - Endpoint 3: `GET /sentiment` - Get latest sentiment score
  - *Concepts*: REST API design, request/response validation (Pydantic)

- Create Docker image, test locally
  - *Concepts*: Docker best practices, environment variables, multi-stage builds

- Set up model versioning & experiment tracking
  - *Concepts*: MLflow or Weights & Biases for tracking hyperparameters, metrics, artifacts

**Week 15-16: START Nutrition Mega-Project - Part 1 + CI/CD**

**Week 15: CI/CD Pipeline for Crypto System**

- GitHub Actions workflow:
  - Lint code (flake8, black)
  - Run unit tests on models
  - Build Docker image on every push
  - Log metrics to MLflow
  - *Concepts*: Automated testing, continuous integration, reproducibility

**Week 16: Nutrition Meal Recommendation System - Part 1**
**"Personal Nutrition Intelligence System"** (PORTFOLIO PIECE)

- Fetch USDA FoodData Central API (~10,000 foods, 28 nutrients each)
- Feature engineering:
  - Macro ratios (protein %, carb %, fat %)
  - Micronutrient density (nutrients per calorie)
  - Functional categories (vitamins, minerals, phytochemicals)
  - Cost estimates (if available)
  - *Concepts*: Normalization for different scales, handling missing data

- Build initial models using Scikit-learn:
  - **Regression**: Predict nutritional value from ingredient composition
  - **Classification**: Classify foods by diet type (vegan, keto, paleo, etc.)
  - *Algorithms*: Random Forest, SVM, logistic regression

**Deliverables for Week 14-16**:

- Crypto system: Fully containerized FastAPI app in Docker
- GitHub repo with CI/CD pipeline (GitHub Actions)
- MLflow tracking dashboard (local)
- Nutrition project: Initial data exploration + Scikit-learn models
- **Portfolio status**: Crypto system = POLISHED PORTFOLIO PIECE. Nutrition = Work in progress

---

## Phase 6: Advanced Techniques, Transformers, C++ Integration & Future Directions (Weeks 17-24)

**Goal**: Production ML, fine-tuned Transformers, C++ inference, AND future ML research directions  
**Timeline**: 8 weeks

### Advanced Concepts

- *Concepts*: Transfer learning, fine-tuning, ensemble methods, attention mechanisms
- *Algorithms*: Transformers, graph neural networks (optional), advanced time-series (Prophet)
- *Frameworks*: TensorFlow basics, Hugging Face Transformers, XGBoost, ONNX, PyTorch model export
- **LeCun-Inspired Directions**: Energy-based models, world models, model predictive control, joint embeddings

### **CONTINUE & FINISH: Nutrition Mega-Project**

**Week 17-18: Neural Networks for Nutrition**

- Build PyTorch models:
  - **Meal recommendation neural network**: Embed foods, learn similarity in latent space
    - *Concepts*: Embedding layers, similarity learning, metric learning, joint embeddings
  - **Dietary classification**: Multi-label classification (foods can fit multiple diets)
    - *Algorithms*: Multi-label sigmoid, BCEWithLogitsLoss
  
- Advanced recommendation approaches:
  - **Content-based with embeddings**: Learn food embeddings that capture nutritional similarity
    - *Concepts*: Representation learning, latent factors, joint embedding spaces (LeCun approach)
  - **Collaborative filtering lite**: Group foods by nutritional profile, recommend within groups
    - *Algorithms*: K-means clustering on embeddings, similarity ranking

**Week 19-20: Transformers & Fine-Tuning for Domain Tasks**

- **Crypto News Classification**: Fine-tune a small Transformer model on crypto news
  - Use Hugging Face Transformers library (DistilBERT or ALBERT for efficiency)
  - Fine-tune on crypto sentiment/category classification
  - *Concepts*: Transfer learning, pre-trained embeddings, fine-tuning heads
  - *Algorithms*: Attention mechanisms, tokenization with special tokens

- **Nutrition Recipe Understanding**: Fine-tune for ingredient extraction or recipe classification
  - Fine-tune on recipe data for named entity recognition or multi-label classification
  - *Concepts*: Domain-specific fine-tuning, handling variable-length inputs
  - *Algorithms*: Transformer encoders, classification heads

- Model comparison: Custom LSTM vs Fine-tuned Transformer on same task
  - Metrics: Accuracy, F1, inference speed
  - *Concepts*: When to use transfer learning vs training from scratch

**Week 20-21: Energy-Based Models & Anomaly Detection (LeCun Future Direction)**

- **Concept**: Energy-based models assign low energy to normal patterns, high energy to anomalies
  - *Why*: LeCun argues this is more fundamental than generative/discriminative distinction
  
- **Crypto anomaly detection**:
  - Train an energy function on normal market conditions using autoencoders or contrastive learning
  - Use it to detect unusual price movements, volume spikes, volatility breaks
  - *Algorithms*: Autoencoder energy scoring, one-class SVM baseline
  - Compare to traditional statistical anomaly detection (z-score, isolation forest)
  - *Concepts*: What makes something "anomalous," learned vs heuristic approaches

- **Nutrition outlier detection**:
  - Learn energy function over normal nutritional profiles
  - Flag foods with unusual nutrient combinations
  - *Concepts*: Latent space energy scoring, regularized methods (LeCun's recommendation)

**Week 21-22: World Models & Model Predictive Control (LeCun Future Direction)**

- **World Model Concept**: Learn to predict future states from current observations
  - *Why*: LeCun argues this scales better than RL for planning
  
- **Crypto price dynamics world model**:
  - Train a model to predict multi-step ahead: given current price/volume → predict next N steps
  - Learn latent representation of market state
  - *Algorithms*: Variational autoencoder (VAE) or latent dynamics model
  - *Concepts*: Stochasticity in predictions, planning with learned models

- **Model Predictive Control approach**:
  - Use learned world model to plan: "What actions maximize returns?"
  - Compare to pure RL approach (which LeCun discourages as primary)
  - *Concepts*: Planning in learned latent space, when to use MPC vs RL

- **Nutrition meal planning optimization**:
  - Learn how meals combine nutritionally (interactions, synergies)
  - Optimize meal plans using world model predictions (energy balance, nutrient bioavailability)
  - *Concepts*: Multi-step planning, optimization in latent space

---

## Phase 6 (Continued): Production ML & C++ Integration (Weeks 22-24)

**Week 22: Production ML & C++ Integration**

- **Model Optimization & Export**:
  - Convert PyTorch models to ONNX format (framework-agnostic)
  - Quantization: INT8 quantization for faster inference
  - Model pruning and distillation basics
  - *Concepts*: Model size reduction, latency vs accuracy tradeoff

- **C++ Inference Pipeline** (relevant to your job):
  - Export a sentiment model and nutrition classifier to C++
  - Use ONNX Runtime or torch::jit for C++ inference
  - Build inference wrapper that matches your job's use case
  - *Concepts*: Stateless inference servers, batch processing, error handling
  - *Skills*: Reading C++ model serving code, understanding deployment constraints
  - Load fine-tuned Transformers in C++ (use onnxruntime or cpp bindings)

- **Refactor to Production Modules**:
  - Nutrition notebooks → production Python modules
  - Create inference-only modules (separate from training)
  - Build FastAPI endpoints that use the optimized models

- **Build FastAPI endpoints**:
  - Endpoint 1: `POST /recommend-meal` - Get meal recommendations for goals
  - Endpoint 2: `POST /analyze-recipe` - Analyze nutritional content + extract ingredients (uses fine-tuned NER)
  - Endpoint 3: `GET /meal-plan` - Generate multi-day meal plan (uses world model optimization)
  - Endpoint 4: `POST /classify-sentiment` - Classify crypto news sentiment (fine-tuned Transformer)
  - Endpoint 5: `POST /detect-anomaly` - Detect market anomalies (energy-based model)
  - *Concepts*: Stateless API design, model loading/caching, concurrent inference

- Containerize with Docker
- Add model versioning with MLflow (track fine-tuning experiments, world model training)

**Week 23-24: Polish & Production Deployment**

- **Crypto System**:
  - Add monitoring/dashboards (optional: Grafana or Prometheus)
  - Document API thoroughly (README, API docs)
  - Add example notebooks showing usage
  - Include C++ inference examples and benchmarks
  - Document energy-based anomaly detector performance

- **Nutrition System**:
  - Same polish as Crypto
  - Ensure all models are exported and optimized for C++
  - Document fine-tuning process (reproducibility)
  - Include performance benchmarks (Python vs C++ inference speed)
  - Document world model planning workflow

- **Learning Documentation**:
  - Write a brief guide: "When to fine-tune vs train from scratch"
  - Document ONNX export workflow for future projects
  - Include C++ integration patterns you learned
  - Write summary: "Energy-based models vs generative models" (LeCun perspective)
  - Document world model approach for time-series prediction

**Deliverables for Week 17-24**:

- Nutrition system: Fully containerized FastAPI app with all endpoints (including world model planning)
- Both Crypto & Nutrition: Polished GitHub repos with:
  - Clean code, proper documentation
  - CI/CD pipelines running
  - Example usage notebooks
  - Performance benchmarks (inference speed, model size, anomaly detection accuracy)
  - ONNX model exports for C++ integration
  - Energy-based anomaly detection results
  - World model predictions and planning examples
- C++ inference examples showing how to use exported models and fine-tuned Transformers
- Optional: Simple web frontend for one system (React, Vue, or Streamlit)
- Research notes: How do energy-based models compare to your current generative approach?

**Portfolio status**: BOTH systems = PRODUCTION-READY PORTFOLIO PIECES with C++ integration, cutting-edge research directions, AND practical industry applications

---

## Week 25+ (If You Have Time)

### Optional Advanced Topics

- **Advanced C++ optimization**: Profile inference code, optimize batch sizes, GPU inference
- **XGBoost & LightGBM**: Gradient boosting for structured data (crypto/nutrition)
- **Distributed inference**: Multi-GPU or multi-model serving
- **Model compression**: Knowledge distillation for Transformer models
- **Kubernetes concepts**: Understand orchestration basics

### Advanced Learning (LeCun-Inspired Research Directions)

If you want to explore future ML directions:

- **Joint Embedding Spaces**: Build models where different modalities (price, sentiment, on-chain) map to same embedding space
- **Regularized Methods**: Replace contrastive learning with regularization-based approaches for embeddings
- **Advanced World Models**: Learn dynamics models with uncertainty quantification, use for robust planning
- **Energy-Based Ranking**: Use energy-based models for ranking/recommendations instead of scoring
- **Self-Supervised Learning**: Pre-train models on unlabeled crypto/nutrition data using energy-based approaches
- These are research-level, optional but impressive if explored

### Optional Third Project

If you want to build momentum:

- **Stock market analysis** (using Alpha Vantage API) with energy-based anomaly detection
- **Multi-asset portfolio optimizer** (crypto + stocks + real estate) using world models and MPC
- This would be lighter than the first two, mostly showing you can apply patterns quickly and incorporate LeCun directions
