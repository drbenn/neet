# ML Engineer Learning Path - 6 Month Sprint
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
- Build basic sentiment classifier
  - *Concepts*: Text preprocessing (tokenization, stopwords), bag-of-words
  - Use pre-trained embeddings or build simple neural network classifier
  - *Algorithms*: Embedding layers, CNN for text, or simple dense networks

- Combine sentiment with price data
  - *Concepts*: Feature engineering across modalities, lagged features

**Deliverables for Week 8-10**:
- Jupyter notebook with LSTM implementation
- Sentiment analysis notebook
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

## Phase 6: Advanced Techniques & Final Portfolio (Weeks 17-24)
**Goal**: Deepen expertise, build second polished portfolio piece  
**Timeline**: 8 weeks (this is where you really shine)

### Advanced Concepts
- *Concepts*: Transfer learning, fine-tuning, ensemble methods, attention mechanisms
- *Algorithms*: Transformers, graph neural networks (optional), advanced time-series (Prophet)
- *Frameworks*: TensorFlow basics, Hugging Face Transformers, XGBoost

### **CONTINUE & FINISH: Nutrition Mega-Project**

**Week 17-18: Neural Networks for Nutrition**
- Build PyTorch models:
  - **Meal recommendation neural network**: Embed foods, learn similarity in latent space
    - *Concepts*: Embedding layers, similarity learning, metric learning
  - **Dietary classification**: Multi-label classification (foods can fit multiple diets)
    - *Algorithms*: Multi-label sigmoid, BCEWithLogitsLoss
  
- Advanced recommendation approaches:
  - **Content-based with embeddings**: Learn food embeddings that capture nutritional similarity
    - *Concepts*: Representation learning, latent factors
  - **Collaborative filtering lite**: Group foods by nutritional profile, recommend within groups
    - *Algorithms*: K-means clustering, similarity ranking

**Week 19-20: NLP for Recipes & Advanced Features**
- Optional: Scrape recipe data (AllRecipes, etc.) or use public recipe datasets
- NLP tasks:
  - **Recipe understanding**: Extract ingredients, cooking methods, tags
    - *Concepts*: Named entity recognition (NER), text parsing
  - **Recipe recommendation**: Recommend recipes matching nutritional goals + cuisine preferences
    - *Concepts*: Multi-modal learning, ranking algorithms

- Advanced feature engineering:
  - Seasonal availability of ingredients
  - Prep time vs nutritional density
  - Allergen risk scoring
  - *Concepts*: Domain-specific feature construction

**Week 21-22: Production & Serving**
- Refactor Nutrition notebooks → production modules
- Build FastAPI endpoints:
  - Endpoint 1: `POST /recommend-meal` - Get meal recommendations for goals (weight loss, muscle gain, etc.)
  - Endpoint 2: `POST /analyze-recipe` - Analyze nutritional content of custom recipe
  - Endpoint 3: `GET /meal-plan` - Generate multi-day meal plan
  - *Concepts*: Stateless API design, caching, lazy loading

- Containerize with Docker
- Add model versioning with MLflow
- Set up CI/CD pipeline (same as Crypto)

**Week 23-24: Polish & Portfolio Finalization**

- **Crypto System**:
  - Add monitoring/dashboards (optional: Grafana or Prometheus)
  - Document API thoroughly (README, API docs)
  - Add example notebooks showing usage
  - Deploy somewhere accessible (local Docker for demo, or free tier cloud if feeling ambitious)

- **Nutrition System**:
  - Same polish as Crypto
  - Ensure both systems are deployment-ready
  - Create comparison document: what you learned from each

**Deliverables for Week 17-24**:
- Nutrition system: Fully containerized FastAPI app with all endpoints
- Both Crypto & Nutrition: Polished GitHub repos with:
  - Clean code, proper documentation
  - CI/CD pipelines running
  - Example usage notebooks
  - Performance benchmarks
- Optional: Simple web frontend for one system (React, Vue, or just Streamlit)

**Portfolio status**: BOTH systems = PRODUCTION-READY PORTFOLIO PIECES

---

## Week 25+ (If You Have Time)

### Optional Advanced Topics
- **TensorFlow basics**: Compare to PyTorch, understand both
- **XGBoost & LightGBM**: Gradient boosting for structured data (revisit crypto/nutrition with these)
- **Fine-tuning large models**: Use Hugging Face to fine-tune a small LLM on crypto news or recipe reviews
- **Graph neural networks**: Model relationships between cryptocurrencies or food ingredients
- **Kubernetes concepts**: Understand orchestration (don't need to implement)

### Optional Third Project
If you want to build momentum, consider a quick third project combining learnings:
- **Stock market analysis** (using Alpha Vantage API)
- **Multi-asset portfolio optimizer** (crypto + stocks + real estate risk modeling)
- This would be lighter than the first two, mostly showing you can apply patterns quickly

---

## Your Portfolio Strategy

### GitHub Repos to Create & Polish
1. **crypto-intelligence-system** (PRIMARY PORTFOLIO)
   - Data pipeline: CoinGecko API → cleaned data
   - Scikit-learn models: regression, classification, clustering
   - PyTorch models: LSTM for forecasting
   - Sentiment analysis on news
   - Recommendation engine (3 approaches)
   - FastAPI application with Docker
   - CI/CD pipeline
   - README with results, architecture diagram, how to run

2. **nutrition-intelligence-system** (PRIMARY PORTFOLIO)
   - Same structure as crypto
   - USDA FoodData Central integration
   - Dietary classification + meal recommendations
   - Recipe analysis (if you go that route)
   - FastAPI application with Docker
   - CI/CD pipeline

### Quick Learning Projects (DON'T PORTFOLIO)
- Week 1-3 EDA notebooks (reference only, don't share)
- Individual model experiments that got superseded

### What to Emphasize in Interviews
"I built two end-to-end ML systems from data to production:
- **Crypto system**: Handles time-series forecasting, NLP sentiment analysis, and recommendation engines, deployed with FastAPI + Docker + CI/CD
- **Nutrition system**: Demonstrates recommendation systems, multi-modal learning, and production ML ops
- Both systems use real-time/continuously updated data, real models across Scikit-learn/PyTorch, and are production-ready"

This is MUCH stronger than "I did Kaggle competitions" or "I trained MNIST"

---

## Windows 11 + WSL Setup Notes

### Tools You'll Need
```
Python 3.10+
pip, virtualenv
Git
Docker Desktop (runs in WSL2)
VS Code + WSL remote extension
Jupyter Lab (for development)
```

### Development Workflow
```bash
# In WSL terminal
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn torch fastapi uvicorn docker pytest

# For each project
git clone <your-repo>
cd project
python -m pytest  # Run tests
uvicorn app:app --reload  # Run API locally
docker build -t my-ml-app .  # Build image
docker run -p 8000:8000 my-ml-app  # Run container
```

### Compute Reality
- Scikit-learn models: Instant
- PyTorch on CPU: Fast enough for learning (seconds to minutes)
- Small LSTM models (crypto): Minutes
- Full nutrition dataset: No problem
- Never going to train huge LLMs, but that's fine—you're not supposed to yet

---

## Pacing Tips for "Fast Baby"
- Don't get stuck on perfect code in weeks 1-3, move fast
- Use Jupyter notebooks for exploration, convert to .py modules for production
- Don't implement from scratch—use libraries (Scikit-learn, PyTorch, etc.)
- Aim for "works" before "perfect"
- Save polish/documentation for week 14 onward when you have full systems

---

## Key Success Metrics
By end of Week 6: Two working end-to-end ML pipelines (one crypto, starting nutrition)
By end of Week 13: Recommendation engines working on both projects
By end of Week 16: Crypto system production-ready with API + Docker + tests
By end of Week 24: Both systems fully polished and deployment-ready

If you hit these, you're in excellent shape for ML engineer interviews.