# 🛒 Instacart Product Recommendation System

A production-ready hybrid recommendation system combining **Collaborative Filtering**, **Matrix Factorization**, and **Learning-to-Rank** models. Features an interactive Streamlit web application with purchase history integration and real-time recommendations across 49K+ products.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io)

## 🎥 Live Demo

🚀 **[Try the live app on Hugging Face Spaces](https://huggingface.co/spaces/Prajwalparajuli/Shopping-recommendation-ML)**

Or run locally with 5 sample users:
```bash
streamlit run hf-deployment/app_streamlit.py
```

**Demo Users**: 3, 12, 15, 20, 25

## Project Overview

This project implements a **hybrid recommendation system** that combines three powerful machine learning approaches to predict which products users are likely to reorder based on their historical purchase patterns:

1. **Item-Item Collaborative Filtering** - Product similarity based on co-purchase patterns
2. **ALS Matrix Factorization** - Latent factor model for user-product affinities  
3. **LightGBM LambdaRank** - Learning-to-rank model with rich feature engineering

**Current Implementation Approach:**
- **Collaborative Filtering**: Alternating Least Squares (ALS) matrix factorization for user-product affinity
- **Item-Item Similarity**: Cosine similarity between products based on purchase co-occurrence  
- **Feature Engineering**: User, product, and interaction features for supervised learning
- **Feature Aggregation**: Combining ALS scores with item-item similarity metrics
- **Ranking Model**: LightGBM LambdaRank for learning-to-rank product recommendations

## Dataset

The project uses the Instacart Market Basket Analysis dataset, which contains:
- **Orders**: 3.4M+ orders from 200K+ users
- **Products**: 50K+ products across 134 aisles and 21 departments
- **Order Products**: Line-item details for prior and training orders
- **Metadata**: Product names, aisles, and department information

## Project Structure

```
insta-rec/
├── app/                          # 🎨 Streamlit Web Application
│   ├── app_streamlit.py          # Main shopping interface with hybrid recommendations
│   ├── pages/
│   │   └── 02_LightGBM_Recs.py   # Model insights dashboard
│   └── theme.css                 # Custom styling
│
├── data/
│   ├── raw/                      # Original CSV files (~680 MB)
│   │   ├── orders.csv
│   │   ├── order_products__prior.csv
│   │   ├── products.csv
│   │   ├── departments.csv
│   │   └── aisles.csv
│   ├── interim/                  # Intermediate processed data
│   └── processed/                # Final datasets (~5.6 GB)
│       ├── test_candidates.csv   # Full test set (1.2 GB)
│       ├── test_candidates_demo.csv # Demo dataset (0.22 MB) ⭐
│       ├── als_user_factors.npy
│       ├── als_product_factors.npy
│       ├── item_item_similarity_score.csv
│       └── catalog_with_dept_images.parquet
│
├── src/                          # 🔧 Source Code & Utilities
│   ├── Training Scripts:
│   │   ├── ingest.py             # Data loading with YAML configuration
│   │   ├── validate.py           # Data integrity checks
│   │   ├── build_features.py     # Feature engineering pipeline
│   │   ├── als_build_interactions.py # ALS interaction matrix
│   │   ├── similarity_agg_als_score.py # Feature aggregation
│   │   ├── catalog_enrichment.py # Product catalog processing
│   │   ├── eval.py               # Model evaluation metrics
│   │   └── utlis.py              # Config management utilities
│   │
│   └── Deployment Utilities:
│       ├── create_demo_dataset.py    # Generate demo data (10 users)
│       └── prepare_deployment.py     # Build HF Spaces deployment folder
│
├── models/                       # 🤖 Machine Learning Models
│   ├── lightgbm_ranker/          # LightGBM artifacts (~10 MB)
│   │   ├── lgbm_lambdarank.txt
│   │   └── feature_cols.json
│   ├── als.py                    # ALS matrix factorization training
│   ├── item_item.py              # Item-Item CF training
│   ├── LightGMB.py               # LightGBM LambdaRank training
│   └── qda.py                    # QDA model training
│
├── assets/                       # 🖼️ Product Images
│   └── thumbnails/               # Product thumbnails (~2 MB)
│
├── notebooks/                    # 📊 Development Notebooks
│   ├── 01_eda_instacart.ipynb    # Exploratory data analysis
│   ├── Item_Item.ipynb           # Item-based CF experiments
│   └── QDA.ipynb                 # QDA experiments
│

├── hf-deployment/                # 🚀 Hugging Face Spaces Deployment (LIVE)
│   ├── app_streamlit.py          # Main app (deployed)
│   ├── pages/
│   │   └── 02_LightGBM_Recs.py   # Model insights page
│   ├── data/                     # Demo data (5 users, ~2 MB)
│   ├── models/                   # Trained models (~60 MB)
│   ├── assets/                   # Product images (~2 MB)
│   ├── Dockerfile               # Docker configuration for HF Spaces
│   ├── .streamlit/config.toml   # Streamlit configuration
│   └── requirements.txt         # Deployment dependencies
│
├── reports/                      # 📈 Analysis Reports
├── conf.yaml                     # ⚙️ Configuration
├── requirements.txt              # 📦 Dependencies
├── PROJECT_STRUCTURE.md          # 📋 Organization guide
└── README.md                     # 📖 This file
```

## Implementation Progress

### ✅ Completed

1. **Data Exploration & Analysis**
   - Comprehensive EDA covering user behavior, product demand, and reorder patterns
   - Temporal analysis (day of week, hour of day, purchase cycles)
   - Product popularity analysis across departments, aisles, and individual items
   - User engagement and retention analysis

2. **Data Infrastructure**
   - Data validation and integrity checks
   - Efficient data loading with optimized dtypes
   - YAML-based configuration management system
   - Centralized utility functions for config loading and path management
   - Project structure setup

3. **Feature Engineering Pipeline**
   - User behavior features (order frequency, recency, seasonality)
   - Product features (popularity, department/aisle statistics)
   - User-product interaction features (reorder probability, cart position)
   - ALS interaction matrix preparation for collaborative filtering
   - ALS+similarity feature aggregation combining multiple signals
   - Product catalog enrichment with department images
5. **Production Web Application** ✓
   - **Hybrid Recommendation System** with 3 distinct recommendation engines
   - **Interactive Streamlit Interface** with modern dark mode UI/UX
   - **Purchase History Integration** showing user's previous orders ("Bought 5x before")
   - **Shopping Cart Functionality** with add/remove/checkout
   - **Product Catalog** with 49K+ products, images, and department organization
   - **Pagination System** (50 items per page, ~994 pages)
   - **Model Insights Dashboard** showing feature importance and metrics
   - **"All Picks" Combined View** showing recommendations from all three engines

6. **Deployment Infrastructure** ✓
   - **Live on Hugging Face Spaces** (deployed via Docker SDK)
   - Demo dataset with 5 users (synchronized across all data sources)
   - Docker configuration for HF Spaces deployment
   - GitHub + HF Spaces dual repository workflow
   - Complete documentation and deployment guides94 pages
   - **Model Insights Dashboard** showing feature importance and metrics
   - **Deployment Ready** for Hugging Face Spaces

6. **Deployment Infrastructure** ✓
   - Demo dataset generation (10 users, 0.22 MB)
   - Automated deployment folder preparation
   - Separate GitHub and HF Spaces repository management
   - Complete documentation and deployment guides

### 🎯 Production Features
#### Web Application Features:
- ✅ **Three Recommendation Engines**:
  - **Item-Item CF**: Product-based similarity recommendations (Similar Products)
  - **ALS Matrix Factorization**: User-based collaborative filtering (Users Like You)
  - **LightGBM Ranking**: ML-powered personalized recommendations (For You)
  - **All Picks**: Combined view showing all three recommendation types together
  
- ✅ **User Experience**:
  - Modern dark mode UI with gradient backgrounds
  - Real-time cart management with session persistence
  - Purchase history display ("⭐ Bought 5x before", "✓ Bought 2x before")
  - Responsive product cards with department-specific images
  - Tab-based navigation between recommendation types
  - Pagination for browsing large catalogs (5 columns per row)
  - Department filters and search functionality

- ✅ **Model Transparency**:
  - LightGBM Insights Dashboard with feature importance
  - Model performance metrics (NDCG@5, NDCG@10, NDCG@20)
  - Department-based filtering of recommendations
  - Top-K ranking visualization

#### Deployment Status:
- ✅ **Live on Hugging Face Spaces** at [huggingface.co/spaces/Prajwalparajuli/Shopping-recommendation-ML](https://huggingface.co/spaces/Prajwalparajuli/Shopping-recommendation-ML)
- ✅ Demo dataset with 5 users (synchronized across orders, purchase history, and LightGBM candidates)
- ✅ Docker SDK deployment (~64 MB total, optimized for HF Spaces)
- ✅ Automatic health checks and proper container configuration
- ✅ Separate GitHub and HF Spaces repository workflow
- ✅ Production-ready with error handling and fallback recommendationsflow
- ✅ One-command deployment preparation

## 🎯 Key Features

### Hybrid Recommendation System
- **Item-Item CF**: Finds similar products based on co-purchase patterns (496K similarity pairs)
- **ALS Matrix Factorization**: Personalized recommendations using 64 latent factors (206K users × 49K products)
- **LightGBM Ranking**: Learning-to-rank with 30+ engineered features (NDCG@10: 0.585)

### Interactive Web Application
- 🛒 **Smart Shopping Cart** with session persistence
- 📊 **Purchase History Integration** showing user's previous orders
- 🎨 **Modern UI/UX** with glassmorphism, gradients, and smooth animations
- 📄 **Pagination System** for browsing 49K+ products (50 items per page)
- 🔍 **Multi-Model Comparison** via tab-based navigation
- 📈 **Model Insights Dashboard** with feature importance and metrics

### Production Ready
- ⚡ **Optimized Performance** with lazy loading and caching
- 🚀 **HF Spaces Deployment** ready (~746 MB total)
- 👥 **Demo Mode** with 10 representative users
- 📦 **One-Command Deployment** preparation

## 💡 Key Insights from EDA

- **Temporal Patterns**: Strong weekly cycles with peak orders on Sundays/Mondays
- **Product Categories**: Fresh produce dominates (~40% of orders)
- **User Behavior**: Power law distribution - few heavy users, many casual shoppers
- **Reorder Patterns**: High reorder rates for staples like bananas, dairy, and produce
- **User Engagement**: Average 17 orders per user, 10.1 products per order
- **Reorder Ratio**: 59% of items are reorders (user loyalty signal)

## Model Architecture

The recommendation system implements a **multi-model approach** combining three complementary machine learning techniques:

### 1. ALS (Alternating Least Squares) Matrix Factorization

**Purpose**: Generate user-product affinity scores through collaborative filtering
- **Confidence Weighting**: Uses `confidence = 1 + alpha * purchase_count` (α=40)
- **Latent Factors**: 64-dimensional user and item embeddings
- **Regularization**: L2 penalty (λ=0.1) to prevent overfitting
- **Output**: Dot product scores for user-product pairs from historical interactions

### 2. Item-Item Collaborative Filtering

**Purpose**: Find similar products based on co-purchase patterns
- **Similarity Metric**: Cosine similarity between product purchase vectors
- **Sparse Matrix**: Efficient CSR format for memory optimization
- **Top-K Selection**: Maintains top-10 similar items per product
- **Aggregation**: Sum-based candidate scoring for multi-anchor recommendations

### 3. LightGBM LambdaRank Implementation

**Purpose**: Learn-to-rank final product recommendations
- **Ranking Objective**: Learns to rank products within each order rather than binary classification
- **Group-wise Learning**: Each order is treated as a ranking group for personalized recommendations
- **NDCG Evaluation**: Uses Normalized Discounted Cumulative Gain at multiple cutoffs (5, 10, 20)
- **Feature Engineering**: Excludes IDs to prevent memorization, focuses on behavioral patterns

**Training Strategy:**
- **Temporal Split**: Last order of each user reserved for validation
- **Early Stopping**: Prevents overfitting with 150-round patience
- **Hyperparameter Tuning**: Optimized learning rate, tree depth, and regularization

**Output:**
- Top-K product recommendations per order (default K=10)
- Ranking scores for recommendation confidence
- CSV export for easy integration with applications

## Setup and Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Prajwalparajuli/Instacart-product-recommendation.git
   cd insta-rec
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download data**
   - Download the Instacart dataset and place CSV files in `data/raw/`
   - Required files: orders.csv, products.csv, aisles.csv, departments.csv, 
     order_products__prior.csv, order_products__train.csv

5. **Run data validation**
   ```bash
   python src/validate.py
   ```

## Usage

### 🚀 Quick Start - Run the App

**Option 1: Live Demo (Recommended)**
- Visit [https://huggingface.co/spaces/Prajwalparajuli/Shopping-recommendation-ML](https://huggingface.co/spaces/Prajwalparajuli/Shopping-recommendation-ML)
- No installation required!

**Option 2: Run Locally**
```bash
# Run the deployed version (uses demo dataset)
cd hf-deployment
streamlit run app_streamlit.py
```

The app includes:
- 5 demo users with full purchase history (users: 3, 12, 15, 20, 25)
- Hybrid recommendations from 3 models (Item-Item, ALS, LightGBM)
- Interactive shopping cart with session persistence
- Product catalog with 49K+ products and department images
- Purchase history integration showing previous orders
The app includes:
- 10 demo users with full purchase history
- Hybrid recommendations from 3 models
- Interactive shopping cart
- Product catalog with 49K+ products

### 📊 Full Training Pipeline

1. **Exploratory Data Analysis**
   ```bash
   # Open and run the EDA notebook
   jupyter notebook notebooks/01_eda_instacart.ipynb
   ```

2. **Data Processing**
   ```bash
   # Load and validate data using YAML configuration
   python src/ingest.py
   python src/build_features.py
   ```

3. **Model Training** 
   ```bash
   # Build ALS interaction matrix
   python src/als_build_interactions.py
   
   # Train ALS matrix factorization model
   python models/als.py
   
   # Train item-item collaborative filtering
   python models/item_item.py
   
   # Generate combined ALS + similarity features  
   python src/similarity_agg_als_score.py
   
   # Train LightGBM LambdaRank model and generate recommendations
   python models/LightGMB.py
   ```

### 🌐 Deployment to Hugging Face Spaces

**Current Status**: ✅ **LIVE** at [huggingface.co/spaces/Prajwalparajuli/Shopping-recommendation-ML](https://huggingface.co/spaces/Prajwalparajuli/Shopping-recommendation-ML)

**Deployment Architecture**:
- **Platform**: Hugging Face Spaces with Docker SDK
- **Runtime**: Python 3.10-slim in Docker container
- **Port**: 7860 (standard Streamlit port for HF Spaces)
- **Health Check**: Automatic container health monitoring
- **Data**: Demo dataset with 5 synchronized users (~2 MB)
- **Models**: All three recommendation engines deployed (~60 MB)

**To deploy your own version**:

```bash
# 1. Create demo dataset (5 users that exist in both orders and candidates)
cd c:\Users\prajw\Desktop\Statistical and Machine Learning\Project\insta-rec
python src/create_demo_dataset.py

# 2. Copy to deployment folder
Copy-Item "data\processed\test_candidates_demo.csv" "hf-deployment\data\processed\" -Force

# 3. Test locally
cd hf-deployment
streamlit run app_streamlit.py

# 4. Deploy to HF Spaces (from hf-deployment directory)
git init
git add .
git commit -m "Deploy to Hugging Face Spaces"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push -u origin main
```

**Important Files for Deployment**:
- `Dockerfile` - Docker configuration for HF Spaces
- `.streamlit/config.toml` - Streamlit settings (headless mode, no usage stats)
- `requirements.txt` - Python dependencies
- `README.md` - Space description and metadata

**See deployment guides** in the repository for detailed instructions.
git init
git add .
git commit -m "Initial deployment"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
git push -u origin main
```

**See `docs/` folder for detailed deployment guides:**
- `DEPLOYMENT_COMPLETE.md` - Quick deployment guide
- `TWO_REPO_WORKFLOW.md` - Managing GitHub + HF Spaces
- `HUGGINGFACE_DEPLOYMENT_GUIDE.md` - Detailed instructions

## Configuration

The project uses a centralized YAML configuration system (`conf.yaml`) for managing:

- **Data Paths**: Input/output directories (raw, interim, processed)
- **File Names**: All CSV file names and generated outputs
- **Data Types**: Pandas dtypes for memory optimization
- **Model Parameters**: ALS hyperparameters (factors=128, regularization=0.05, iterations=50)
- **Feature Settings**: Feature engineering parameters and thresholds

### Configuration Usage

The configuration system is implemented in `src/utlis.py` with utilities:

```python
from utlis import load_config, get_paths, get_data_files, get_dtypes

# Load configuration
config = load_config()

# Access specific sections
paths = get_paths(config)          # Get data directories
files = get_data_files(config)     # Get file names
dtypes = get_dtypes(config)        # Get pandas dtypes
```

Scripts that use configuration:
- `src/ingest.py` - Uses configured paths, filenames, and dtypes
- `src/utlis.py` - Provides configuration management utilities

This system ensures consistency across the project and makes experimentation easier by centralizing parameter management.

## Tech Stack & Dependencies
## 📊 Model Performance

### LightGBM LambdaRank Results
- **NDCG@5**: 
- **NDCG@10**:   
- **NDCG@20**: 
- **Training**: 75K users, temporal validation split
- **Features**: 30+ engineered features (user, product, interaction)
- **Top Features**: ALS scores, item-item similarity, purchase recency, user-product interaction history

### ALS Matrix Factorization
- **Latent Factors**: 64 dimensions
- **Confidence Weighting**: α=40 for implicit feedback
- **Regularization**: λ=0.1 to prevent overfitting
- **User Factors**: 206K users × 64 dimensions
- **Product Factors**: 49K products × 64 dimensions

### Item-Item Collaborative Filtering
- **Similarity Metric**: Cosine similarity
- **Similarity Pairs**: 496K item-item pairs computed
- **Top-K**: Maintains top-10 similar items per product
- **Sparse Matrix**: Efficient CSR format for memory optimization

### Dataset Statistics
- **Orders**: 3.4M+ orders from 206K users
- **Products**: 49,688 unique products
- **Departments**: 21 categories
- **Aisles**: 134 subcategories
- **Training Data**: Prior orders + training set
- **Demo Users**: 5 users (synchronized across all data sources)
### Deployment (Minimal for HF Spaces)
```txt
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.3
implicit==0.7.2
lightgbm==4.3.0
pyarrow==15.0.0
Pillow==10.2.0
```

## 📊 Model Performance

### LightGBM LambdaRank Results
- **NDCG@5**: ~0.565
- **NDCG@10**: ~0.585  
- **NDCG@20**: ~0.605
- **Training**: 75K users, temporal validation split
- **Features**: 30+ engineered features (user, product, interaction)

### Dataset Statistics
- **Orders**: 3.4M+ orders from 206K users
- **Products**: 49,688 unique products
- **Departments**: 21 categories
- **Aisles**: 134 subcategories
- **Training Data**: Prior orders + training set
- **Similarity Pairs**: 496K item-item pairs computed

## 🗂️ Repository Organization

For detailed information about file organization and deployment:
- See `PROJECT_STRUCTURE.md` for complete folder structure
- See `docs/` for deployment guides and workflow documentation

## 🤝 Contributing

This is an educational project demonstrating production ML systems. Contributions, suggestions, and feedback are welcome!

### Areas for Improvement:
- Additional recommendation algorithms (neural collaborative filtering, graph-based)
- Real-time model updates and online learning
- A/B testing framework for recommendation quality
- Enhanced feature engineering (temporal patterns, seasonal effects)
- Hyperparameter optimization and ensemble methods

## 📄 License

This project is for educational purposes. The Instacart dataset is publicly available with its own [terms of use](https://www.instacart.com/datasets/grocery-shopping-2017).

## 🙏 Acknowledgments

- **Dataset**: Instacart Market Basket Analysis (Kaggle, 2017)
- **Models**: LightGBM, Implicit (ALS implementation)
- **Framework**: Streamlit for rapid web app development

---

**Built with ❤️ for learning and demonstration purposes**