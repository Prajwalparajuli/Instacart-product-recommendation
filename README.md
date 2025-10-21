# Instacart Product Recommendation System

A machine learning project that builds a product recommendation system using the Instacart Market Basket Analysis dataset. The system predicts which products a user is likely to reorder based on their historical purchase patterns.

## Project Overview

This project implements multiple recommendation approaches including:
- **Collaborative Filtering**: Using Alternating Least Squares (ALS) matrix factorization
- **Content-Based Filtering**: Product similarity and user preference modeling
- **Feature Engineering**: Advanced user, product, and interaction features
- **Ensemble Methods**: Combining multiple models for improved performance

## Dataset

The project uses the Instacart Market Basket Analysis dataset, which contains:
- **Orders**: 3.4M+ orders from 200K+ users
- **Products**: 50K+ products across 134 aisles and 21 departments
- **Order Products**: Line-item details for prior and training orders
- **Metadata**: Product names, aisles, and department information

## Project Structure

```
insta-rec/
├── data/
│   ├── raw/                    # Original CSV files (excluded from git)
│   ├── interim/               # Intermediate processed data
│   └── processed/             # Final feature-engineered datasets
├── notebooks/
│   ├── 01_eda_instacart.ipynb # Comprehensive exploratory data analysis
│   └── Item_Item.ipynb        # Item-based collaborative filtering
├── src/
│   ├── ingest.py             # Data loading and basic preprocessing
│   ├── validate.py           # Data integrity checks
│   ├── build_features.py     # Feature engineering pipeline
│   ├── als_build_interactions.py # ALS model data preparation
│   ├── similarity_agg_als_score.py # Model scoring and aggregation
│   ├── qda_feature_extraction.py # QDA-based feature extraction
│   ├── eval.py               # Model evaluation metrics
│   └── utlis.py              # Utility functions
├── models/                   # Trained model artifacts
├── reports/                  # Analysis reports and visualizations
├── conf.yaml                # Configuration parameters
├── requirements.txt          # Python dependencies
└── app_streamlit.py         # Web demo interface (planned)
```

## Implementation Progress

### Completed

1. **Data Exploration & Analysis**
   - Comprehensive EDA covering user behavior, product demand, and reorder patterns
   - Temporal analysis (day of week, hour of day, purchase cycles)
   - Product popularity analysis across departments, aisles, and individual items
   - User engagement and retention analysis

2. **Data Infrastructure**
   - Data validation and integrity checks
   - Efficient data loading with optimized dtypes
   - Configuration management system
   - Project structure setup

3. **Feature Engineering Foundation**
   - User behavior features (order frequency, recency, seasonality)
   - Product features (popularity, department/aisle statistics)
   - Interaction features (reorder probability, cart position)

### In Progress

1. **Model Implementation**
   - ALS (Alternating Least Squares) collaborative filtering
   - Item-item similarity models
   - Feature extraction for supervised learning

2. **Evaluation Framework**
   - Recommendation quality metrics
   - Model comparison and validation

### Planned

1. **Advanced Models**
   - Gradient boosting (LightGBM/XGBoost) for ranking
   - Deep learning approaches
   - Ensemble methods

2. **Production Pipeline**
   - Model serving infrastructure
   - Streamlit web interface
   - Performance optimization

## Key Insights from EDA

- **Temporal Patterns**: Strong weekly cycles with peak orders on Sundays/Mondays
- **Product Categories**: Fresh produce dominates (~40% of orders)
- **User Behavior**: Power law distribution - few heavy users, many casual shoppers
- **Reorder Patterns**: High reorder rates for staples like bananas, dairy, and produce

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

1. **Exploratory Data Analysis**
   ```bash
   # Open and run the EDA notebook
   jupyter notebook notebooks/01_eda_instacart.ipynb
   ```

2. **Data Processing**
   ```bash
   python src/ingest.py
   python src/build_features.py
   ```

3. **Model Training** (when implemented)
   ```bash
   python src/als_build_interactions.py
   python src/similarity_agg_als_score.py
   ```

## Configuration

Key parameters are managed in `conf.yaml`:
- Data paths and directories
- ALS model hyperparameters (factors, regularization, iterations)
- Feature engineering settings

## Dependencies

- **Core**: pandas, numpy, scipy, scikit-learn
- **ML Models**: implicit (ALS), lightgbm, xgboost
- **Visualization**: matplotlib, seaborn
- **Utilities**: pyyaml for configuration

## Contributing

This is a learning project. Feel free to explore the code and suggest improvements!

## License

This project is for educational purposes. The Instacart dataset has its own terms of use.