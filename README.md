# Instacart Product Recommendation System

A machine learning project that builds a product recommendation system using the Instacart Market Basket Analysis dataset. The system predicts which products a user is likely to reorder based on their historical purchase patterns.

## Project Overview

This project implements a product recommendation system using the Instacart Market Basket Analysis dataset. The system predicts which products a user is likely to reorder based on their historical purchase patterns.

**Current Implementation Approach:**
- **Collaborative Filtering**: Alternating Least Squares (ALS) matrix factorization for user-product affinity
- **Item-Item Similarity**: Cosine similarity between products based on purchase co-occurrence  
- **Feature Engineering**: User, product, and interaction features for supervised learning
- **Feature Aggregation**: Combining ALS scores with item-item similarity metrics

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
│   ├── Item_Item.ipynb        # Item-based collaborative filtering
│   └── QDA.ipynb             # Quadratic Discriminant Analysis experiments
├── src/
│   ├── ingest.py             # Data loading with YAML configuration
│   ├── validate.py           # Data integrity checks
│   ├── build_features.py     # Feature engineering pipeline
│   ├── als_build_interactions.py # ALS interaction matrix preparation
│   ├── similarity_agg_als_score.py # ALS+similarity feature aggregation
│   ├── eval.py               # Model evaluation metrics (stub)
│   └── utlis.py              # Configuration management and utilities
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
   - **YAML-based configuration management system**
   - **Centralized utility functions for config loading and path management**
   - Project structure setup

3. **Feature Engineering Pipeline**
   - User behavior features (order frequency, recency, seasonality)
   - Product features (popularity, department/aisle statistics)
   - User-product interaction features (reorder probability, cart position)
   - **ALS interaction matrix preparation for collaborative filtering**
   - **ALS+similarity feature aggregation combining multiple signals**

### In Progress

1. **Model Implementation**
   - ALS (Alternating Least Squares) collaborative filtering ✓ (interaction matrix ready)
   - Item-item similarity computation ✓ (similarity aggregation implemented)
   - Supervised learning pipeline integration

2. **Model Training & Evaluation**
   - QDA (Quadratic Discriminant Analysis) experiments
   - Gradient boosting model implementation (LightGBM/XGBoost)
   - Model evaluation metrics and validation framework

### Planned

1. **Production Pipeline**
   - Model serving infrastructure
   - Streamlit web interface implementation
   - Performance optimization and caching

2. **Model Enhancements**
   - Hyperparameter optimization for gradient boosting models
   - Cross-validation and model selection framework
   - A/B testing simulation for recommendation evaluation

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
   # Load and validate data using YAML configuration
   python src/ingest.py
   python src/build_features.py
   ```

3. **Model Training** 
   ```bash
   # Build ALS interaction matrix
   python src/als_build_interactions.py
   
   # Generate combined ALS + similarity features  
   python src/similarity_agg_als_score.py
   ```

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

## Dependencies

- **Core**: pandas, numpy, scipy, scikit-learn
- **ML Models**: implicit (ALS), lightgbm, xgboost
- **Visualization**: matplotlib, seaborn
- **Configuration**: pyyaml for YAML config management
- **Utilities**: pathlib for path handling

## Contributing

This is a learning project. Feel free to explore the code and suggest improvements!

## License

This project is for educational purposes. The Instacart dataset has its own terms of use.