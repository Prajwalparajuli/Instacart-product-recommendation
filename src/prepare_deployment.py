"""
Prepare Hugging Face Spaces Deployment Folder

This script creates a separate 'hf-deployment/' folder with only the files
needed for Hugging Face Spaces deployment, leaving your original project
intact for GitHub commits.

Run this script from the project root:
    python src/prepare_deployment.py
"""

import shutil
from pathlib import Path
import json

print("=" * 70)
print("🚀 Preparing Hugging Face Spaces Deployment")
print("=" * 70)

# Configuration
# Support running from both project root and src/ directory
PROJECT_ROOT = Path(".") if (Path(".") / "app").exists() else Path("..")
DEPLOY_DIR = PROJECT_ROOT / "hf-deployment"

# Files and folders to copy
COPY_STRUCTURE = {
    "app/app_streamlit.py": "app_streamlit.py",  # Move to root for HF Spaces
    "app/pages/": "pages/",
    "app/theme.css": "theme.css",
    
    "models/lightgbm_ranker/": "models/lightgbm_ranker/",
    
    "data/processed/test_candidates_demo.csv": "data/processed/test_candidates_demo.csv",
    "data/processed/als_user_factors.npy": "data/processed/als_user_factors.npy",
    "data/processed/als_product_factors.npy": "data/processed/als_product_factors.npy",
    "data/processed/item_item_similarity_score.csv": "data/processed/item_item_similarity_score.csv",
    "data/processed/catalog_with_dept_images.parquet": "data/processed/catalog_with_dept_images.parquet",
    
    "data/raw/orders.csv": "data/raw/orders.csv",
    "data/raw/order_products__prior.csv": "data/raw/order_products__prior.csv",
    "data/raw/products.csv": "data/raw/products.csv",
    "data/raw/departments.csv": "data/raw/departments.csv",
    "data/raw/aisles.csv": "data/raw/aisles.csv",
    
    "assets/": "assets/",
}

# Step 1: Clean/Create deployment directory
print("\n📁 Step 1: Setting up deployment directory")
if DEPLOY_DIR.exists():
    print(f"   ⚠️  Removing existing: {DEPLOY_DIR}")
    shutil.rmtree(DEPLOY_DIR)

DEPLOY_DIR.mkdir()
print(f"   ✅ Created: {DEPLOY_DIR}")

# Step 2: Copy files
print("\n📋 Step 2: Copying files...")
total_size = 0

for source_path, dest_path in COPY_STRUCTURE.items():
    source = PROJECT_ROOT / source_path
    dest = DEPLOY_DIR / dest_path
    
    if not source.exists():
        print(f"   ⚠️  SKIP (not found): {source_path}")
        continue
    
    # Create parent directory
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if source.is_dir():
        shutil.copytree(source, dest)
        # Calculate size
        size = sum(f.stat().st_size for f in dest.rglob('*') if f.is_file())
        total_size += size
        print(f"   ✅ {source_path} → {dest_path} ({size / (1024**2):.2f} MB)")
    else:
        shutil.copy2(source, dest)
        size = dest.stat().st_size
        total_size += size
        print(f"   ✅ {source_path} → {dest_path} ({size / (1024**2):.2f} MB)")

# Step 3: Create minimal requirements.txt
print("\n📦 Step 3: Creating minimal requirements.txt")
requirements_minimal = """# Streamlit App Dependencies (Inference Only)
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.3
implicit==0.7.2
lightgbm==4.3.0
pyarrow==15.0.0
Pillow==10.2.0
"""

requirements_path = DEPLOY_DIR / "requirements.txt"
requirements_path.write_text(requirements_minimal.strip())
print(f"   ✅ Created: requirements.txt")
total_size += requirements_path.stat().st_size

# Step 4: Create README for HF Spaces
print("\n📝 Step 4: Creating README.md")
readme_content = """---
title: Instacart Recommendation System
emoji: 🛒
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.31.0
app_file: app_streamlit.py
pinned: false
---

# 🛒 Instacart Recommendation System

A hybrid recommendation system combining three powerful approaches:
- **Item-Item Collaborative Filtering**: Find similar products based on purchase patterns
- **ALS Matrix Factorization**: Personalized recommendations using latent factors
- **LightGBM LambdaRank**: Learning-to-rank with 30+ engineered features

## Features
✨ Real-time recommendations across 49K+ products
🎯 Purchase history integration
🛍️ Interactive shopping cart
📊 Model insights and explanations

## Demo Users
Try these user IDs: `101`, `318`, `451`, `999`, `3678`, `5890`, `45678`, `56789`, `67890`, `89012`

## Tech Stack
- **Frontend**: Streamlit
- **Models**: LightGBM, Implicit (ALS), Custom CF
- **Data**: Instacart Market Basket Analysis dataset

---
Built with ❤️ using Streamlit
"""

readme_path = DEPLOY_DIR / "README.md"
readme_path.write_text(readme_content.strip(), encoding='utf-8')
print(f"   ✅ Created: README.md")
total_size += readme_path.stat().st_size

# Step 5: Create .gitignore for deployment repo
print("\n🚫 Step 5: Creating .gitignore")
gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml
"""

gitignore_path = DEPLOY_DIR / ".gitignore"
gitignore_path.write_text(gitignore_content.strip())
print(f"   ✅ Created: .gitignore")

# Step 6: Create .streamlit/config.toml
print("\n⚙️  Step 6: Creating Streamlit config")
streamlit_dir = DEPLOY_DIR / ".streamlit"
streamlit_dir.mkdir(exist_ok=True)

config_content = """[theme]
primaryColor = "#667eea"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#1a1d29"
textColor = "#fafafa"
font = "sans serif"

[server]
headless = true
port = 7860
enableCORS = false
enableXsrfProtection = false
"""

config_path = streamlit_dir / "config.toml"
config_path.write_text(config_content.strip())
print(f"   ✅ Created: .streamlit/config.toml")

# Step 7: Summary
print("\n" + "=" * 70)
print("✅ Deployment folder ready!")
print("=" * 70)

print(f"\n📦 Total Size: {total_size / (1024**2):.2f} MB")

print("\n📂 Deployment Structure:")
print(f"   {DEPLOY_DIR}/")
print("   ├── app_streamlit.py          # Main app (moved to root)")
print("   ├── pages/                    # Additional pages")
print("   ├── theme.css                 # Styling")
print("   ├── models/                   # LightGBM model")
print("   ├── data/                     # Demo data + matrices")
print("   ├── assets/                   # Thumbnails")
print("   ├── requirements.txt          # Minimal dependencies")
print("   ├── README.md                 # HF Spaces description")
print("   ├── .gitignore                # Git ignore rules")
print("   └── .streamlit/config.toml    # Streamlit config")

print("\n🔍 Verification:")
# Check critical files
critical_files = [
    "app_streamlit.py",
    "requirements.txt",
    "README.md",
    "models/lightgbm_ranker/lgbm_lambdarank.txt",
    "data/processed/test_candidates_demo.csv",
]

all_good = True
for file in critical_files:
    file_path = DEPLOY_DIR / file
    if file_path.exists():
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ MISSING: {file}")
        all_good = False

if all_good:
    print("\n✨ All critical files present!")
else:
    print("\n⚠️  Some files are missing. Check the output above.")

print("\n" + "=" * 70)
print("📝 Next Steps:")
print("=" * 70)
print("\n1. Test locally:")
print(f"   cd {DEPLOY_DIR}")
print("   streamlit run app_streamlit.py")
print("\n2. Create HF Space:")
print("   - Go to https://huggingface.co/new-space")
print("   - Choose: Streamlit SDK")
print("   - Upload the entire hf-deployment/ folder")
print("\n3. Or use Git:")
print(f"   cd {DEPLOY_DIR}")
print("   git init")
print("   git add .")
print('   git commit -m "Initial deployment"')
print("   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME")
print("   git push -u origin main")

print("\n💡 Tips:")
print("   - Your original project remains unchanged")
print("   - Commit original project to GitHub as usual")
print("   - Deploy hf-deployment/ folder to HF Spaces")
print("   - Re-run this script anytime to update deployment")

print("\n" + "=" * 70)
