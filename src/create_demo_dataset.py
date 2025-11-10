"""
Create Demo Dataset for Hugging Face Spaces Deployment

This script creates a reduced version of test_candidates.csv
with only 15-20 demo users for deployment.

Run this ONCE before deploying to Hugging Face Spaces:
    python src/create_demo_dataset.py
"""

import pandas as pd
from pathlib import Path

print("=" * 60)
print("Creating Demo Dataset for HF Spaces Deployment")
print("=" * 60)

# Configuration
# Support running from both project root and src/ directory
if (Path(".") / "data").exists():
    INPUT_FILE = Path("data/processed/test_candidates.csv")
    OUTPUT_FILE = Path("data/processed/test_candidates_demo.csv")
else:
    INPUT_FILE = Path("../data/processed/test_candidates.csv")
    OUTPUT_FILE = Path("../data/processed/test_candidates_demo.csv")

# Demo users - matching the actual users in orders.csv
DEMO_USERS = [1, 2, 3, 5, 8, 10, 12, 15, 20, 25]

print(f"\nConfiguration:")
print(f"  Input file: {INPUT_FILE}")
print(f"  Output file: {OUTPUT_FILE}")
print(f"  Demo users: {len(DEMO_USERS)} users")
print(f"  User IDs: {DEMO_USERS[:10]}... (showing first 10)")

# Check if input file exists
if not INPUT_FILE.exists():
    print(f"\n❌ ERROR: Input file not found: {INPUT_FILE}")
    print("   Please ensure test_candidates.csv exists in data/processed/")
    exit(1)

# Load full dataset
print(f"\n📂 Loading full dataset...")
try:
    df_full = pd.read_csv(INPUT_FILE)
    print(f"   ✅ Loaded: {len(df_full):,} rows, {df_full.shape[1]} columns")
    print(f"   File size: {INPUT_FILE.stat().st_size / (1024**2):.2f} MB")
except Exception as e:
    print(f"   ❌ ERROR loading file: {e}")
    exit(1)

# Get unique users in dataset
unique_users = df_full['user_id'].unique()
print(f"\n👥 Total users in dataset: {len(unique_users):,}")

# Filter for demo users that exist in the dataset
existing_demo_users = [u for u in DEMO_USERS if u in unique_users]
missing_users = [u for u in DEMO_USERS if u not in unique_users]

if missing_users:
    print(f"\n⚠️  Warning: {len(missing_users)} demo users not found in dataset:")
    print(f"   {missing_users}")
    print(f"   Will use {len(existing_demo_users)} available users")

if len(existing_demo_users) == 0:
    print(f"\n❌ ERROR: None of the demo users exist in the dataset!")
    print(f"   Available user IDs: {list(unique_users[:20])} (showing first 20)")
    print(f"\n💡 Suggestion: Update DEMO_USERS list in this script with valid user IDs")
    exit(1)

# Create demo dataset
print(f"\n🔄 Filtering for {len(existing_demo_users)} demo users...")
df_demo = df_full[df_full['user_id'].isin(existing_demo_users)].copy()

# Calculate statistics
rows_original = len(df_full)
rows_demo = len(df_demo)
reduction_pct = (1 - rows_demo / rows_original) * 100

print(f"\n📊 Dataset Statistics:")
print(f"   Original rows: {rows_original:,}")
print(f"   Demo rows: {rows_demo:,}")
print(f"   Reduction: {reduction_pct:.2f}%")
print(f"   Avg candidates per user: {rows_demo / len(existing_demo_users):,.0f}")

# Save demo dataset
print(f"\n💾 Saving demo dataset...")
try:
    df_demo.to_csv(OUTPUT_FILE, index=False)
    print(f"   ✅ Saved to: {OUTPUT_FILE}")
    
    # Show file sizes
    size_original_mb = INPUT_FILE.stat().st_size / (1024**2)
    size_demo_mb = OUTPUT_FILE.stat().st_size / (1024**2)
    size_reduction = (1 - size_demo_mb / size_original_mb) * 100
    
    print(f"\n📦 File Size Comparison:")
    print(f"   Original: {size_original_mb:,.2f} MB")
    print(f"   Demo: {size_demo_mb:,.2f} MB")
    print(f"   Reduction: {size_reduction:.2f}%")
    print(f"   Space saved: {size_original_mb - size_demo_mb:,.2f} MB")
    
except Exception as e:
    print(f"   ❌ ERROR saving file: {e}")
    exit(1)

# Show sample data
print(f"\n📋 Demo Dataset Preview:")
print(f"   Users included: {sorted(existing_demo_users)}")
print(f"\n   Sample rows:")
print(df_demo.head(3).to_string(index=False))

# Final instructions
print("\n" + "=" * 60)
print("✅ Demo dataset created successfully!")
print("=" * 60)
print("\n📝 Next Steps:")
print("   1. Update app/app_streamlit.py:")
print("      Change line ~158:")
print("      CANDIDATES_PATH = Path('data/processed/test_candidates_demo.csv')")
print("\n   2. Test the app locally:")
print("      streamlit run app/app_streamlit.py")
print("\n   3. Verify these demo users work in the app:")
print(f"      {existing_demo_users}")
print("\n   4. Deploy to Hugging Face Spaces!")
print("      (See HUGGINGFACE_DEPLOYMENT_GUIDE.md for details)")
print("\n" + "=" * 60)
