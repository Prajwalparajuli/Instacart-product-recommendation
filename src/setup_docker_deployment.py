"""
Setup Docker Deployment for Hugging Face Spaces

This script automates the Docker deployment setup process:
1. Ensures deployment folder is ready
2. Creates Dockerfile
3. Creates .dockerignore
4. Updates README.md with Docker SDK config
5. Initializes git repository
6. Provides next steps

Run from project root:
    python src/setup_docker_deployment.py
"""

from pathlib import Path
import subprocess
import sys

print("=" * 70)
print("🐳 Docker Deployment Setup for Hugging Face Spaces")
print("=" * 70)

# Configuration
PROJECT_ROOT = Path(".") if (Path(".") / "app").exists() else Path("..")
DEPLOY_DIR = PROJECT_ROOT / "hf-deployment"

# Step 1: Check deployment folder
print("\n📁 Step 1: Checking deployment folder...")
if not DEPLOY_DIR.exists():
    print("   ❌ Deployment folder not found!")
    print("   Please run: python src/prepare_deployment.py")
    sys.exit(1)

if not (DEPLOY_DIR / "app_streamlit.py").exists():
    print("   ❌ app_streamlit.py not found in deployment folder!")
    print("   Please run: python src/prepare_deployment.py")
    sys.exit(1)

print(f"   ✅ Found: {DEPLOY_DIR}")

# Step 2: Create Dockerfile
print("\n🐳 Step 2: Creating Dockerfile...")
dockerfile_content = """FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 7860

# Environment variables
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=7860", "--server.address=0.0.0.0"]
"""

dockerfile_path = DEPLOY_DIR / "Dockerfile"
dockerfile_path.write_text(dockerfile_content.strip(), encoding='utf-8')
print(f"   ✅ Created: {dockerfile_path}")

# Step 3: Create .dockerignore
print("\n🚫 Step 3: Creating .dockerignore...")
dockerignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info

# Virtual environments
venv/
env/
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

# Git
.git/
.gitignore
.gitattributes

# Streamlit
.streamlit/secrets.toml

# Development
*.ipynb
notebooks/
tests/
docs/
README_DEV.md
"""

dockerignore_path = DEPLOY_DIR / ".dockerignore"
dockerignore_path.write_text(dockerignore_content.strip(), encoding='utf-8')
print(f"   ✅ Created: {dockerignore_path}")

# Step 4: Update README.md for Docker SDK
print("\n📝 Step 4: Updating README.md for Docker SDK...")
readme_path = DEPLOY_DIR / "README.md"

if readme_path.exists():
    content = readme_path.read_text(encoding='utf-8')
    
    # Check if header already has Docker config
    if "sdk: docker" in content:
        print("   ℹ️  README.md already configured for Docker")
    else:
        # Replace or add header
        if "---" in content:
            # Find and replace existing header
            parts = content.split("---", 2)
            if len(parts) >= 3:
                # Replace middle part
                new_header = """
title: Instacart Recommendation System
emoji: 🛒
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
"""
                content = f"---{new_header}---{parts[2]}"
        else:
            # Add header at the beginning
            new_header = """---
title: Instacart Recommendation System
emoji: 🛒
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

"""
            content = new_header + content
        
        readme_path.write_text(content, encoding='utf-8')
        print(f"   ✅ Updated: {readme_path}")
else:
    print("   ⚠️  README.md not found, skipping...")

# Step 5: Check git initialization
print("\n🔧 Step 5: Checking Git initialization...")
git_dir = DEPLOY_DIR / ".git"

if git_dir.exists():
    print("   ℹ️  Git already initialized")
    print("   Current remotes:")
    try:
        result = subprocess.run(
            ["git", "remote", "-v"],
            cwd=DEPLOY_DIR,
            capture_output=True,
            text=True,
            check=False
        )
        if result.stdout:
            print("   " + "\n   ".join(result.stdout.strip().split("\n")))
        else:
            print("   (No remotes configured)")
    except Exception as e:
        print(f"   ⚠️  Could not check remotes: {e}")
else:
    print("   ℹ️  Git not initialized yet (will do in next steps)")

# Step 6: Calculate sizes
print("\n📦 Step 6: Checking deployment size...")
try:
    total_size = sum(f.stat().st_size for f in DEPLOY_DIR.rglob('*') if f.is_file())
    size_mb = total_size / (1024**2)
    print(f"   Total size: {size_mb:.2f} MB")
    
    if size_mb > 5000:
        print("   ⚠️  WARNING: Size exceeds 5 GB HF Spaces limit!")
    elif size_mb > 1000:
        print("   ⚠️  Large deployment - may take longer to build")
    else:
        print("   ✅ Size is acceptable for HF Spaces")
except Exception as e:
    print(f"   ⚠️  Could not calculate size: {e}")

# Step 7: Verify critical files
print("\n🔍 Step 7: Verifying deployment files...")
critical_files = {
    "app_streamlit.py": "Main application",
    "Dockerfile": "Docker configuration",
    "requirements.txt": "Python dependencies",
    "README.md": "HF Spaces description",
    "data/processed/test_candidates_demo.csv": "Demo dataset",
    "models/lightgbm_ranker/lgbm_lambdarank.txt": "LightGBM model",
}

all_good = True
for file, description in critical_files.items():
    file_path = DEPLOY_DIR / file
    if file_path.exists():
        size = file_path.stat().st_size / (1024**2)
        print(f"   ✅ {file} ({size:.2f} MB) - {description}")
    else:
        print(f"   ❌ MISSING: {file} - {description}")
        all_good = False

if not all_good:
    print("\n   ⚠️  Some files are missing. Please run: python src/prepare_deployment.py")

# Summary
print("\n" + "=" * 70)
print("✅ Docker deployment setup complete!")
print("=" * 70)

print("\n📋 Files created:")
print(f"   • {dockerfile_path}")
print(f"   • {dockerignore_path}")
print(f"   • {readme_path} (updated)")

print("\n" + "=" * 70)
print("🚀 Next Steps - Deploy to Hugging Face Spaces")
print("=" * 70)

print("\n1️⃣  Create Space on Hugging Face:")
print("   → Go to: https://huggingface.co/new-space")
print("   → Name: instacart-recommender (or your choice)")
print("   → SDK: Docker")
print("   → Visibility: Public or Private")
print("   → Click 'Create Space'")

print("\n2️⃣  Initialize Git (if not already done):")
print(f"   cd {DEPLOY_DIR.absolute()}")
print("   git init")

print("\n3️⃣  Add all files:")
print("   git add .")
print('   git commit -m "Initial Docker deployment"')

print("\n4️⃣  Connect to Hugging Face Space:")
print("   git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME")
print("   (Replace YOUR_USERNAME and SPACE_NAME with your values)")

print("\n5️⃣  Push to deploy:")
print("   git branch -M main")
print("   git push -u origin main")

print("\n6️⃣  Monitor deployment:")
print("   → Go to your Space URL")
print("   → Click 'Logs' tab to see build progress")
print("   → Build takes ~5-10 minutes")
print("   → Your app will be live!")

print("\n" + "=" * 70)
print("🧪 Optional: Test locally with Docker")
print("=" * 70)
print(f"\n   cd {DEPLOY_DIR.absolute()}")
print("   docker build -t instacart-rec .")
print("   docker run -p 7860:7860 instacart-rec")
print("   # Open browser to http://localhost:7860")

print("\n" + "=" * 70)
print("📚 Documentation")
print("=" * 70)
print("   See docs/DOCKER_DEPLOYMENT_GUIDE.md for detailed guide")

print("\n" + "=" * 70)
print("💡 Pro Tips:")
print("   • Test locally before deploying")
print("   • Check 'Logs' tab on HF Spaces if build fails")
print("   • Free tier includes 2 CPU, 16 GB RAM - perfect for this app!")
print("   • Your Space URL will be: https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME")
print("=" * 70)
