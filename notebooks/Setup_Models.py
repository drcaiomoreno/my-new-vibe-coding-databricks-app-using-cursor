# Databricks notebook source
# MAGIC %md
# MAGIC # Setup Models for London Housing App
# MAGIC 
# MAGIC Run this notebook ONCE before deploying the app to generate data and train models.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Update the path below to match your repo location

# COMMAND ----------

import os

# UPDATE THIS PATH TO YOUR REPO LOCATION
REPO_PATH = "/Workspace/Repos/<your-username>/my-new-vibe-coding-databricks-app-using-cursor"

# Change to repo directory
os.chdir(REPO_PATH)
print(f"✅ Working directory: {os.getcwd()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Generate Dataset

# COMMAND ----------

print("Generating London housing dataset...")
exec(open('data/generate_london_housing_data.py').read())
print("✅ Data generation complete!")

# Verify data file exists
import os
if os.path.exists('data/london_housing_data.csv'):
    import pandas as pd
    df = pd.read_csv('data/london_housing_data.csv')
    print(f"✅ Dataset created: {len(df)} records")
    display(df.head())
else:
    print("❌ Data file not found!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Train Model

# COMMAND ----------

print("Training ML model...")
exec(open('model/train_model.py').read())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Verify Model Files

# COMMAND ----------

import os
import glob

model_files = glob.glob('model/*.pkl')
if model_files:
    print("✅ Model files created:")
    for f in model_files:
        size = os.path.getsize(f) / (1024*1024)
        print(f"  • {f} ({size:.1f} MB)")
else:
    print("❌ No model files found!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Test Model Loading

# COMMAND ----------

import joblib

try:
    # Try to load model
    if os.path.exists('model/random_forest.pkl'):
        model = joblib.load('model/random_forest.pkl')
        print("✅ Random Forest model loaded successfully!")
    elif os.path.exists('model/gradient_boosting.pkl'):
        model = joblib.load('model/gradient_boosting.pkl')
        print("✅ Gradient Boosting model loaded successfully!")
    
    # Load preprocessors
    preprocessors = joblib.load('model/preprocessors.pkl')
    print("✅ Preprocessors loaded successfully!")
    
    print("\n🎉 Setup complete! Your app is ready to run.")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ Done!
# MAGIC 
# MAGIC Now you can deploy or restart your Databricks App:
# MAGIC 
# MAGIC 1. Go to **Apps** in Databricks
# MAGIC 2. Find your app or create new one
# MAGIC 3. Deploy from your repo with `app.yaml`
# MAGIC 4. The app should now work!

