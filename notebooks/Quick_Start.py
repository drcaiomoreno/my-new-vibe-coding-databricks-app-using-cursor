# Databricks notebook source
# MAGIC %md
# MAGIC # London Housing Price Predictor - Quick Start
# MAGIC 
# MAGIC This notebook helps you set up and test the London Housing Price Predictor app on Databricks.
# MAGIC 
# MAGIC ## Steps:
# MAGIC 1. Install dependencies
# MAGIC 2. Generate dataset
# MAGIC 3. Train model
# MAGIC 4. Test predictions
# MAGIC 5. Deploy app

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies

# COMMAND ----------

# MAGIC %pip install streamlit==1.28.1 pandas==2.1.3 numpy==1.24.3 scikit-learn==1.3.2 plotly==5.18.0 seaborn==0.13.0 matplotlib==3.8.2 joblib==1.3.2

# COMMAND ----------

# Restart Python to use newly installed packages
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Set Working Directory

# COMMAND ----------

import os

# Change this to your repo path
REPO_PATH = "/Workspace/Repos/<your-username>/my-new-vibe-coding-databricks-app-using-cursor"

# Change to repo directory
os.chdir(REPO_PATH)
print(f"Working directory: {os.getcwd()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Generate Dataset

# COMMAND ----------

# Import the data generation function
import sys
sys.path.append(os.path.join(REPO_PATH, 'data'))

from generate_london_housing_data import generate_london_housing_data

# Generate dataset
print("Generating London housing dataset...")
df = generate_london_housing_data(5000)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save dataset
df.to_csv('data/london_housing_data.csv', index=False)

print(f"✓ Generated {len(df)} housing records")
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
display(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Explore the Data

# COMMAND ----------

import pandas as pd
import plotly.express as px

# Load the data
df = pd.read_csv('data/london_housing_data.csv')

# Display summary statistics
print("Dataset Summary:")
display(df.describe())

# COMMAND ----------

# Price distribution
fig = px.histogram(df, x='price', nbins=50, title='Distribution of Property Prices')
fig.show()

# COMMAND ----------

# Average price by borough
borough_avg = df.groupby('borough')['price'].mean().sort_values(ascending=False)
fig = px.bar(x=borough_avg.index, y=borough_avg.values, 
             title='Average Property Price by Borough',
             labels={'x': 'Borough', 'y': 'Average Price (£)'})
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Train the Model

# COMMAND ----------

# Import training functions
sys.path.append(os.path.join(REPO_PATH, 'model'))
from train_model import (
    load_and_preprocess_data,
    train_random_forest,
    train_gradient_boosting,
    evaluate_model,
    save_model_and_preprocessors
)

# COMMAND ----------

print("Loading and preprocessing data...")
X_train_scaled, X_test_scaled, y_train, y_test, preprocessors, X_train, X_test = load_and_preprocess_data('data/london_housing_data.csv')

print(f"\nTraining set size: {len(X_train_scaled)}")
print(f"Test set size: {len(X_test_scaled)}")

# COMMAND ----------

# Train Random Forest
print("Training Random Forest model...")
rf_model = train_random_forest(X_train_scaled, y_train)
rf_metrics = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")

# COMMAND ----------

# Train Gradient Boosting
print("Training Gradient Boosting model...")
gb_model = train_gradient_boosting(X_train_scaled, y_train)
gb_metrics = evaluate_model(gb_model, X_test_scaled, y_test, "Gradient Boosting")

# COMMAND ----------

# Select best model
if rf_metrics['r2'] > gb_metrics['r2']:
    print("✓ Random Forest selected as best model")
    best_model = rf_model
    best_model_name = 'random_forest'
else:
    print("✓ Gradient Boosting selected as best model")
    best_model = gb_model
    best_model_name = 'gradient_boosting'

# COMMAND ----------

# Feature importance
import pandas as pd
import plotly.express as px

feature_importance = pd.DataFrame({
    'feature': preprocessors['feature_names'],
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
display(feature_importance)

# Visualize
fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
             title='Feature Importance in Price Prediction')
fig.show()

# COMMAND ----------

# Save model and preprocessors
print("Saving model and preprocessors...")
save_model_and_preprocessors(best_model, preprocessors, model_name=best_model_name)
print("✓ Model saved successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Test Predictions

# COMMAND ----------

import joblib
import numpy as np

# Load model and preprocessors
model = joblib.load(f'model/{best_model_name}.pkl')
preprocessors = joblib.load('model/preprocessors.pkl')

print("✓ Model and preprocessors loaded successfully!")

# COMMAND ----------

# Test prediction function
def predict_price(borough, property_type, bedrooms, bathrooms, square_feet,
                 year_built, distance_to_station, has_garden, has_parking, energy_rating):
    """Make a price prediction"""
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'square_feet': [square_feet],
        'year_built': [year_built],
        'distance_to_station_miles': [distance_to_station],
        'has_garden': [has_garden],
        'has_parking': [has_parking],
        'borough_encoded': [preprocessors['le_borough'].transform([borough])[0]],
        'property_type_encoded': [preprocessors['le_property_type'].transform([property_type])[0]],
        'energy_rating_encoded': [preprocessors['le_energy_rating'].transform([energy_rating])[0]]
    })
    
    # Reorder columns
    input_data = input_data[preprocessors['feature_names']]
    
    # Scale
    input_scaled = preprocessors['scaler'].transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    return prediction

# COMMAND ----------

# Example prediction 1: Luxury property in Westminster
prediction = predict_price(
    borough='Westminster',
    property_type='Detached',
    bedrooms=4,
    bathrooms=3,
    square_feet=2500,
    year_built=2020,
    distance_to_station=0.3,
    has_garden=1,
    has_parking=1,
    energy_rating='A'
)

print(f"Predicted Price: £{prediction:,.0f}")
print(f"Price per sq ft: £{prediction/2500:,.0f}")

# COMMAND ----------

# Example prediction 2: Affordable flat in Newham
prediction = predict_price(
    borough='Newham',
    property_type='Flat',
    bedrooms=2,
    bathrooms=1,
    square_feet=650,
    year_built=2010,
    distance_to_station=0.5,
    has_garden=0,
    has_parking=0,
    energy_rating='C'
)

print(f"Predicted Price: £{prediction:,.0f}")
print(f"Price per sq ft: £{prediction/650:,.0f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Deploy the App
# MAGIC 
# MAGIC Now that the model is trained, you can deploy the Streamlit app:
# MAGIC 
# MAGIC 1. Go to **Apps** in the Databricks sidebar
# MAGIC 2. Click **Create App**
# MAGIC 3. Select **From Repo**
# MAGIC 4. Configure:
# MAGIC    - **Name**: `london-housing-predictor`
# MAGIC    - **Source**: Your repo
# MAGIC    - **Config File**: `app.yaml`
# MAGIC 5. Click **Create App**
# MAGIC 
# MAGIC The app will be deployed and you'll get a URL to access it!

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎉 Setup Complete!
# MAGIC 
# MAGIC Your London Housing Price Predictor is now ready to use. The model has been trained and saved, and you can deploy the Streamlit app to start making predictions.
# MAGIC 
# MAGIC ### Next Steps:
# MAGIC - Deploy the app via Databricks Apps
# MAGIC - Test the prediction interface
# MAGIC - Explore the data visualizations
# MAGIC - Review model insights
# MAGIC 
# MAGIC ### Files Created:
# MAGIC - ✓ `data/london_housing_data.csv` - Training dataset
# MAGIC - ✓ `model/random_forest.pkl` or `model/gradient_boosting.pkl` - Trained model
# MAGIC - ✓ `model/preprocessors.pkl` - Feature preprocessors

