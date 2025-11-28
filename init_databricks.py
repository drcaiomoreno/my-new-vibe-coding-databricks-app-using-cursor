#!/usr/bin/env python3
"""
Initialization script for Databricks deployment
Run this once before starting the app to generate data and train models
"""
import os
import sys

def main():
    """Initialize data and models for Databricks"""
    print("="*60)
    print("  Databricks App Initialization")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found. Are you in the correct directory?")
        sys.exit(1)
    
    print("\n📊 Step 1: Generating dataset...")
    try:
        exec(open('data/generate_london_housing_data.py').read())
        print("✅ Dataset generated successfully!")
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        sys.exit(1)
    
    print("\n🤖 Step 2: Training model...")
    try:
        exec(open('model/train_model.py').read())
        print("✅ Model trained successfully!")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        sys.exit(1)
    
    print("\n🔍 Step 3: Verifying files...")
    required_files = [
        'data/london_housing_data.csv',
        'model/preprocessors.pkl'
    ]
    
    model_files = ['model/random_forest.pkl', 'model/gradient_boosting.pkl']
    has_model = any(os.path.exists(f) for f in model_files)
    
    all_good = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing!")
            all_good = False
    
    if has_model:
        model_file = next(f for f in model_files if os.path.exists(f))
        print(f"✅ {model_file}")
    else:
        print("❌ No model file found!")
        all_good = False
    
    if all_good:
        print("\n" + "="*60)
        print("  🎉 Initialization Complete!")
        print("="*60)
        print("\nYour Databricks App is ready to run.")
        print("You can now start or restart the app.")
        return 0
    else:
        print("\n❌ Initialization failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

