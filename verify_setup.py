#!/usr/bin/env python3
"""
Verification script to check if all components are set up correctly
"""
import os
import sys

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_mark(condition, success_msg, failure_msg):
    """Print check mark or X based on condition"""
    if condition:
        print(f"✅ {success_msg}")
        return True
    else:
        print(f"❌ {failure_msg}")
        return False

def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 8:
        check_mark(True, f"Python {version_str} ✓", "")
        return True
    else:
        check_mark(False, "", f"Python {version_str} - Need 3.8 or higher")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'plotly',
        'joblib'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            check_mark(True, f"{package} installed", "")
        except ImportError:
            check_mark(False, "", f"{package} not installed")
            all_installed = False
    
    return all_installed

def check_files():
    """Check if required files exist"""
    print_header("Checking Project Files")
    
    required_files = [
        ('app.py', 'Main application'),
        ('app.yaml', 'Databricks config'),
        ('requirements.txt', 'Dependencies'),
        ('data/generate_london_housing_data.py', 'Data generator'),
        ('model/train_model.py', 'Model trainer'),
        ('README.md', 'Documentation'),
    ]
    
    all_exist = True
    for file_path, description in required_files:
        exists = os.path.exists(file_path)
        check_mark(exists, f"{description}: {file_path}", f"Missing: {file_path}")
        all_exist = all_exist and exists
    
    return all_exist

def check_data():
    """Check if data has been generated"""
    print_header("Checking Data")
    
    data_file = 'data/london_housing_data.csv'
    
    if os.path.exists(data_file):
        size = os.path.getsize(data_file)
        size_kb = size / 1024
        check_mark(True, f"Data file exists ({size_kb:.1f} KB)", "")
        
        # Try to read the file
        try:
            import pandas as pd
            df = pd.read_csv(data_file)
            check_mark(True, f"Data readable ({len(df)} records)", "")
            return True
        except Exception as e:
            check_mark(False, "", f"Data file corrupted: {e}")
            return False
    else:
        check_mark(False, "", "Data not generated. Run: python3 data/generate_london_housing_data.py")
        return False

def check_model():
    """Check if model has been trained"""
    print_header("Checking Model")
    
    model_files = ['model/random_forest.pkl', 'model/gradient_boosting.pkl']
    preprocessor_file = 'model/preprocessors.pkl'
    
    model_exists = False
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file)
            size_mb = size / (1024 * 1024)
            check_mark(True, f"Model exists: {model_file} ({size_mb:.1f} MB)", "")
            model_exists = True
            break
    
    if not model_exists:
        check_mark(False, "", "Model not trained. Run: python3 model/train_model.py")
    
    preprocessor_exists = os.path.exists(preprocessor_file)
    check_mark(preprocessor_exists, "Preprocessors exist", "Preprocessors missing")
    
    return model_exists and preprocessor_exists

def check_model_loading():
    """Try to load the model"""
    print_header("Testing Model Loading")
    
    try:
        import joblib
        
        # Try to load model
        model = None
        if os.path.exists('model/random_forest.pkl'):
            model = joblib.load('model/random_forest.pkl')
            model_name = "Random Forest"
        elif os.path.exists('model/gradient_boosting.pkl'):
            model = joblib.load('model/gradient_boosting.pkl')
            model_name = "Gradient Boosting"
        
        if model is not None:
            check_mark(True, f"{model_name} model loaded successfully", "")
            
            # Try to load preprocessors
            preprocessors = joblib.load('model/preprocessors.pkl')
            check_mark(True, "Preprocessors loaded successfully", "")
            return True
        else:
            check_mark(False, "", "No model file found")
            return False
            
    except Exception as e:
        check_mark(False, "", f"Failed to load model: {e}")
        return False

def print_summary(results):
    """Print summary of results"""
    print_header("Summary")
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"\nTotal Checks: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All checks passed! Your setup is complete!")
        print("\nTo run the app:")
        print("  streamlit run app.py --server.port 8080")
        print("\nThen open: http://localhost:8080")
        return True
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")
        print("\nQuick fixes:")
        
        if not results['dependencies']:
            print("  1. Install dependencies: pip3 install -r requirements.txt")
        
        if not results['data']:
            print("  2. Generate data: python3 data/generate_london_housing_data.py")
        
        if not results['model']:
            print("  3. Train model: python3 model/train_model.py")
        
        print("\nOr run the automated setup:")
        print("  ./setup.sh")
        
        return False

def main():
    """Main verification function"""
    print("\n" + "🔍 " * 20)
    print("  London Housing Price Predictor - Setup Verification")
    print("🔍 " * 20)
    
    results = {
        'python': check_python_version(),
        'dependencies': check_dependencies(),
        'files': check_files(),
        'data': check_data(),
        'model': check_model(),
    }
    
    # Only check model loading if model exists
    if results['model']:
        results['model_loading'] = check_model_loading()
    
    success = print_summary(results)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

