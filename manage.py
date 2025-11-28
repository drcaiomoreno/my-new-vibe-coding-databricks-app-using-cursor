#!/usr/bin/env python3
"""
Command-line management tool for London Housing Price Predictor
"""
import sys
import os
import subprocess
import argparse

def run_command(cmd, description):
    """Run a shell command with output"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print('='*60)
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def install_dependencies():
    """Install Python dependencies"""
    return run_command('pip3 install -r requirements.txt', 'Installing Dependencies')

def generate_data():
    """Generate synthetic dataset"""
    return run_command('python3 data/generate_london_housing_data.py', 'Generating Dataset')

def train_model():
    """Train the ML model"""
    return run_command('python3 model/train_model.py', 'Training Model')

def run_app():
    """Run the Streamlit app"""
    print("\n" + "="*60)
    print("  Starting Streamlit App")
    print("="*60)
    print("\n🌐 App will be available at: http://localhost:8080")
    print("   Press Ctrl+C to stop\n")
    subprocess.run('streamlit run app.py --server.port 8080', shell=True)

def verify_setup():
    """Verify the setup"""
    return run_command('python3 verify_setup.py', 'Verifying Setup')

def run_tests():
    """Run unit tests"""
    return run_command('python3 tests/test_model.py', 'Running Tests')

def clean():
    """Clean generated files"""
    print("\n" + "="*60)
    print("  Cleaning Generated Files")
    print("="*60)
    
    files_to_remove = [
        'data/london_housing_data.csv',
        'model/random_forest.pkl',
        'model/gradient_boosting.pkl',
        'model/preprocessors.pkl'
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ Removed: {file_path}")
        else:
            print(f"⏭️  Skipped: {file_path} (doesn't exist)")
    
    # Clean Python cache
    subprocess.run('find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null', shell=True)
    subprocess.run('find . -type f -name "*.pyc" -delete', shell=True)
    print("✅ Cleaned Python cache files")
    print("\n✅ Cleanup complete!")
    return True

def setup_all():
    """Complete setup process"""
    print("\n" + "🏠"*20)
    print("  London Housing Price Predictor - Setup")
    print("🏠"*20)
    
    steps = [
        (install_dependencies, "Installing dependencies"),
        (generate_data, "Generating data"),
        (train_model, "Training model")
    ]
    
    for step_func, step_name in steps:
        if not step_func():
            print(f"\n❌ Failed at: {step_name}")
            return False
    
    print("\n" + "="*60)
    print("  🎉 Setup Complete!")
    print("="*60)
    print("\nTo run the app:")
    print("  python3 manage.py run")
    print("\nOr:")
    print("  make run")
    print()
    
    return True

def show_status():
    """Show project status"""
    print("\n" + "📊"*20)
    print("  Project Status")
    print("📊"*20)
    
    checks = [
        ('requirements.txt', 'Dependencies file'),
        ('app.py', 'Main application'),
        ('data/london_housing_data.csv', 'Dataset'),
        ('model/random_forest.pkl', 'Random Forest model'),
        ('model/gradient_boosting.pkl', 'Gradient Boosting model'),
        ('model/preprocessors.pkl', 'Preprocessors')
    ]
    
    print("\n📁 Files:")
    for file_path, description in checks:
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        size = ""
        if exists:
            size_bytes = os.path.getsize(file_path)
            if size_bytes > 1024*1024:
                size = f" ({size_bytes/(1024*1024):.1f} MB)"
            elif size_bytes > 1024:
                size = f" ({size_bytes/1024:.1f} KB)"
            else:
                size = f" ({size_bytes} B)"
        print(f"  {status} {description}: {file_path}{size}")
    
    # Check if app is ready
    print("\n🚀 Readiness:")
    has_data = os.path.exists('data/london_housing_data.csv')
    has_model = os.path.exists('model/random_forest.pkl') or os.path.exists('model/gradient_boosting.pkl')
    has_preprocessors = os.path.exists('model/preprocessors.pkl')
    
    if has_data and has_model and has_preprocessors:
        print("  ✅ App is ready to run!")
        print("  Run: python3 manage.py run")
    else:
        print("  ❌ App is not ready")
        if not has_data:
            print("  • Missing: Dataset (run: python3 manage.py generate-data)")
        if not has_model or not has_preprocessors:
            print("  • Missing: Trained model (run: python3 manage.py train)")
        print("\n  Quick fix: python3 manage.py setup")
    
    print()

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='London Housing Price Predictor - Management Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 manage.py setup        # Complete setup
  python3 manage.py run          # Run the app
  python3 manage.py verify       # Verify setup
  python3 manage.py status       # Show project status
        """
    )
    
    parser.add_argument('command', choices=[
        'install', 'generate-data', 'train', 'run', 
        'verify', 'test', 'clean', 'setup', 'status'
    ], help='Command to execute')
    
    args = parser.parse_args()
    
    commands = {
        'install': install_dependencies,
        'generate-data': generate_data,
        'train': train_model,
        'run': run_app,
        'verify': verify_setup,
        'test': run_tests,
        'clean': clean,
        'setup': setup_all,
        'status': show_status
    }
    
    command_func = commands.get(args.command)
    if command_func:
        success = command_func()
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()

