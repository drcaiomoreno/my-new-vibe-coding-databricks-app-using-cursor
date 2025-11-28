#!/bin/bash

# Setup script for London Housing Price Predictor Databricks App

echo "========================================="
echo "London Housing Price Predictor Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Create virtual environment (optional but recommended)
read -p "Do you want to create a virtual environment? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment activated!"
    echo ""
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed successfully!"
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p data
mkdir -p model
echo "Directories created!"
echo ""

# Generate dataset
echo "Generating London housing dataset..."
python data/generate_london_housing_data.py
echo "Dataset generated successfully!"
echo ""

# Train model
echo "Training machine learning model..."
echo "This may take a few minutes..."
python model/train_model.py
echo "Model trained successfully!"
echo ""

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To run the application:"
echo "  streamlit run app.py --server.port 8080"
echo ""
echo "The app will be available at: http://localhost:8080"
echo ""

