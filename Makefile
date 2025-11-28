.PHONY: help install generate-data train run test verify clean all

help:
	@echo "London Housing Price Predictor - Makefile Commands"
	@echo "===================================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install       - Install Python dependencies"
	@echo "  make all           - Complete setup (install + generate + train)"
	@echo ""
	@echo "Data Commands:"
	@echo "  make generate-data - Generate synthetic London housing data"
	@echo ""
	@echo "Model Commands:"
	@echo "  make train         - Train the ML model"
	@echo ""
	@echo "Run Commands:"
	@echo "  make run           - Run the Streamlit app"
	@echo "  make verify        - Verify setup and check all components"
	@echo ""
	@echo "Testing Commands:"
	@echo "  make test          - Run unit tests"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean         - Clean generated files (data, models, cache)"
	@echo "  make help          - Show this help message"
	@echo ""

install:
	@echo "Installing dependencies..."
	pip3 install -r requirements.txt
	@echo "✅ Dependencies installed!"

generate-data:
	@echo "Generating London housing data..."
	python3 data/generate_london_housing_data.py
	@echo "✅ Data generated!"

train:
	@echo "Training ML model..."
	python3 model/train_model.py
	@echo "✅ Model trained!"

run:
	@echo "Starting Streamlit app..."
	@echo "App will be available at: http://localhost:8080"
	streamlit run app.py --server.port 8080

verify:
	@echo "Verifying setup..."
	python3 verify_setup.py

test:
	@echo "Running tests..."
	python3 tests/test_model.py

clean:
	@echo "Cleaning generated files..."
	rm -f data/london_housing_data.csv
	rm -f model/*.pkl
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"

all: install generate-data train
	@echo ""
	@echo "======================================"
	@echo "Setup Complete! 🎉"
	@echo "======================================"
	@echo ""
	@echo "To run the app:"
	@echo "  make run"
	@echo ""
	@echo "To verify setup:"
	@echo "  make verify"
	@echo ""

