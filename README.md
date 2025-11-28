# 🏠 London Housing Price Predictor - Databricks App

A comprehensive Databricks application for predicting London housing prices using machine learning. This app provides an interactive interface for property price predictions, data exploration, and model insights.

## 📋 Features

- **Price Prediction**: Predict property prices based on multiple features including location, size, and amenities
- **Data Exploration**: Interactive visualizations and analysis of London housing market data
- **Model Insights**: Feature importance analysis and model performance metrics
- **Real-time Predictions**: Instant price estimates with confidence ranges
- **Beautiful UI**: Modern, responsive interface built with Streamlit

## 🏗️ Project Structure

```
.
├── app.py                              # Main Streamlit application
├── app.yaml                            # Databricks App configuration
├── requirements.txt                    # Python dependencies
├── data/
│   ├── generate_london_housing_data.py # Data generation script
│   └── london_housing_data.csv         # Generated dataset (created after running)
├── model/
│   ├── train_model.py                  # Model training pipeline
│   ├── random_forest.pkl               # Trained model (created after training)
│   └── preprocessors.pkl               # Feature preprocessors (created after training)
└── README.md                           # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd my-new-vibe-coding-databricks-app-using-cursor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Generate the dataset**
```bash
python data/generate_london_housing_data.py
```

4. **Train the model**
```bash
python model/train_model.py
```

5. **Run the application**
```bash
streamlit run app.py --server.port 8080
```

The app will be available at `http://localhost:8080`

## 🎯 Usage

### Price Prediction

1. Navigate to the "Price Prediction" page
2. Enter property details:
   - Borough (location in London)
   - Property type (Flat, Terraced, Semi-Detached, Detached)
   - Number of bedrooms and bathrooms
   - Square footage
   - Year built
   - Distance to nearest station
   - Additional features (garden, parking, energy rating)
3. Click "Predict Price" to get instant price estimates

### Data Exploration

- View dataset statistics and distributions
- Analyze price trends by borough
- Explore property type distributions
- Examine feature correlations

### Model Insights

- View feature importance rankings
- Understand model performance
- Review training data statistics

## 📊 Dataset

The application uses a synthetic London housing dataset with realistic features:

- **Size**: 5,000 property records
- **Features**:
  - Location: 20 London boroughs
  - Property types: Flat, Terraced, Semi-Detached, Detached
  - Bedrooms: 1-5
  - Bathrooms: 1-4
  - Square footage: 400-3,000 sq ft
  - Year built: 1900-2024
  - Distance to station: 0.1-2.5 miles
  - Garden and parking availability
  - Energy ratings: A-G

## 🤖 Machine Learning Model

### Model Architecture

- **Algorithm**: Random Forest / Gradient Boosting Regressor
- **Features**: 10 input features (numeric and encoded categorical)
- **Target**: Property price in GBP (£)

### Model Performance

The model achieves high accuracy with:
- Mean Absolute Error (MAE): ~£50,000-70,000
- R² Score: >0.85
- RMSE: ~£80,000-100,000

### Key Predictive Features

1. Borough location (neighborhood)
2. Square footage
3. Number of bedrooms
4. Property type
5. Distance to station

## 🎨 Technologies Used

- **Python 3.8+**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Joblib**: Model serialization

## 📝 Databricks Deployment

This app is configured for deployment on Databricks with the `app.yaml` configuration file.

### Deploying to Databricks

1. Upload the project to your Databricks workspace
2. **IMPORTANT:** Run `notebooks/Setup_Models.py` to generate data and train model
3. Deploy using Databricks Apps with `app.yaml`
4. Start/restart the app

**⚠️ Common Issue:** If you see "Model not found" error, see **[DATABRICKS_TROUBLESHOOTING.md](DATABRICKS_TROUBLESHOOTING.md)**

For detailed deployment instructions, see **[DATABRICKS_DEPLOYMENT.md](DATABRICKS_DEPLOYMENT.md)**

## 🔧 Configuration

### app.yaml

```yaml
command: ["streamlit", "run", "app.py", "--server.port", "8080"]
```

This configuration tells Databricks how to run the application.

## 🧪 Model Training

The training pipeline includes:

1. **Data Loading**: Load synthetic London housing data
2. **Preprocessing**: 
   - Encode categorical variables
   - Scale numerical features
   - Split into train/test sets
3. **Model Training**: Train multiple models (Random Forest, Gradient Boosting)
4. **Evaluation**: Compare model performance
5. **Serialization**: Save best model and preprocessors

Run training with:
```bash
python model/train_model.py
```

## 📈 Future Enhancements

- [ ] Real London housing data integration
- [ ] Time series forecasting
- [ ] Neighborhood comparison tool
- [ ] Market trend analysis
- [ ] Property recommendation system
- [ ] Advanced filtering options
- [ ] Export predictions to CSV
- [ ] API endpoint for predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- London housing market data structure inspired by UK property listings
- Built with modern data science best practices
- Designed for Databricks platform deployment

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Note**: This application uses synthetic data for demonstration purposes. For production use with real data, please ensure proper data validation and regulatory compliance.
my-new-vibe-coding-databricks-app-using-cursor
