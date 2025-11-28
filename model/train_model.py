"""
Train a machine learning model to predict London housing prices
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_preprocess_data(data_path):
    """
    Load and preprocess the housing data
    
    Parameters:
    data_path (str): Path to the CSV data file
    
    Returns:
    tuple: X_train, X_test, y_train, y_test, preprocessors
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Drop listing_date for now (could be used for time series)
    df = df.drop('listing_date', axis=1)
    
    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Encode categorical variables
    le_borough = LabelEncoder()
    le_property_type = LabelEncoder()
    le_energy_rating = LabelEncoder()
    
    X['borough_encoded'] = le_borough.fit_transform(X['borough'])
    X['property_type_encoded'] = le_property_type.fit_transform(X['property_type'])
    X['energy_rating_encoded'] = le_energy_rating.fit_transform(X['energy_rating'])
    
    # Drop original categorical columns
    X = X.drop(['borough', 'property_type', 'energy_rating'], axis=1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Store preprocessors
    preprocessors = {
        'scaler': scaler,
        'le_borough': le_borough,
        'le_property_type': le_property_type,
        'le_energy_rating': le_energy_rating,
        'feature_names': X.columns.tolist()
    }
    
    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessors, X_train, X_test

def train_random_forest(X_train, y_train):
    """Train a Random Forest model"""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    """Train a Gradient Boosting model"""
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"MAE: £{mae:,.2f}")
    print(f"RMSE: £{rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def save_model_and_preprocessors(model, preprocessors, model_path='model', model_name='best_model'):
    """Save the trained model and preprocessors"""
    os.makedirs(model_path, exist_ok=True)
    
    # Save model
    joblib.dump(model, os.path.join(model_path, f'{model_name}.pkl'))
    
    # Save preprocessors
    joblib.dump(preprocessors, os.path.join(model_path, 'preprocessors.pkl'))
    
    print(f"\nModel and preprocessors saved to {model_path}/")

def main():
    """Main training pipeline"""
    print("Loading and preprocessing data...")
    data_path = 'data/london_housing_data.csv'
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Generating data first...")
        from data.generate_london_housing_data import generate_london_housing_data
        df = generate_london_housing_data(5000)
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_path, index=False)
        print("Data generated successfully!")
    
    X_train_scaled, X_test_scaled, y_train, y_test, preprocessors, X_train, X_test = load_and_preprocess_data(data_path)
    
    print(f"\nTraining set size: {len(X_train_scaled)}")
    print(f"Test set size: {len(X_test_scaled)}")
    
    # Train Random Forest
    print("\n" + "="*50)
    print("Training Random Forest model...")
    rf_model = train_random_forest(X_train_scaled, y_train)
    rf_metrics = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
    
    # Train Gradient Boosting
    print("\n" + "="*50)
    print("Training Gradient Boosting model...")
    gb_model = train_gradient_boosting(X_train_scaled, y_train)
    gb_metrics = evaluate_model(gb_model, X_test_scaled, y_test, "Gradient Boosting")
    
    # Select best model based on R² score
    print("\n" + "="*50)
    if rf_metrics['r2'] > gb_metrics['r2']:
        print("Random Forest selected as best model")
        best_model = rf_model
        best_model_name = 'random_forest'
    else:
        print("Gradient Boosting selected as best model")
        best_model = gb_model
        best_model_name = 'gradient_boosting'
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': preprocessors['feature_names'],
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save model and preprocessors
    save_model_and_preprocessors(best_model, preprocessors, model_name=best_model_name)
    
    print("\n" + "="*50)
    print("Training complete!")

if __name__ == "__main__":
    main()

