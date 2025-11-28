"""
Utility functions for making predictions
"""
import pandas as pd
import joblib
import os

def load_model(model_path='model'):
    """
    Load the trained model and preprocessors
    
    Parameters:
    model_path (str): Path to the model directory
    
    Returns:
    tuple: (model, preprocessors)
    """
    # Try to load Random Forest first
    try:
        model = joblib.load(os.path.join(model_path, 'random_forest.pkl'))
    except FileNotFoundError:
        # Fall back to Gradient Boosting
        model = joblib.load(os.path.join(model_path, 'gradient_boosting.pkl'))
    
    preprocessors = joblib.load(os.path.join(model_path, 'preprocessors.pkl'))
    
    return model, preprocessors

def predict_single_property(model, preprocessors, property_details):
    """
    Predict price for a single property
    
    Parameters:
    model: Trained model
    preprocessors: Dictionary of preprocessors
    property_details (dict): Property details with keys:
        - borough (str)
        - property_type (str)
        - bedrooms (int)
        - bathrooms (int)
        - square_feet (int)
        - year_built (int)
        - distance_to_station_miles (float)
        - has_garden (int, 0 or 1)
        - has_parking (int, 0 or 1)
        - energy_rating (str)
    
    Returns:
    float: Predicted price in GBP
    """
    # Create input dataframe
    input_data = pd.DataFrame({
        'bedrooms': [property_details['bedrooms']],
        'bathrooms': [property_details['bathrooms']],
        'square_feet': [property_details['square_feet']],
        'year_built': [property_details['year_built']],
        'distance_to_station_miles': [property_details['distance_to_station_miles']],
        'has_garden': [property_details['has_garden']],
        'has_parking': [property_details['has_parking']],
        'borough_encoded': [preprocessors['le_borough'].transform([property_details['borough']])[0]],
        'property_type_encoded': [preprocessors['le_property_type'].transform([property_details['property_type']])[0]],
        'energy_rating_encoded': [preprocessors['le_energy_rating'].transform([property_details['energy_rating']])[0]]
    })
    
    # Reorder columns to match training data
    input_data = input_data[preprocessors['feature_names']]
    
    # Scale the input
    input_scaled = preprocessors['scaler'].transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    return prediction

def predict_batch(model, preprocessors, properties_df):
    """
    Predict prices for multiple properties
    
    Parameters:
    model: Trained model
    preprocessors: Dictionary of preprocessors
    properties_df (pd.DataFrame): DataFrame with property details
    
    Returns:
    pd.Series: Predicted prices
    """
    # Encode categorical variables
    df = properties_df.copy()
    
    df['borough_encoded'] = preprocessors['le_borough'].transform(df['borough'])
    df['property_type_encoded'] = preprocessors['le_property_type'].transform(df['property_type'])
    df['energy_rating_encoded'] = preprocessors['le_energy_rating'].transform(df['energy_rating'])
    
    # Select and reorder features
    feature_columns = preprocessors['feature_names']
    X = df[feature_columns]
    
    # Scale features
    X_scaled = preprocessors['scaler'].transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    
    return pd.Series(predictions, index=properties_df.index)

def get_price_confidence_interval(model, X_scaled, confidence=0.9):
    """
    Calculate confidence interval for predictions
    (Only works with tree-based models that support prediction intervals)
    
    Parameters:
    model: Trained model
    X_scaled: Scaled input features
    confidence (float): Confidence level (0-1)
    
    Returns:
    tuple: (lower_bound, upper_bound)
    """
    # For ensemble models, we can use the predictions from individual trees
    # to estimate uncertainty
    
    if hasattr(model, 'estimators_'):
        # Get predictions from all trees
        predictions = []
        for estimator in model.estimators_:
            pred = estimator.predict(X_scaled)
            predictions.append(pred)
        
        predictions = pd.DataFrame(predictions)
        
        # Calculate percentiles
        alpha = (1 - confidence) / 2
        lower_bound = predictions.quantile(alpha, axis=0)
        upper_bound = predictions.quantile(1 - alpha, axis=0)
        
        return lower_bound.values, upper_bound.values
    else:
        # For single models, use a simple margin
        prediction = model.predict(X_scaled)
        margin = prediction * 0.1  # ±10%
        return prediction - margin, prediction + margin

if __name__ == "__main__":
    # Example usage
    model, preprocessors = load_model()
    
    # Test single prediction
    property_details = {
        'borough': 'Westminster',
        'property_type': 'Flat',
        'bedrooms': 2,
        'bathrooms': 1,
        'square_feet': 800,
        'year_built': 2015,
        'distance_to_station_miles': 0.3,
        'has_garden': 0,
        'has_parking': 0,
        'energy_rating': 'C'
    }
    
    price = predict_single_property(model, preprocessors, property_details)
    print(f"Predicted price: £{price:,.0f}")

