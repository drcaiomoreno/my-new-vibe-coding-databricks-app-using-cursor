"""
Generate synthetic London housing data for prediction model
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_london_housing_data(n_samples=5000):
    """
    Generate synthetic London housing data with realistic features
    
    Parameters:
    n_samples (int): Number of samples to generate
    
    Returns:
    pd.DataFrame: Synthetic housing dataset
    """
    np.random.seed(42)
    
    # London boroughs with typical price ranges
    boroughs = ['Westminster', 'Kensington and Chelsea', 'Camden', 'Hammersmith and Fulham',
                'Islington', 'Wandsworth', 'Lambeth', 'Southwark', 'Tower Hamlets', 'Hackney',
                'Greenwich', 'Lewisham', 'Newham', 'Barking and Dagenham', 'Havering',
                'Redbridge', 'Waltham Forest', 'Haringey', 'Enfield', 'Barnet']
    
    # Property types
    property_types = ['Flat', 'Terraced', 'Semi-Detached', 'Detached']
    
    # Generate features
    data = {
        'borough': np.random.choice(boroughs, n_samples),
        'property_type': np.random.choice(property_types, n_samples),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.15, 0.30, 0.35, 0.15, 0.05]),
        'bathrooms': np.random.choice([1, 2, 3, 4], n_samples, p=[0.40, 0.40, 0.15, 0.05]),
        'square_feet': np.random.randint(400, 3000, n_samples),
        'year_built': np.random.randint(1900, 2024, n_samples),
        'distance_to_station_miles': np.random.uniform(0.1, 2.5, n_samples),
        'has_garden': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'has_parking': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'energy_rating': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_samples, 
                                         p=[0.05, 0.15, 0.25, 0.30, 0.15, 0.07, 0.03])
    }
    
    df = pd.DataFrame(data)
    
    # Calculate price based on features with realistic London pricing
    base_price = 200000
    
    # Borough multipliers (premium areas cost more)
    borough_multiplier = {
        'Westminster': 2.5, 'Kensington and Chelsea': 2.8, 'Camden': 2.0,
        'Hammersmith and Fulham': 1.8, 'Islington': 1.9, 'Wandsworth': 1.7,
        'Lambeth': 1.5, 'Southwark': 1.6, 'Tower Hamlets': 1.5, 'Hackney': 1.5,
        'Greenwich': 1.3, 'Lewisham': 1.2, 'Newham': 1.1, 'Barking and Dagenham': 0.9,
        'Havering': 1.0, 'Redbridge': 1.1, 'Waltham Forest': 1.2, 'Haringey': 1.4,
        'Enfield': 1.1, 'Barnet': 1.3
    }
    
    # Property type multipliers
    property_type_multiplier = {
        'Flat': 0.8, 'Terraced': 1.0, 'Semi-Detached': 1.2, 'Detached': 1.5
    }
    
    # Calculate price
    df['price'] = base_price
    df['price'] *= df['borough'].map(borough_multiplier)
    df['price'] *= df['property_type'].map(property_type_multiplier)
    df['price'] += df['bedrooms'] * 80000
    df['price'] += df['bathrooms'] * 40000
    df['price'] += df['square_feet'] * 150
    df['price'] += (2024 - df['year_built']) * -500  # Newer = more expensive
    df['price'] -= df['distance_to_station_miles'] * 30000
    df['price'] += df['has_garden'] * 50000
    df['price'] += df['has_parking'] * 40000
    
    # Energy rating impact
    energy_multiplier = {'A': 1.1, 'B': 1.05, 'C': 1.0, 'D': 0.98, 'E': 0.95, 'F': 0.90, 'G': 0.85}
    df['price'] *= df['energy_rating'].map(energy_multiplier)
    
    # Add some noise
    df['price'] += np.random.normal(0, 50000, n_samples)
    df['price'] = df['price'].clip(lower=150000)  # Minimum price
    df['price'] = df['price'].round(-3)  # Round to nearest 1000
    
    # Add listing date (last 2 years)
    start_date = datetime(2023, 1, 1)
    df['listing_date'] = [start_date + timedelta(days=int(x)) for x in np.random.randint(0, 730, n_samples)]
    
    # Reorder columns
    df = df[['listing_date', 'borough', 'property_type', 'bedrooms', 'bathrooms', 
             'square_feet', 'year_built', 'distance_to_station_miles', 
             'has_garden', 'has_parking', 'energy_rating', 'price']]
    
    return df

if __name__ == "__main__":
    # Generate data
    df = generate_london_housing_data(5000)
    
    # Save to CSV
    df.to_csv('london_housing_data.csv', index=False)
    print(f"Generated {len(df)} housing records")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nPrice statistics:")
    print(df['price'].describe())

