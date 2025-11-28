"""
Unit tests for the model training and prediction pipeline
"""
import unittest
import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_london_housing_data import generate_london_housing_data

class TestDataGeneration(unittest.TestCase):
    """Test data generation functions"""
    
    def test_generate_data_shape(self):
        """Test that generated data has correct shape"""
        df = generate_london_housing_data(100)
        self.assertEqual(len(df), 100)
        self.assertEqual(len(df.columns), 12)
    
    def test_generate_data_columns(self):
        """Test that generated data has correct columns"""
        df = generate_london_housing_data(10)
        expected_columns = ['listing_date', 'borough', 'property_type', 'bedrooms', 
                          'bathrooms', 'square_feet', 'year_built', 
                          'distance_to_station_miles', 'has_garden', 'has_parking', 
                          'energy_rating', 'price']
        self.assertListEqual(list(df.columns), expected_columns)
    
    def test_price_range(self):
        """Test that prices are in reasonable range"""
        df = generate_london_housing_data(100)
        self.assertTrue(df['price'].min() >= 150000)
        self.assertTrue(df['price'].max() <= 5000000)
    
    def test_bedrooms_range(self):
        """Test that bedrooms are in valid range"""
        df = generate_london_housing_data(100)
        self.assertTrue(df['bedrooms'].min() >= 1)
        self.assertTrue(df['bedrooms'].max() <= 5)
    
    def test_data_types(self):
        """Test that data types are correct"""
        df = generate_london_housing_data(10)
        self.assertTrue(df['borough'].dtype == 'object')
        self.assertTrue(df['property_type'].dtype == 'object')
        self.assertTrue(df['bedrooms'].dtype in ['int64', 'int32'])
        self.assertTrue(df['price'].dtype in ['float64', 'int64'])

class TestModelPrediction(unittest.TestCase):
    """Test model prediction functions"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        # Generate small dataset
        cls.df = generate_london_housing_data(100)
        
        # Save to temp location
        os.makedirs('tests/temp', exist_ok=True)
        cls.data_path = 'tests/temp/test_data.csv'
        cls.df.to_csv(cls.data_path, index=False)
    
    def test_data_loading(self):
        """Test that data can be loaded"""
        df = pd.read_csv(self.data_path)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 100)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        if os.path.exists(cls.data_path):
            os.remove(cls.data_path)
        if os.path.exists('tests/temp'):
            os.rmdir('tests/temp')

class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions"""
    
    def test_encoding(self):
        """Test categorical encoding"""
        from sklearn.preprocessing import LabelEncoder
        
        boroughs = ['Westminster', 'Camden', 'Westminster', 'Hackney']
        le = LabelEncoder()
        encoded = le.fit_transform(boroughs)
        
        self.assertEqual(len(encoded), 4)
        self.assertTrue(all(isinstance(x, (int, np.integer)) for x in encoded))
    
    def test_scaling(self):
        """Test feature scaling"""
        from sklearn.preprocessing import StandardScaler
        
        data = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)
        
        # Scaled data should have mean ~0 and std ~1
        self.assertAlmostEqual(scaled.mean(), 0, places=5)
        self.assertAlmostEqual(scaled.std(), 1, places=5)

class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering"""
    
    def test_price_calculation(self):
        """Test that price calculation is reasonable"""
        df = generate_london_housing_data(1000)
        
        # More bedrooms should generally mean higher prices
        avg_price_by_bedrooms = df.groupby('bedrooms')['price'].mean()
        self.assertTrue(avg_price_by_bedrooms.is_monotonic_increasing)
        
        # Westminster should be more expensive than outer boroughs
        westminster_avg = df[df['borough'] == 'Westminster']['price'].mean()
        newham_avg = df[df['borough'] == 'Newham']['price'].mean()
        self.assertGreater(westminster_avg, newham_avg)

def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)

if __name__ == '__main__':
    run_tests()

