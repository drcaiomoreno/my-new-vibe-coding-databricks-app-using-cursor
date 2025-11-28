"""
Databricks App for London Housing Price Prediction
A Streamlit application for predicting London housing prices
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="London Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and preprocessors"""
    try:
        model = joblib.load('model/random_forest.pkl')
        preprocessors = joblib.load('model/preprocessors.pkl')
        return model, preprocessors
    except FileNotFoundError:
        try:
            model = joblib.load('model/gradient_boosting.pkl')
            preprocessors = joblib.load('model/preprocessors.pkl')
            return model, preprocessors
        except:
            return None, None

@st.cache_data
def load_data():
    """Load the housing data for analysis"""
    try:
        df = pd.read_csv('data/london_housing_data.csv')
        return df
    except FileNotFoundError:
        return None

def preprocess_input(borough, property_type, bedrooms, bathrooms, square_feet, 
                     year_built, distance_to_station, has_garden, has_parking, 
                     energy_rating, preprocessors):
    """Preprocess user input for prediction"""
    # Create input dataframe
    input_data = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'square_feet': [square_feet],
        'year_built': [year_built],
        'distance_to_station_miles': [distance_to_station],
        'has_garden': [has_garden],
        'has_parking': [has_parking],
        'borough_encoded': [preprocessors['le_borough'].transform([borough])[0]],
        'property_type_encoded': [preprocessors['le_property_type'].transform([property_type])[0]],
        'energy_rating_encoded': [preprocessors['le_energy_rating'].transform([energy_rating])[0]]
    })
    
    # Reorder columns to match training data
    input_data = input_data[preprocessors['feature_names']]
    
    # Scale the input
    input_scaled = preprocessors['scaler'].transform(input_data)
    
    return input_scaled

def main():
    """Main application"""
    
    # Header
    st.title("🏠 London Housing Price Predictor")
    st.markdown("### Predict housing prices in London using Machine Learning")
    st.markdown("---")
    
    # Load model and data
    model, preprocessors = load_model_and_preprocessors()
    data = load_data()
    
    if model is None or preprocessors is None:
        st.error("⚠️ Model not found! Please train the model first by running: `python model/train_model.py`")
        st.info("This will generate the data and train the model.")
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Price Prediction", "Data Exploration", "Model Insights"])
    
    if page == "Price Prediction":
        show_prediction_page(model, preprocessors)
    elif page == "Data Exploration":
        show_data_exploration_page(data)
    else:
        show_model_insights_page(model, preprocessors, data)

def show_prediction_page(model, preprocessors):
    """Show the prediction interface"""
    st.header("Predict Property Price")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Location & Property Details")
        
        borough = st.selectbox(
            "Borough",
            options=sorted(preprocessors['le_borough'].classes_),
            help="Select the London borough"
        )
        
        property_type = st.selectbox(
            "Property Type",
            options=preprocessors['le_property_type'].classes_,
            help="Type of property"
        )
        
        bedrooms = st.slider("Number of Bedrooms", 1, 5, 3)
        bathrooms = st.slider("Number of Bathrooms", 1, 4, 2)
        
        square_feet = st.number_input(
            "Square Feet",
            min_value=300,
            max_value=5000,
            value=1000,
            step=50,
            help="Total living area in square feet"
        )
    
    with col2:
        st.subheader("Additional Features")
        
        year_built = st.slider(
            "Year Built",
            min_value=1900,
            max_value=2024,
            value=2000,
            help="Year the property was built"
        )
        
        distance_to_station = st.slider(
            "Distance to Station (miles)",
            min_value=0.1,
            max_value=2.5,
            value=0.5,
            step=0.1,
            help="Distance to nearest tube/train station"
        )
        
        has_garden = st.checkbox("Has Garden", value=True)
        has_parking = st.checkbox("Has Parking", value=False)
        
        energy_rating = st.selectbox(
            "Energy Rating",
            options=preprocessors['le_energy_rating'].classes_,
            index=3,
            help="Energy efficiency rating (A is best, G is worst)"
        )
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Price", type="primary"):
        with st.spinner("Calculating prediction..."):
            # Preprocess input
            input_scaled = preprocess_input(
                borough, property_type, bedrooms, bathrooms, square_feet,
                year_built, distance_to_station, has_garden, has_parking,
                energy_rating, preprocessors
            )
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display prediction
            st.success("Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Predicted Price",
                    value=f"£{prediction:,.0f}"
                )
            
            with col2:
                st.metric(
                    label="Price per Sq Ft",
                    value=f"£{prediction/square_feet:,.0f}"
                )
            
            with col3:
                # Estimate range (±10%)
                lower_bound = prediction * 0.9
                upper_bound = prediction * 1.1
                st.metric(
                    label="Estimated Range",
                    value=f"£{lower_bound:,.0f} - £{upper_bound:,.0f}"
                )
            
            # Additional insights
            st.markdown("### 📊 Price Insights")
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.info(f"""
                **Property Summary:**
                - 📍 Location: {borough}
                - 🏗️ Type: {property_type}
                - 🛏️ {bedrooms} bed, {bathrooms} bath
                - 📏 {square_feet:,} sq ft
                - 📅 Built in {year_built}
                """)
            
            with insights_col2:
                st.info(f"""
                **Features:**
                - 🚇 {distance_to_station} miles to station
                - {'🌳 Garden' if has_garden else '❌ No Garden'}
                - {'🚗 Parking' if has_parking else '❌ No Parking'}
                - ⚡ Energy Rating: {energy_rating}
                """)

def show_data_exploration_page(data):
    """Show data exploration and visualizations"""
    st.header("Data Exploration")
    
    if data is None:
        st.warning("No data available. Please generate data first.")
        return
    
    # Overview statistics
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", f"{len(data):,}")
    with col2:
        st.metric("Average Price", f"£{data['price'].mean():,.0f}")
    with col3:
        st.metric("Median Price", f"£{data['price'].median():,.0f}")
    with col4:
        st.metric("Price Range", f"£{data['price'].max() - data['price'].min():,.0f}")
    
    st.markdown("---")
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Price Distribution", "Borough Analysis", "Property Features", "Correlations"])
    
    with tab1:
        st.subheader("Price Distribution")
        fig = px.histogram(
            data, 
            x='price', 
            nbins=50,
            title='Distribution of Property Prices',
            labels={'price': 'Price (£)', 'count': 'Number of Properties'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Average Price by Borough")
        borough_avg = data.groupby('borough')['price'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=borough_avg.index,
            y=borough_avg.values,
            title='Average Property Price by Borough',
            labels={'x': 'Borough', 'y': 'Average Price (£)'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Property type distribution
        st.subheader("Property Type Distribution")
        property_counts = data['property_type'].value_counts()
        fig = px.pie(
            values=property_counts.values,
            names=property_counts.index,
            title='Distribution of Property Types'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Property Features Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bedrooms vs Price
            fig = px.box(
                data,
                x='bedrooms',
                y='price',
                title='Price Distribution by Number of Bedrooms',
                labels={'bedrooms': 'Bedrooms', 'price': 'Price (£)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Square feet vs Price
            fig = px.scatter(
                data.sample(min(1000, len(data))),
                x='square_feet',
                y='price',
                color='property_type',
                title='Price vs Square Feet',
                labels={'square_feet': 'Square Feet', 'price': 'Price (£)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Feature Correlations")
        
        # Select numeric columns
        numeric_cols = ['bedrooms', 'bathrooms', 'square_feet', 'year_built', 
                       'distance_to_station_miles', 'has_garden', 'has_parking', 'price']
        corr_data = data[numeric_cols].corr()
        
        fig = px.imshow(
            corr_data,
            title='Feature Correlation Heatmap',
            labels=dict(color="Correlation"),
            x=corr_data.columns,
            y=corr_data.columns,
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Show sample data
    st.markdown("---")
    st.subheader("Sample Data")
    st.dataframe(data.head(100), use_container_width=True)

def show_model_insights_page(model, preprocessors, data):
    """Show model performance and feature importance"""
    st.header("Model Insights")
    
    # Feature importance
    st.subheader("Feature Importance")
    
    feature_importance = pd.DataFrame({
        'Feature': preprocessors['feature_names'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Rename features for better display
    feature_display_names = {
        'bedrooms': 'Bedrooms',
        'bathrooms': 'Bathrooms',
        'square_feet': 'Square Feet',
        'year_built': 'Year Built',
        'distance_to_station_miles': 'Distance to Station',
        'has_garden': 'Has Garden',
        'has_parking': 'Has Parking',
        'borough_encoded': 'Borough',
        'property_type_encoded': 'Property Type',
        'energy_rating_encoded': 'Energy Rating'
    }
    
    feature_importance['Feature'] = feature_importance['Feature'].map(feature_display_names)
    
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance in Price Prediction',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Model information
    st.markdown("---")
    st.subheader("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Model Type:** Random Forest / Gradient Boosting
        
        **Training Details:**
        - Algorithm: Ensemble Learning
        - Features: 10 input features
        - Target: Property price in GBP
        
        **Key Features:**
        - Borough location
        - Property type
        - Number of bedrooms/bathrooms
        - Property size (sq ft)
        - Age and proximity to stations
        """)
    
    with col2:
        st.info("""
        **Model Performance:**
        - High accuracy predictions
        - Handles complex feature interactions
        - Robust to outliers
        
        **Use Cases:**
        - Property valuation
        - Investment analysis
        - Market research
        - Price comparison
        """)
    
    # Data statistics
    if data is not None:
        st.markdown("---")
        st.subheader("Training Data Statistics")
        st.dataframe(data.describe(), use_container_width=True)

if __name__ == "__main__":
    main()

