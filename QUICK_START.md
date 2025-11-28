# 🚀 Quick Start Guide

Get your London Housing Price Predictor running in 5 minutes!

## Prerequisites Check

Before starting, ensure you have:
- ✅ Python 3.8 or higher (`python3 --version`)
- ✅ pip package manager (`pip3 --version`)
- ✅ Git (for version control)

## Option 1: Automated Setup (Recommended)

### One-Command Setup

Run the automated setup script:

```bash
./setup.sh
```

This will:
1. Check Python version
2. Optionally create a virtual environment
3. Install all dependencies
4. Generate the dataset
5. Train the model

### After Setup

Run the app:

```bash
streamlit run app.py --server.port 8080
```

Open your browser to: `http://localhost:8080`

---

## Option 2: Manual Setup

### Step 1: Install Dependencies

```bash
pip3 install -r requirements.txt
```

### Step 2: Generate Dataset

```bash
python3 data/generate_london_housing_data.py
```

**Output**: Creates `data/london_housing_data.csv` with 5,000 property records

### Step 3: Train Model

```bash
python3 model/train_model.py
```

**Output**: 
- Creates `model/random_forest.pkl` (or `gradient_boosting.pkl`)
- Creates `model/preprocessors.pkl`
- Displays model performance metrics

### Step 4: Run Application

```bash
streamlit run app.py --server.port 8080
```

**Access**: Open browser to `http://localhost:8080`

---

## Option 3: Using Virtual Environment

### Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install and Run

```bash
pip install -r requirements.txt
python data/generate_london_housing_data.py
python model/train_model.py
streamlit run app.py --server.port 8080
```

---

## Verification Steps

### 1. Check Data Generation

```bash
ls -lh data/london_housing_data.csv
```

Should see a CSV file (~500KB)

### 2. Check Model Files

```bash
ls -lh model/*.pkl
```

Should see:
- `random_forest.pkl` or `gradient_boosting.pkl` (~10-20MB)
- `preprocessors.pkl` (~5KB)

### 3. Test Prediction

```python
# In Python interpreter
import joblib
import pandas as pd

# Load model
model = joblib.load('model/random_forest.pkl')
preprocessors = joblib.load('model/preprocessors.pkl')

print("✅ Model loaded successfully!")
```

---

## Using the Application

### Page 1: Price Prediction

1. Select property details:
   - **Borough**: Choose from 20 London boroughs
   - **Property Type**: Flat, Terraced, Semi-Detached, or Detached
   - **Bedrooms**: 1-5
   - **Bathrooms**: 1-4
   - **Square Feet**: 300-5000
   - **Year Built**: 1900-2024
   - **Distance to Station**: 0.1-2.5 miles
   - **Garden**: Yes/No
   - **Parking**: Yes/No
   - **Energy Rating**: A-G

2. Click "🔮 Predict Price"

3. View results:
   - Predicted price
   - Price per square foot
   - Estimated range (±10%)
   - Property summary

### Page 2: Data Exploration

- View dataset statistics
- Interactive visualizations:
  - Price distribution
  - Borough analysis
  - Property features
  - Feature correlations
- Sample data table

### Page 3: Model Insights

- Feature importance rankings
- Model performance metrics
- Training data statistics

---

## Troubleshooting

### Issue: Dependencies Won't Install

**Solution**:
```bash
pip3 install --upgrade pip
pip3 install --upgrade setuptools wheel
pip3 install -r requirements.txt
```

### Issue: Streamlit Not Found

**Solution**:
```bash
pip3 install streamlit
```

### Issue: Model Not Found Error

**Solution**:
Run the training script first:
```bash
python3 model/train_model.py
```

### Issue: Port Already in Use

**Solution**:
Use a different port:
```bash
streamlit run app.py --server.port 8081
```

### Issue: Import Errors

**Solution**:
Ensure all packages are installed:
```bash
pip3 install pandas numpy scikit-learn plotly joblib
```

---

## Databricks Deployment

### Quick Databricks Setup

1. **Upload to Repos**:
   - Go to Databricks workspace
   - Repos → Add Repo
   - Enter your Git URL

2. **Run Quick Start Notebook**:
   - Open `notebooks/Quick_Start.py`
   - Run all cells

3. **Deploy App**:
   - Apps → Create App
   - Select repo and `app.yaml`
   - Deploy!

For detailed instructions, see: [DATABRICKS_DEPLOYMENT.md](DATABRICKS_DEPLOYMENT.md)

---

## Example Usage

### Example 1: Luxury Westminster Property

```
Borough: Westminster
Type: Detached
Bedrooms: 4
Bathrooms: 3
Square Feet: 2500
Year Built: 2020
Distance to Station: 0.3 miles
Garden: Yes
Parking: Yes
Energy Rating: A

Predicted Price: ~£2,000,000-£2,500,000
```

### Example 2: Affordable Flat

```
Borough: Newham
Type: Flat
Bedrooms: 2
Bathrooms: 1
Square Feet: 650
Year Built: 2010
Distance to Station: 0.5 miles
Garden: No
Parking: No
Energy Rating: C

Predicted Price: ~£300,000-£400,000
```

---

## Testing

Run unit tests:

```bash
python3 tests/test_model.py
```

---

## Configuration

### Change Port

Edit `app.yaml`:
```yaml
command: ["streamlit", "run", "app.py", "--server.port", "8081"]
```

### Change Dataset Size

Edit `data/generate_london_housing_data.py`:
```python
df = generate_london_housing_data(10000)  # Generate 10,000 samples
```

### Adjust Model Parameters

Edit `model/train_model.py`:
```python
model = RandomForestRegressor(
    n_estimators=200,  # More trees
    max_depth=25,      # Deeper trees
    ...
)
```

---

## Next Steps

After successful setup:

1. ✅ **Explore the app**: Try different property configurations
2. ✅ **View visualizations**: Check data exploration page
3. ✅ **Review model**: Look at feature importance
4. ✅ **Deploy to Databricks**: Follow deployment guide
5. ✅ **Customize**: Add new features or modify the UI

---

## Need Help?

- 📖 Read [README.md](README.md) for detailed documentation
- 🚀 Check [DATABRICKS_DEPLOYMENT.md](DATABRICKS_DEPLOYMENT.md) for deployment
- 📊 Review [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for architecture
- 💬 Open an issue on GitHub

---

## Success Checklist

- [ ] Dependencies installed
- [ ] Data generated
- [ ] Model trained
- [ ] App running locally
- [ ] Tested predictions
- [ ] Viewed visualizations
- [ ] Ready for Databricks deployment

---

**Congratulations! Your London Housing Price Predictor is ready to use!** 🎉

For best results:
- Ensure all dependencies are installed
- Generate fresh data periodically
- Retrain model with new data
- Monitor app performance

Enjoy predicting London housing prices! 🏠

