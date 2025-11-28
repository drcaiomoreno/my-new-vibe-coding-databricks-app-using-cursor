# Databricks Deployment Guide

This guide walks you through deploying the London Housing Price Predictor app on Databricks.

## Prerequisites

- Databricks workspace access
- Databricks CLI installed (optional, for command-line deployment)
- Git repository with the application code

## Deployment Methods

### Method 1: Databricks UI Deployment

#### Step 1: Upload Code to Databricks

1. **Login to Databricks Workspace**
   - Navigate to your Databricks workspace URL
   - Sign in with your credentials

2. **Create a New Repo**
   - Go to "Repos" in the sidebar
   - Click "Add Repo"
   - Enter your Git repository URL
   - Select the branch (usually `main`)
   - Click "Create Repo"

3. **Verify Files**
   - Ensure all files are present:
     - `app.py`
     - `app.yaml`
     - `requirements.txt`
     - `data/generate_london_housing_data.py`
     - `model/train_model.py`

#### Step 2: Install Dependencies

1. **Create a Cluster** (if not already available)
   - Go to "Compute" → "Create Cluster"
   - Choose runtime version (13.3 LTS or higher recommended)
   - Select cluster size based on your needs
   - Click "Create Cluster"

2. **Install Python Libraries**
   
   Option A: Install via Cluster UI
   - Go to your cluster
   - Click "Libraries" tab
   - Click "Install New"
   - Select "PyPI"
   - Enter package names from `requirements.txt` one by one

   Option B: Install via Notebook
   ```python
   %pip install -r requirements.txt
   ```

#### Step 3: Prepare the Model

1. **Create a Notebook**
   - Create a new Python notebook
   - Attach it to your cluster

2. **Generate Data and Train Model**
   ```python
   # Change to repo directory
   import os
   os.chdir('/Workspace/Repos/<your-username>/<repo-name>')
   
   # Generate data
   %run data/generate_london_housing_data.py
   
   # Train model
   %run model/train_model.py
   ```

3. **Verify Model Files**
   - Check that `model/random_forest.pkl` or `model/gradient_boosting.pkl` exists
   - Check that `model/preprocessors.pkl` exists
   - Check that `data/london_housing_data.csv` exists

#### Step 4: Deploy the App

1. **Create Databricks App**
   - Go to "Apps" in the sidebar
   - Click "Create App"
   - Select "From Repo"

2. **Configure App**
   - **Name**: `london-housing-predictor`
   - **Source**: Select your repo
   - **Path**: Root directory of your repo
   - **Config File**: `app.yaml`
   - **Compute**: Select or create a cluster

3. **Deploy**
   - Click "Create App"
   - Wait for deployment (may take a few minutes)
   - Once deployed, you'll get a URL to access your app

#### Step 5: Access Your App

- Click on the app URL provided by Databricks
- The app should load with the Streamlit interface
- Test the prediction functionality

---

### Method 2: Databricks CLI Deployment

#### Step 1: Install Databricks CLI

```bash
pip install databricks-cli
```

#### Step 2: Configure Authentication

```bash
databricks configure --token
```

Enter:
- Databricks Host (e.g., `https://<workspace-name>.cloud.databricks.com`)
- Personal Access Token (generate from User Settings → Access Tokens)

#### Step 3: Upload Files

```bash
# Create workspace directory
databricks workspace mkdirs /Workspace/london-housing-app

# Upload files
databricks workspace import-dir . /Workspace/london-housing-app
```

#### Step 4: Run Setup via Notebook

Create and run a notebook with:

```python
%sh
cd /Workspace/london-housing-app
python data/generate_london_housing_data.py
python model/train_model.py
```

#### Step 5: Deploy App

Use the Databricks UI as described in Method 1, Step 4.

---

## Configuration

### app.yaml

The `app.yaml` file tells Databricks how to run your app:

```yaml
command: ["streamlit", "run", "app.py", "--server.port", "8080"]
```

### Environment Variables (Optional)

If you need environment variables, create a `.env` file:

```bash
MODEL_PATH=model/
DATA_PATH=data/
LOG_LEVEL=INFO
```

And update `app.yaml`:

```yaml
command: ["streamlit", "run", "app.py", "--server.port", "8080"]
env:
  - MODEL_PATH: model/
  - DATA_PATH: data/
  - LOG_LEVEL: INFO
```

---

## Troubleshooting

### Issue: App Won't Start

**Solution:**
- Check that all dependencies are installed
- Verify `app.yaml` syntax
- Check cluster logs for errors
- Ensure model files exist

### Issue: Import Errors

**Solution:**
- Install missing packages via cluster libraries
- Check Python version compatibility
- Verify all files are uploaded

### Issue: Model Not Found

**Solution:**
- Run the training script first
- Verify model files are in the correct location
- Check file paths in `app.py`

### Issue: Data Loading Errors

**Solution:**
- Generate data using `generate_london_housing_data.py`
- Check CSV file exists in `data/` directory
- Verify file permissions

### Issue: Port Already in Use

**Solution:**
- Change port in `app.yaml` (e.g., 8081, 8082)
- Restart the app

---

## Monitoring and Maintenance

### Viewing Logs

1. Go to Apps → Your App
2. Click "Logs" tab
3. View real-time logs

### Updating the App

1. Update code in your Git repo
2. In Databricks, go to Repos → Your Repo
3. Click "Pull" to get latest changes
4. Restart the app

### Scaling

For better performance:
- Use larger cluster sizes
- Enable autoscaling
- Use Databricks SQL endpoints for data queries

---

## Security Best Practices

1. **Access Control**
   - Use Databricks workspace permissions
   - Restrict app access to authorized users

2. **Data Security**
   - Store sensitive data in Databricks tables
   - Use secrets for API keys
   - Enable audit logging

3. **Model Security**
   - Version control models with MLflow
   - Track model lineage
   - Implement model approval workflows

---

## Cost Optimization

1. **Right-size Clusters**
   - Use smallest cluster that meets performance needs
   - Enable auto-termination

2. **Use Spot Instances**
   - Configure spot instances for non-critical workloads

3. **Monitor Usage**
   - Track DBU consumption
   - Set up usage alerts

---

## Advanced Features

### MLflow Integration

Track model experiments:

```python
import mlflow

with mlflow.start_run():
    # Train model
    model = train_model()
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### Delta Lake Integration

Store data in Delta tables:

```python
# Save to Delta
df.write.format("delta").mode("overwrite").save("/mnt/housing-data")

# Read from Delta
df = spark.read.format("delta").load("/mnt/housing-data")
```

### Scheduled Retraining

Use Databricks Jobs to retrain models:

1. Create a notebook with training code
2. Go to Workflows → Create Job
3. Set schedule (e.g., weekly)
4. Configure notifications

---

## Support

For issues or questions:
- Check Databricks documentation
- Contact Databricks support
- Open an issue in the GitHub repository

---

## Next Steps

After deployment:
1. ✅ Test all features
2. ✅ Monitor performance
3. ✅ Gather user feedback
4. ✅ Plan improvements
5. ✅ Set up monitoring and alerts

Happy deploying! 🚀

