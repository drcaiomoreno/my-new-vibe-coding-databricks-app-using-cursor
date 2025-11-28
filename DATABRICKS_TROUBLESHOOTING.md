# 🔧 Databricks App Troubleshooting Guide

## ❌ Error: "Model not found! Please train the model first"

This is the **most common issue** when deploying to Databricks. Here's why and how to fix it:

### 🔍 **Why This Happens**

The model files (`*.pkl`) are:
- ✅ Generated locally when you run training
- ❌ **NOT committed to Git** (they're in `.gitignore`)
- ❌ **NOT automatically created** in Databricks

**You must generate and train the model IN Databricks before the app can run.**

---

## ✅ **Solution: 3 Methods**

### **Method 1: Use Setup Notebook (Easiest)**

1. **In Databricks, go to your repo:**
   ```
   Workspace → Repos → <your-username> → my-new-vibe-coding-databricks-app-using-cursor
   ```

2. **Open the setup notebook:**
   ```
   notebooks/Setup_Models.py
   ```

3. **Update the path in Cell 2:**
   ```python
   REPO_PATH = "/Workspace/Repos/<your-username>/my-new-vibe-coding-databricks-app-using-cursor"
   ```
   Replace `<your-username>` with your actual username!

4. **Attach to a cluster** (any cluster will work)

5. **Click "Run All"** at the top

6. **Wait for completion** (2-3 minutes)
   - Cell 1: Sets working directory ✅
   - Cell 2: Generates 5,000 property records ✅
   - Cell 3: Trains ML model ✅
   - Cell 4: Verifies model files ✅

7. **Verify success:**
   - You should see "🎉 Setup complete!"
   - Model files listed (~10-20 MB each)

8. **Restart your app:**
   - Go to **Apps** → Your App
   - Click **Restart** (or deploy if not yet created)

9. **Test the app** - should work now! ✅

---

### **Method 2: Using Databricks Notebook (Manual)**

If the setup notebook doesn't work, create a new notebook:

1. **Create new Python notebook** in Databricks

2. **Cell 1 - Set path:**
   ```python
   import os
   REPO_PATH = "/Workspace/Repos/<your-username>/my-new-vibe-coding-databricks-app-using-cursor"
   os.chdir(REPO_PATH)
   print(f"Working in: {os.getcwd()}")
   ```

3. **Cell 2 - Generate data:**
   ```python
   %run data/generate_london_housing_data.py
   ```

4. **Cell 3 - Train model:**
   ```python
   %run model/train_model.py
   ```

5. **Cell 4 - Verify:**
   ```python
   import os
   import glob
   
   print("Model files:")
   for f in glob.glob('model/*.pkl'):
       size = os.path.getsize(f) / (1024*1024)
       print(f"  ✅ {f} ({size:.1f} MB)")
   ```

6. **Run all cells**

7. **Restart your app**

---

### **Method 3: Using init_databricks.py Script**

1. **In Databricks notebook, run:**
   ```python
   import os
   os.chdir("/Workspace/Repos/<your-username>/my-new-vibe-coding-databricks-app-using-cursor")
   
   exec(open('init_databricks.py').read())
   ```

2. **Wait for completion**

3. **Restart your app**

---

## 🔍 **Verification Steps**

After running setup, verify the files exist:

```python
import os

# Check data file
if os.path.exists('data/london_housing_data.csv'):
    size = os.path.getsize('data/london_housing_data.csv') / 1024
    print(f"✅ Data file: {size:.0f} KB")
else:
    print("❌ Data file missing!")

# Check model files
model_files = [
    'model/random_forest.pkl',
    'model/gradient_boosting.pkl',
    'model/preprocessors.pkl'
]

for f in model_files:
    if os.path.exists(f):
        size = os.path.getsize(f) / (1024*1024)
        print(f"✅ {f}: {size:.1f} MB")
```

**Expected output:**
```
✅ Data file: ~500 KB
✅ model/random_forest.pkl: 10-20 MB (or gradient_boosting.pkl)
✅ model/preprocessors.pkl: 0.01 MB
```

---

## 🔄 **App Deployment Checklist**

- [ ] 1. **Repo uploaded** to Databricks
- [ ] 2. **Dependencies installed** (happens automatically with `requirements.txt`)
- [ ] 3. **Data generated** (`data/london_housing_data.csv` exists)
- [ ] 4. **Model trained** (`model/*.pkl` files exist)
- [ ] 5. **App created** in Databricks Apps
- [ ] 6. **App started/restarted** after model creation

---

## 📝 **Common Issues & Solutions**

### Issue 1: "No module named 'pandas'"
**Solution:** Dependencies should install automatically, but if not:
```python
%pip install -r requirements.txt
dbutils.library.restartPython()
```

### Issue 2: "Permission denied"
**Solution:** Ensure you have write access to the repo directory
```python
import os
os.chmod('data', 0o755)
os.chmod('model', 0o755)
```

### Issue 3: "File not found: data/generate_london_housing_data.py"
**Solution:** Check you're in the correct directory:
```python
import os
print(os.getcwd())
print(os.listdir('.'))
```

### Issue 4: Training takes too long
**Solution:** Reduce dataset size temporarily in `generate_london_housing_data.py`:
```python
# In the notebook cell, before running:
import sys
sys.path.append('data')
from generate_london_housing_data import generate_london_housing_data

# Generate smaller dataset for quick testing
df = generate_london_housing_data(1000)  # Only 1000 records
df.to_csv('data/london_housing_data.csv', index=False)
```

Then train:
```python
%run model/train_model.py
```

### Issue 5: "Model loaded but predictions fail"
**Solution:** Check feature names match:
```python
import joblib
preprocessors = joblib.load('model/preprocessors.pkl')
print("Expected features:", preprocessors['feature_names'])
```

---

## 🎯 **Quick Fix Script**

If you just want a one-liner to fix everything, use this in a Databricks notebook:

```python
# One-cell solution
import os
os.chdir("/Workspace/Repos/<your-username>/my-new-vibe-coding-databricks-app-using-cursor")

# Generate data
exec(open('data/generate_london_housing_data.py').read())

# Train model
exec(open('model/train_model.py').read())

# Verify
import glob
print("\n✅ Files created:")
for f in glob.glob('model/*.pkl') + glob.glob('data/*.csv'):
    print(f"  • {f}")
    
print("\n🎉 Done! Restart your app now.")
```

**Remember to replace `<your-username>`!**

---

## 🔄 **Workflow After Code Changes**

If you update your code and redeploy:

1. **Pull latest code:**
   - Databricks will auto-pull from Git
   - Or manually: Repos → Your Repo → "Pull"

2. **If you changed data generation or training:**
   - Re-run the setup notebook
   - Restart the app

3. **If you only changed app.py:**
   - Just restart the app
   - No need to retrain

---

## 🆘 **Still Having Issues?**

### Check App Logs

1. Go to **Apps** → Your App
2. Click **"Logs"** tab
3. Look for error messages

### Common log errors:

**"FileNotFoundError: model/random_forest.pkl"**
→ Run setup notebook

**"ModuleNotFoundError"**
→ Check `requirements.txt` includes all packages

**"Permission denied"**
→ Check cluster has write access to repo

---

## 📊 **Recommended Workflow**

### First-Time Setup:
```
1. Upload repo to Databricks
2. Run notebooks/Setup_Models.py
3. Create App from repo
4. Deploy with app.yaml
5. ✅ App runs!
```

### After Updates:
```
1. Push changes to Git
2. Pull in Databricks
3. If data/model changed → Re-run setup notebook
4. Restart app
5. ✅ Updated app runs!
```

---

## 💡 **Pro Tips**

1. **Use a dedicated cluster** for the setup notebook to avoid conflicts

2. **Save the setup notebook output** for future reference

3. **Add verification** to your app startup:
   ```python
   import os
   if not os.path.exists('model/random_forest.pkl') and not os.path.exists('model/gradient_boosting.pkl'):
       st.error("⚠️ Models not found! Run notebooks/Setup_Models.py first.")
       st.stop()
   ```

4. **Consider using MLflow** for model versioning in production

5. **Automate with Jobs:**
   - Create a Databricks Job that runs setup notebook
   - Schedule it to run before app deployment
   - Link the job to your app lifecycle

---

## 📞 **Need More Help?**

- Check the main **DATABRICKS_DEPLOYMENT.md** guide
- Review **QUICK_START.md** for local testing
- Open an issue on GitHub with:
  - Error message
  - Steps you tried
  - Databricks runtime version
  - Cluster configuration

---

## ✅ **Success Indicators**

You'll know everything is working when:

- ✅ Setup notebook runs without errors
- ✅ Model files show in file browser
- ✅ App starts without "model not found" error
- ✅ Prediction page accepts input
- ✅ Price predictions appear
- ✅ All three pages load correctly

---

## 🎉 **After Successful Setup**

Once your app is running:

1. **Test all features:**
   - Try a prediction
   - View data exploration
   - Check model insights

2. **Share the URL** with your team

3. **Set up monitoring** (optional)

4. **Consider adding authentication** for production use

---

**Good luck! Your app should be working now.** 🚀

If you followed Method 1 (Setup Notebook) and it still doesn't work, please share:
- The error message from the notebook
- The error from the app logs
- Your Databricks runtime version

We'll get it fixed!

