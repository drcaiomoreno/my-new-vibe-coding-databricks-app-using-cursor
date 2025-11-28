# ⚡ QUICK FIX: "Model not found" Error

## 🎯 The Problem

You deployed to Databricks and got:
```
❌ Model not found! Please train the model first by running: python model/train_model.py
```

## ✅ The Solution (5 Minutes)

### Step 1: Open Setup Notebook

In your Databricks workspace:
```
Navigate to: Repos → <your-username> → my-new-vibe-coding-databricks-app-using-cursor → notebooks → Setup_Models.py
```

**Click to open:** `Setup_Models.py`

---

### Step 2: Update the Path

In **Cell 2** of the notebook, change this line:
```python
REPO_PATH = "/Workspace/Repos/<your-username>/my-new-vibe-coding-databricks-app-using-cursor"
```

**Replace `<your-username>`** with your actual Databricks username!

**Example:**
```python
REPO_PATH = "/Workspace/Repos/john.doe/my-new-vibe-coding-databricks-app-using-cursor"
```

---

### Step 3: Run the Notebook

1. **Attach to any cluster**
2. **Click "Run All"** at the top
3. **Wait 2-3 minutes** for completion

**You should see:**
- ✅ Data generation complete! (5,000 records)
- ✅ Model training complete!
- ✅ Model files created (10-20 MB)
- 🎉 Setup complete!

---

### Step 4: Restart Your App

1. Go to **Apps** in Databricks sidebar
2. Find your app
3. Click **Restart**
4. Wait for app to restart

---

### Step 5: Test It!

Open your app URL and try predicting a price:
- **Borough:** Westminster
- **Type:** Flat
- **Bedrooms:** 2
- Click **"Predict Price"**

**Should work now!** ✅

---

## 🔄 Alternative: One-Cell Quick Fix

If the notebook doesn't work, create a **new notebook** with this single cell:

```python
# Replace <your-username> with your actual username!
import os
os.chdir("/Workspace/Repos/<your-username>/my-new-vibe-coding-databricks-app-using-cursor")

# Generate data
print("📊 Generating data...")
exec(open('data/generate_london_housing_data.py').read())

# Train model  
print("🤖 Training model...")
exec(open('model/train_model.py').read())

# Verify
import glob
print("\n✅ Files created:")
for f in glob.glob('model/*.pkl'):
    size = os.path.getsize(f) / (1024*1024)
    print(f"  • {f} ({size:.1f} MB)")

print("\n🎉 Done! Restart your app now.")
```

**Run the cell**, then restart your app.

---

## ❓ Still Not Working?

### Check These:

1. **Username correct?**
   - Run this in a cell: `import os; print(os.path.expanduser('~'))`
   - Use the path shown

2. **Files created?**
   ```python
   import os
   print("Data:", os.path.exists('data/london_housing_data.csv'))
   print("Model:", os.path.exists('model/random_forest.pkl'))
   ```
   Should show: `Data: True`, `Model: True`

3. **App restarted?**
   - Must restart AFTER creating model files

---

## 📚 More Help

- **Detailed guide:** See `DATABRICKS_TROUBLESHOOTING.md`
- **Full deployment:** See `DATABRICKS_DEPLOYMENT.md`

---

## 🎉 Success!

Once working, you should see:
- ✅ App loads without errors
- ✅ Three pages accessible
- ✅ Price predictions work
- ✅ Visualizations display

**Enjoy your London Housing Price Predictor!** 🏠

