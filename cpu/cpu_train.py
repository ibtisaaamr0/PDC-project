# ==============================
# CPU Training Script
# ==============================
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import platform
import lightgbm as lgb
from preprocessing.preprocess import load_and_preprocess
from sklearn.metrics import accuracy_score

# Paths
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'healthcare-dataset-stroke-data.csv'))
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(results_dir, exist_ok=True)

# Load and preprocess
X_train, X_test, y_train, y_test = load_and_preprocess(data_path)

# CPU Model
cpu_model = lgb.LGBMClassifier(device="cpu", n_estimators=200, random_state=42)

print("\nTraining on CPU...")
start_time = time.time()
cpu_model.fit(X_train, y_train)
end_time = time.time()

training_time = end_time - start_time
preds = cpu_model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

# Save results
results = {
    "Model": "LightGBM",
    "Device": "CPU",
    "Training Time (s)": training_time,
    "Accuracy (%)": accuracy*100,
    "CPU": platform.processor()
}

pd.DataFrame([results]).to_json(os.path.join(results_dir, "cpu_results.json"), orient="records")
print("\nCPU Training Complete")
print(results)
