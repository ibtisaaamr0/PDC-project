# gpu/gpu_train.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import platform
import lightgbm as lgb
import joblib
from preprocessing.preprocess import load_and_preprocess
from sklearn.metrics import accuracy_score , f1_score
import pyopencl as cl

# ==============================
# Paths
# ==============================
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'healthcare-dataset-stroke-data.csv'))
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(results_dir, exist_ok=True)

# ==============================
# Load and preprocess dataset
# ==============================
X_train, X_test, y_train, y_test = load_and_preprocess(data_path)

# Show preprocessed data snapshot
print("\n=== Preprocessed Data Snapshot ===")
print("X_train (first 5 rows):\n", X_train.head())
print("\ny_train distribution:\n", y_train.value_counts())
print("\nX_test (first 5 rows):\n", X_test.head())
print("\ny_test distribution:\n", y_test.value_counts())

# Save preprocessed data in one CSV (train + test)
preprocessed_df = pd.concat([
    X_train.assign(stroke=y_train, split='train'),
    X_test.assign(stroke=y_test, split='test')
], axis=0)
preprocessed_file = os.path.join(results_dir, "preprocessed_data.csv")
preprocessed_df.to_csv(preprocessed_file, index=False)
print(f"\n✅ Preprocessed data saved to: {preprocessed_file}")

# ==============================
# GPU Model Training
# ==============================
gpu_model = lgb.LGBMClassifier(device="gpu", gpu_platform_id=0, gpu_device_id=0, n_estimators=200, random_state=42)

print("\nTraining on GPU...")
start_time = time.time()
gpu_model.fit(X_train, y_train)
end_time = time.time()

training_time = end_time - start_time
preds = gpu_model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)  # Compute F1 Score


# Detect GPU
gpus = []
try:
    for platform_ in cl.get_platforms():
        for device in platform_.get_devices(device_type=cl.device_type.GPU):
            gpus.append(f"{device.name} | Vendor: {device.vendor}")
    gpu_name = gpus[0] if gpus else "No GPU detected"
except:
    gpu_name = "No GPU detected"

# ==============================
# Save Training Results
# ==============================
results = {
    "Model": "LightGBM",
    "Device": "GPU",
    "Training Time (s)": training_time,
    "Accuracy (%)": accuracy*100,
    "F1 Score": f1,  # <-- Added
    "GPU": gpu_name
}

results_file = os.path.join(results_dir, "gpu_results.json")
pd.DataFrame([results]).to_json(results_file, orient="records")
print("\n✅ GPU Training Complete")
print(results)
print(f"Results saved to: {results_file}")

# ==============================
# Save Trained GPU Model
# ==============================
model_file = os.path.join(results_dir, "gpu_model.pkl")
joblib.dump(gpu_model, model_file)
print(f"✅ GPU model saved to: {model_file}")
