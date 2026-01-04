
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import pandas as pd
import platform
import lightgbm as lgb
from preprocessing.preprocess import load_and_preprocess
from sklearn.metrics import accuracy_score
import pyopencl as cl

# Paths
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'healthcare-dataset-stroke-data.csv'))
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
os.makedirs(results_dir, exist_ok=True)

# Load and preprocess
X_train, X_test, y_train, y_test = load_and_preprocess(data_path)

# GPU Model
gpu_model = lgb.LGBMClassifier(device="gpu", gpu_platform_id=0, gpu_device_id=0, n_estimators=200, random_state=42)

print("\nTraining on GPU...")
start_time = time.time()
gpu_model.fit(X_train, y_train)
end_time = time.time()

training_time = end_time - start_time
preds = gpu_model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

# Detect GPU
gpus = []
try:
    for platform_ in cl.get_platforms():
        for device in platform_.get_devices(device_type=cl.device_type.GPU):
            gpus.append(f"{device.name} | Vendor: {device.vendor}")
    gpu_name = gpus[0] if gpus else "No GPU detected"
except:
    gpu_name = "No GPU detected"

# Save results
results = {
    "Model": "LightGBM",
    "Device": "GPU",
    "Training Time (s)": training_time,
    "Accuracy (%)": accuracy*100,
    "GPU": gpu_name
}

pd.DataFrame([results]).to_json(os.path.join(results_dir, "gpu_results2.json"), orient="records")
print("\nGPU Training Complete")
print(results)
