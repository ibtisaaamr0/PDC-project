# gpu_confusion_matrix.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib  # to load saved GPU model

# Paths
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
preprocessed_file = os.path.join(results_dir, "preprocessed_data.csv")
gpu_model_file = os.path.join(results_dir, "gpu_model.pkl")  # make sure you saved your GPU model as .pkl

# ==============================
# Load preprocessed test data
# ==============================
df = pd.read_csv(preprocessed_file)

# Ensure stroke is numeric
df['stroke'] = pd.to_numeric(df['stroke'], errors='coerce')

# Filter test set
test_df = df[df['split'] == 'test'].copy()

# Drop any rows with NaN in stroke
test_df = test_df.dropna(subset=['stroke'])

X_test = test_df.drop(['stroke','split'], axis=1)
y_test = test_df['stroke'].astype(int)  # ensure integer labels

# Filter test set and remove NaNs
test_df = df[df['split'] == 'test'].copy()
test_df = test_df.dropna(subset=['stroke'])
X_test = test_df.drop(['stroke', 'split'], axis=1)
y_test = test_df['stroke'].astype(int)

# Load trained GPU model
gpu_model = joblib.load(gpu_model_file)

# Predict on test set
y_pred = gpu_model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("GPU Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Save figure
os.makedirs(results_dir, exist_ok=True)
plt.savefig(os.path.join(results_dir, "gpu_confusion_matrix.png"))
plt.close()
print("GPU confusion matrix saved at:", os.path.join(results_dir, "gpu_confusion_matrix.png"))
