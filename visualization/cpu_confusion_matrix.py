# cpu_confusion_matrix.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

# ==============================
# Paths
# ==============================
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(base_dir, "results")

cpu_model_file = os.path.join(results_dir, "cpu_model.pkl")
gpu_model_file = os.path.join(results_dir, "gpu_model.pkl")
preprocessed_file = os.path.join(results_dir, "preprocessed_data.csv")  

# Load preprocessed test data
df = pd.read_csv(preprocessed_file)

# Ensure stroke is numeric
df['stroke'] = pd.to_numeric(df['stroke'], errors='coerce')

# Filter test set
test_df = df[df['split'] == 'test'].copy()

# Drop any rows with NaN in stroke
test_df = test_df.dropna(subset=['stroke'])

X_test = test_df.drop(['stroke','split'], axis=1)
y_test = test_df['stroke'].astype(int)  


# Load CPU model
cpu_model = joblib.load(cpu_model_file)

# Predict and create confusion matrix
y_pred = cpu_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("CPU Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Save figure
cm_file = os.path.join(results_dir, "cpu_confusion_matrix.png")
plt.savefig(cm_file)
plt.close()
print(f"✅ CPU confusion matrix saved to: {cm_file}")
