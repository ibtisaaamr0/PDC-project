# generate_report.py
import os
import pandas as pd
from fpdf import FPDF
from datetime import datetime

# Paths
results_dir = os.path.join(os.path.dirname(__file__), 'results')
pdf_file = os.path.join(results_dir, "PDC_Project_Report.pdf")

# Load results (CPU + GPU)
# Load results (CPU + GPU)
cpu_res = pd.read_json(os.path.join(results_dir, "cpu_results.json"))
gpu_res = pd.read_json(os.path.join(results_dir, "gpu_results.json"))

# Ensure numeric columns are correct type
for col in ["Training Time (s)", "Accuracy (%)", "F1 Score"]:
    cpu_res[col] = pd.to_numeric(cpu_res[col], errors='coerce')
    gpu_res[col] = pd.to_numeric(gpu_res[col], errors='coerce')

results = pd.concat([cpu_res, gpu_res], ignore_index=True)


# Preprocessed data snapshot (first 5 rows)
preprocessed_file = os.path.join(results_dir, "preprocessed_data.csv")
preprocessed_df = pd.read_csv(preprocessed_file)
preprocessed_head = preprocessed_df.head().to_string()

# Determine faster device dynamically
cpu_time = results.loc[results['Device'] == 'CPU', 'Training Time (s)'].values[0]
gpu_time = results.loc[results['Device'] == 'GPU', 'Training Time (s)'].values[0]
if cpu_time < gpu_time:
    faster_device = "CPU"
    speedup = gpu_time / cpu_time
else:
    faster_device = "GPU"
    speedup = cpu_time / gpu_time

# Initialize PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Parallel & Distributed Computing Project Report", ln=True, align="C")
pdf.set_font("Arial", "", 12)
pdf.ln(5)
pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
pdf.ln(5)

# Dataset snapshot
pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "Dataset Snapshot", ln=True)
pdf.set_font("Arial", "", 11)
pdf.multi_cell(0, 5, preprocessed_head)
pdf.ln(5)

# CPU & GPU Results Table (with F1 Score)
pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "CPU & GPU Results", ln=True)
pdf.set_font("Arial", "", 11)
for idx, row in results.iterrows():
    device_name = row['CPU'] if 'CPU' in row and pd.notna(row['CPU']) else row['GPU']
    device_name = device_name if pd.notna(device_name) else "N/A"

    pdf.multi_cell(0, 5,
        f"Model: {row['Model']}\n"
        f"Device: {row['Device']}\n"
        f"Training Time (s): {row['Training Time (s)']:.4f}\n"
        f"Accuracy (%): {row['Accuracy (%)']:.2f}\n"
        f"F1 Score: {row.get('F1 Score', 0):.2f}\n"
        f"Processor/GPU: {device_name}\n"
    )
    pdf.ln(2)


# Plots
pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "Performance Plots", ln=True)
plot_files = [
    "training_time_comparison.png",
    "accuracy_comparison.png",
    "speedup_ratio.png",
    "cpu_confusion_matrix.png",
    "gpu_confusion_matrix.png"
]
for plot in plot_files:
    plot_path = os.path.join(results_dir, plot)
    if os.path.exists(plot_path):
        pdf.image(plot_path, w=180)
        pdf.ln(5)

# Dynamic Conclusion with actual training times
cpu_time_val = results.loc[results['Device'] == 'CPU', 'Training Time (s)'].values[0]
gpu_time_val = results.loc[results['Device'] == 'GPU', 'Training Time (s)'].values[0]

pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "Conclusion & Observations", ln=True)
pdf.set_font("Arial", "", 11)
pdf.multi_cell(0, 5,
    f"1. CPU and GPU models were trained on the preprocessed dataset.\n"
    f"2. CPU training time: {cpu_time_val:.4f} seconds.\n"
    f"3. GPU training time: {gpu_time_val:.4f} seconds.\n"
    f"4. Accuracy and F1 score of both CPU and GPU models are shown above.\n"
    f"5. Confusion matrices visualize correct vs incorrect predictions.\n"
    "6. Further improvements could include hyperparameter tuning, feature engineering, and larger datasets."
)

pdf.output(pdf_file)
print(f"\n✅ PDF report generated: {pdf_file}")
