# ==============================
# Generate PDF Report for CPU vs GPU ML
# ==============================

import pandas as pd
from fpdf import FPDF
import os

# Paths
results_dir = os.path.abspath("results")
pdf_path = os.path.join(results_dir, "CPU_vs_GPU_Report.pdf")

# Load results
cpu_results = pd.read_json(os.path.join(results_dir, "cpu_results.json"))
gpu_results = pd.read_json(os.path.join(results_dir, "gpu_results.json"))

# Combine results
all_results = pd.concat([cpu_results, gpu_results], ignore_index=True)

# Initialize PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "CPU vs GPU ML Performance Report", ln=True, align="C")
pdf.ln(5)

pdf.set_font("Arial", "", 12)
pdf.cell(0, 8, "Model: LightGBM", ln=True)
pdf.ln(5)

# Add results table
for index, row in all_results.iterrows():
    device_info = row.get("CPU") if row["Device"] == "CPU" else row.get("GPU")
    pdf.cell(0, 8, f"{row['Device']} Training:", ln=True)
    pdf.cell(0, 8, f"Training Time: {row['Training Time (s)']:.3f} s", ln=True)
    pdf.cell(0, 8, f"Accuracy: {row['Accuracy (%)']:.2f}%", ln=True)
    pdf.cell(0, 8, f"Device Name: {device_info}", ln=True)
    pdf.ln(5)

# Save PDF
pdf.output(pdf_path)
print(f"\nPDF report generated at: {pdf_path}")
