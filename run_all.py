import subprocess
import sys
import os
from fpdf import FPDF
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

scripts = [
    os.path.join(BASE_DIR, "cpu", "cpu_train.py"),
    os.path.join(BASE_DIR, "gpu", "gpu_train.py"),
    os.path.join(BASE_DIR, "visualization", "plot_results.py"),
    os.path.join(BASE_DIR, "visualization", "cpu_confusion_matrix.py"),
    os.path.join(BASE_DIR, "visualization", "gpu_confusion_matrix.py"),
    os.path.join(BASE_DIR, "generate_report.py")
]

# Capture terminal output for PDF
terminal_output = []

for script_path in scripts:
    script_name = os.path.basename(script_path)
    print(f"\n>>> Running {script_name} ...")
    terminal_output.append(f">>> Running {script_name} ...\n")
    try:
        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        terminal_output.append(result.stdout)
        print(f"[SUCCESS] {script_name} completed successfully.")
        terminal_output.append(f"[SUCCESS] {script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {script_name} failed:\n{e.stderr}")
        terminal_output.append(f"[ERROR] {script_name} failed:\n{e.stderr}\n")
        break

print("\nAll scripts executed.")

# -----------------------------
# Generate PDF of terminal output
# -----------------------------
pdf_file = os.path.join(RESULTS_DIR, "Terminal_Output_Report.pdf")

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(0, 10, "PDC Project: Terminal Output Report", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.set_font("Helvetica", "", 10)
pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)

# Write captured terminal output
for output in terminal_output:
    pdf.multi_cell(0, 5, output)

pdf.output(pdf_file)
print(f"\n[SUCCESS] Terminal output PDF generated: {pdf_file}")
