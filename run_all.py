import subprocess
import sys
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

scripts = [
    os.path.join(BASE_DIR, "cpu", "cpu_train.py"),
    os.path.join(BASE_DIR, "gpu", "gpu_train.py"),
    os.path.join(BASE_DIR, "visualization", "plot_results.py"),
    os.path.join(BASE_DIR, "visualization", "cpu_confusion_matrix.py"),
    os.path.join(BASE_DIR, "visualization", "gpu_confusion_matrix.py"),
    os.path.join(BASE_DIR, "generate_report.py")
]

for script_path in scripts:
    print(f"\n>>> Running {os.path.basename(script_path)} ...")
    try:
        subprocess.run([sys.executable, script_path], check=True)
        print(f"✅ {os.path.basename(script_path)} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {os.path.basename(script_path)}: {e}")
        break

print("\nAll scripts executed.")
