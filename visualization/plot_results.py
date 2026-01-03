# ==============================
# Visualization Script
# ==============================
import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
plots_dir = results_dir  # save plots in results folder

# Load results
cpu_res = pd.read_json(os.path.join(results_dir, "cpu_results.json"))
gpu_res = pd.read_json(os.path.join(results_dir, "gpu_results.json"))

# Combine
results = pd.concat([cpu_res, gpu_res], ignore_index=True)

# Training Time Comparison
plt.figure(figsize=(6,4))
plt.bar(results["Device"], results["Training Time (s)"], color=["skyblue", "lightgreen"])
plt.title("CPU vs GPU Training Time")
plt.ylabel("Time (seconds)")
plt.xlabel("Device")
plt.savefig(os.path.join(plots_dir, "training_time_comparison.png"))
plt.close()

# Accuracy Comparison
plt.figure(figsize=(6,4))
plt.bar(results["Device"], results["Accuracy (%)"], color=["skyblue", "lightgreen"])
plt.title("CPU vs GPU Accuracy")
plt.ylabel("Accuracy (%)")
plt.xlabel("Device")
plt.savefig(os.path.join(plots_dir, "accuracy_comparison.png"))
plt.close()

# Speedup Ratio
cpu_time = results[results["Device"]=="CPU"]["Training Time (s)"].values[0]
gpu_time = results[results["Device"]=="GPU"]["Training Time (s)"].values[0]
speedup = cpu_time / gpu_time

plt.figure(figsize=(6,4))
plt.bar(["CPU/GPU"], [speedup], color="orange")
plt.title("CPU/GPU Speedup Ratio")
plt.ylabel("Speedup (x)")
plt.savefig(os.path.join(plots_dir, "speedup_ratio.png"))
plt.close()

print("\nAll plots saved in results folder.")
