# CPU vs GPU Machine Learning Performance Comparison

## Course
Parallel & Distributed Computing (CS09645)

## Description
This project compares CPU and GPU performance for machine learning model training using LightGBM.
GPU acceleration is achieved using OpenCL on an AMD GPU.

## Steps to Run
1. Install requirements
   pip install -r requirements.txt

2. Preprocess data
   python preprocessing/preprocess.py

3. Train on CPU
   python cpu/cpu_train.py

4. Train on GPU
   python gpu/gpu_train.py

5. Generate graphs
   python visualization/plot_results.py

## Outputs
- Training time comparison
- Accuracy comparison
- Speedup ratio
