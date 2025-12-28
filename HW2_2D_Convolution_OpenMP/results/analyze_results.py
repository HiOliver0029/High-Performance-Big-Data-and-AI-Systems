import pandas as pd 
import matplotlib.pyplot as plt 
 
# Read performance data 
serial_df = pd.read_csv('serial_performance_2025-09-22_12-49-59.csv') 
openmp_df = pd.read_csv('openmp_performance_2025-09-22_12-49-59.csv') 
benchmark_df = pd.read_csv('benchmark_summary_2025-09-22_12-49-59.csv') 
 
print("Serial Performance Summary:") 
print(serial_df.groupby(['Matrix Size', 'Kernel Size'])['Execution Time (us)'].mean()) 
 
print("\nOpenMP Performance Summary:") 
print(openmp_df.groupby(['Threads', 'Matrix Size'])['Execution Time (us)'].mean()) 
