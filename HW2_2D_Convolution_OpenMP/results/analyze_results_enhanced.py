# High Performance Computing HW2 - Enhanced Analysis Script 
# Generated on 2025-09-22 15:10:03 
 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
import os 
 
def load_latest_data(): 
    """Load the most recent CSV files""" 
    try: 
        # Find CSV files 
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')] 
        serial_files = [f for f in csv_files if 'serial_performance' in f] 
        openmp_files = [f for f in csv_files if 'openmp_performance' in f] 
 
        if not serial_files or not openmp_files: 
            print("CSV files not found!") 
            return None, None 
 
        # Load latest files 
        serial_df = pd.read_csv(sorted(serial_files)[-1]) 
        openmp_df = pd.read_csv(sorted(openmp_files)[-1]) 
 
        # Clean data 
        serial_df['Execution_Time_us'] = pd.to_numeric(serial_df['Execution_Time_us'], errors='coerce') 
        openmp_df['Execution_Time_us'] = pd.to_numeric(openmp_df['Execution_Time_us'], errors='coerce') 
 
        print(f"Loaded: {sorted(serial_files)[-1]}, {sorted(openmp_files)[-1]}") 
        return serial_df, openmp_df 
 
    except Exception as e: 
        print(f"Error: {e}") 
        return None, None 
 
def analyze_and_plot(): 
    """Main analysis function""" 
    serial_df, openmp_df = load_latest_data() 
 
    if serial_df is None or openmp_df is None: 
        return 
 
    # Display basic statistics 
    print("="*60) 
    print("PERFORMANCE ANALYSIS REPORT") 
    print("="*60) 
 
    print("\nSerial Performance Summary:") 
    print(serial_df.groupby(['Matrix_Size', 'Kernel_Size'])['Execution_Time_us'].describe()) 
 
    print("\nOpenMP Performance Summary:") 
    print(openmp_df.groupby(['Threads', 'Matrix_Size'])['Execution_Time_us'].describe()) 
 
    # Calculate speedup 
    print("\nSpeedup Analysis:") 
    for matrix in sorted(openmp_df['Matrix_Size'].unique()): 
        for kernel in sorted(openmp_df['Kernel_Size'].unique()): 
            subset = openmp_df[(openmp_df['Matrix_Size'] == matrix) & (openmp_df['Kernel_Size'] == kernel)] 
            baseline = subset[subset['Threads'] == 1]['Execution_Time_us'] 
            if len(baseline) > 0: 
                baseline_time = baseline.iloc[0] 
                print(f"\nMatrix {matrix}x{matrix}, Kernel {kernel}x{kernel}:") 
                for threads in sorted(subset['Threads'].unique()): 
                    thread_time = subset[subset['Threads'] == threads]['Execution_Time_us'] 
                    if len(thread_time) > 0: 
                        speedup = baseline_time / thread_time.iloc[0] 
                        print(f"  {threads} threads: {speedup:.2f}x speedup") 
 
if __name__ == "__main__": 
    analyze_and_plot() 
