# High Performance Computing HW2 - Enhanced Analysis Script 
# Updated with comprehensive plotting and analysis

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
import os 
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

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
        print(f"Serial data points: {len(serial_df)}")
        print(f"OpenMP data points: {len(openmp_df)}")
        return serial_df, openmp_df 
 
    except Exception as e: 
        print(f"Error: {e}") 
        return None, None

def plot_serial_performance(serial_df):
    """Plot serial performance analysis"""
    if serial_df is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Serial Performance Analysis', fontsize=16, fontweight='bold')
    
    kernels = sorted(serial_df['Kernel_Size'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, kernel in enumerate(kernels):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        data = serial_df[serial_df['Kernel_Size'] == kernel].sort_values('Matrix_Size')
        
        ax.loglog(data['Matrix_Size'], data['Execution_Time_us'], 'o-', 
                 color=colors[i], linewidth=2, markersize=8, 
                 label=f'Kernel {kernel}x{kernel}')
        
        ax.set_title(f'Kernel {kernel}x{kernel}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('Execution Time (μs)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add trend line
        if len(data) > 1:
            z = np.polyfit(np.log(data['Matrix_Size']), np.log(data['Execution_Time_us']), 1)
            ax.loglog(data['Matrix_Size'], np.exp(z[1]) * data['Matrix_Size']**z[0], 
                     '--', alpha=0.7, color=colors[i], 
                     label=f'Trend O(n^{z[0]:.2f})')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig('serial_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_speedup_analysis(openmp_df):
    """Plot comprehensive speedup analysis"""
    if openmp_df is None:
        return
    
    # Calculate speedup data
    speedup_data = []
    
    for matrix in sorted(openmp_df['Matrix_Size'].unique()):
        for kernel in sorted(openmp_df['Kernel_Size'].unique()):
            subset = openmp_df[(openmp_df['Matrix_Size'] == matrix) & 
                              (openmp_df['Kernel_Size'] == kernel)]
            
            baseline_data = subset[subset['Threads'] == 1]['Execution_Time_us']
            if len(baseline_data) > 0:
                baseline_time = baseline_data.iloc[0]
                
                for _, row in subset.iterrows():
                    if pd.notna(row['Execution_Time_us']) and row['Execution_Time_us'] > 0:
                        speedup = baseline_time / row['Execution_Time_us']
                        efficiency = speedup / row['Threads'] * 100
                        
                        speedup_data.append({
                            'Matrix_Size': matrix,
                            'Kernel_Size': kernel,
                            'Threads': row['Threads'],
                            'Speedup': speedup,
                            'Efficiency': efficiency,
                            'Execution_Time_us': row['Execution_Time_us']
                        })
    
    speedup_df = pd.DataFrame(speedup_data)
    
    # Plot 1: Speedup curves
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('OpenMP Speedup Analysis', fontsize=16, fontweight='bold')
    
    matrices = sorted(speedup_df['Matrix_Size'].unique())[:4]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, matrix in enumerate(matrices):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        matrix_data = speedup_df[speedup_df['Matrix_Size'] == matrix]
        
        for j, kernel in enumerate(sorted(matrix_data['Kernel_Size'].unique())):
            kernel_data = matrix_data[matrix_data['Kernel_Size'] == kernel].sort_values('Threads')
            
            ax.plot(kernel_data['Threads'], kernel_data['Speedup'], 'o-',
                   color=colors[j], linewidth=2, markersize=6,
                   label=f'Kernel {kernel}x{kernel}')
        
        # Ideal speedup line
        max_threads = matrix_data['Threads'].max()
        ax.plot([1, max_threads], [1, max_threads], 'k--', alpha=0.5, 
               linewidth=2, label='Ideal Speedup')
        
        ax.set_title(f'Matrix {matrix}x{matrix}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Threads')
        ax.set_ylabel('Speedup')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig('speedup_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Efficiency analysis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    for i, matrix in enumerate(matrices):
        matrix_data = speedup_df[speedup_df['Matrix_Size'] == matrix]
        avg_efficiency = matrix_data.groupby('Threads')['Efficiency'].mean()
        
        ax.plot(avg_efficiency.index, avg_efficiency.values, 'o-',
               color=colors[i], linewidth=2, markersize=6,
               label=f'Matrix {matrix}x{matrix}')
    
    ax.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Ideal Efficiency (100%)')
    ax.set_title('Parallel Efficiency Analysis', fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Efficiency (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig('efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return speedup_df

def plot_comparison_analysis(serial_df, openmp_df):
    """Plot serial vs parallel comparison"""
    if serial_df is None or openmp_df is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Serial vs Parallel Performance Comparison', fontsize=16, fontweight='bold')
    
    matrices = sorted(serial_df['Matrix_Size'].unique())[:4]
    
    for i, matrix in enumerate(matrices):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Serial data
        serial_data = serial_df[serial_df['Matrix_Size'] == matrix]
        kernels = sorted(serial_data['Kernel_Size'].unique())
        
        x_pos = np.arange(len(kernels))
        width = 0.15
        
        serial_times = [serial_data[serial_data['Kernel_Size'] == k]['Execution_Time_us'].iloc[0] 
                       for k in kernels]
        
        ax.bar(x_pos - width*1.5, serial_times, width, label='Serial', alpha=0.8)
        
        # OpenMP data for different thread counts
        thread_counts = [1, 2, 4, 8]
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for j, threads in enumerate(thread_counts):
            openmp_subset = openmp_df[(openmp_df['Matrix_Size'] == matrix) & 
                                     (openmp_df['Threads'] == threads)]
            
            if len(openmp_subset) > 0:
                openmp_times = []
                for k in kernels:
                    kernel_data = openmp_subset[openmp_subset['Kernel_Size'] == k]['Execution_Time_us']
                    if len(kernel_data) > 0:
                        openmp_times.append(kernel_data.iloc[0])
                    else:
                        openmp_times.append(0)
                
                ax.bar(x_pos - width*0.5 + j*width, openmp_times, width, 
                      label=f'{threads} threads', alpha=0.8, color=colors[j])
        
        ax.set_title(f'Matrix {matrix}x{matrix}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Kernel Size')
        ax.set_ylabel('Execution Time (μs)')
        ax.set_yscale('log')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{k}x{k}' for k in kernels])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_detailed_report(serial_df, openmp_df, speedup_df):
    """Generate comprehensive analysis report"""
    
    report = f"""
# High Performance Computing HW2 - Convolution Performance Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a comprehensive analysis of 2D convolution implementation performance, 
comparing serial and OpenMP parallel versions across different matrix sizes and kernel sizes.

## Experimental Setup

### Platform Specifications
- **Operating System**: Windows
- **Compiler**: GCC with OpenMP support
- **Test Configuration**:
  - Matrix Sizes: {sorted(serial_df['Matrix_Size'].unique())}
  - Kernel Sizes: {sorted(serial_df['Kernel_Size'].unique())}
  - Thread Counts: {sorted(openmp_df['Threads'].unique())}

## Performance Analysis Results

### Serial Performance Analysis
"""

    # Serial performance summary
    report += "\n#### Serial Execution Times (microseconds)\n\n"
    serial_pivot = serial_df.pivot(index='Matrix_Size', columns='Kernel_Size', values='Execution_Time_us')
    report += serial_pivot.to_string() + "\n\n"
    
    # Complexity analysis
    report += "#### Computational Complexity Analysis\n\n"
    for kernel in sorted(serial_df['Kernel_Size'].unique()):
        data = serial_df[serial_df['Kernel_Size'] == kernel].sort_values('Matrix_Size')
        if len(data) > 1:
            sizes = data['Matrix_Size'].values
            times = data['Execution_Time_us'].values
            # Calculate complexity exponent
            log_sizes = np.log(sizes)
            log_times = np.log(times)
            complexity = np.polyfit(log_sizes, log_times, 1)[0]
            report += f"- Kernel {kernel}x{kernel}: O(n^{complexity:.2f})\n"
    
    # OpenMP performance analysis
    report += "\n### Parallel Performance Analysis\n\n"
    
    if speedup_df is not None and not speedup_df.empty:
        report += "#### Speedup Results\n\n"
        for matrix in sorted(speedup_df['Matrix_Size'].unique()):
            report += f"**Matrix {matrix}x{matrix}:**\n\n"
            matrix_data = speedup_df[speedup_df['Matrix_Size'] == matrix]
            
            for kernel in sorted(matrix_data['Kernel_Size'].unique()):
                kernel_data = matrix_data[matrix_data['Kernel_Size'] == kernel].sort_values('Threads')
                report += f"- Kernel {kernel}x{kernel}:\n"
                for _, row in kernel_data.iterrows():
                    report += f"  - {int(row['Threads'])} threads: {row['Speedup']:.2f}x speedup ({row['Efficiency']:.1f}% efficiency)\n"
                report += "\n"
        
        # Best performance summary
        report += "#### Performance Highlights\n\n"
        best_speedup = speedup_df.loc[speedup_df['Speedup'].idxmax()]
        best_efficiency = speedup_df.loc[speedup_df['Efficiency'].idxmax()]
        
        report += f"- **Best Speedup**: {best_speedup['Speedup']:.2f}x with {int(best_speedup['Threads'])} threads "
        report += f"(Matrix {int(best_speedup['Matrix_Size'])}x{int(best_speedup['Matrix_Size'])}, "
        report += f"Kernel {int(best_speedup['Kernel_Size'])}x{int(best_speedup['Kernel_Size'])})\n"
        
        report += f"- **Best Efficiency**: {best_efficiency['Efficiency']:.1f}% with {int(best_efficiency['Threads'])} threads "
        report += f"(Matrix {int(best_efficiency['Matrix_Size'])}x{int(best_efficiency['Matrix_Size'])}, "
        report += f"Kernel {int(best_efficiency['Kernel_Size'])}x{int(best_efficiency['Kernel_Size'])})\n\n"
    
    # Scalability analysis
    report += "### Scalability Analysis\n\n"
    report += "#### Strong Scaling\n\n"
    report += "Strong scaling measures how execution time decreases as the number of processors increases "
    report += "for a fixed problem size. Our analysis shows:\n\n"
    
    if speedup_df is not None:
        # Calculate average efficiency for each thread count
        avg_efficiency = speedup_df.groupby('Threads')['Efficiency'].mean()
        for threads, eff in avg_efficiency.items():
            report += f"- {int(threads)} threads: {eff:.1f}% average efficiency\n"
    
    report += "\n#### Optimization Techniques Applied\n\n"
    report += "1. **OpenMP Parallelization**: Used `#pragma omp parallel for collapse(2)` to parallelize the outer two loops\n"
    report += "2. **Static Scheduling**: Applied static scheduling for better load balancing\n"
    report += "3. **Zero Padding**: Implemented efficient boundary condition handling\n"
    report += "4. **Memory Access Optimization**: Optimized array indexing for better cache locality\n\n"
    
    report += "## Conclusions and Recommendations\n\n"
    
    if speedup_df is not None:
        max_threads = speedup_df['Threads'].max()
        avg_speedup_max = speedup_df[speedup_df['Threads'] == max_threads]['Speedup'].mean()
        
        report += f"1. **Parallel Efficiency**: The OpenMP implementation shows good scalability up to {int(max_threads)} threads, "
        report += f"achieving an average speedup of {avg_speedup_max:.2f}x.\n\n"
        
        report += "2. **Problem Size Impact**: Larger matrices benefit more from parallelization due to better "
        report += "computation-to-overhead ratio.\n\n"
        
        report += "3. **Kernel Size Effect**: Larger kernels show better parallel efficiency due to increased "
        report += "computational intensity per output element.\n\n"
    
    report += "4. **Optimization Opportunities**: Further improvements could include SIMD vectorization, "
    report += "cache blocking, and GPU acceleration for very large problem sizes.\n\n"
    
    # Save report
    with open('performance_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Detailed report saved as: performance_analysis_report.md")
    return report

def analyze_and_plot(): 
    """Main analysis function""" 
    serial_df, openmp_df = load_latest_data() 
 
    if serial_df is None or openmp_df is None: 
        print("Cannot load data files. Please run the test batch file first.")
        return 
 
    # Display basic statistics 
    print("="*60) 
    print("PERFORMANCE ANALYSIS REPORT") 
    print("="*60) 
 
    print("\nSerial Performance Summary:") 
    print(serial_df.groupby(['Matrix_Size', 'Kernel_Size'])['Execution_Time_us'].describe()) 
 
    print("\nOpenMP Performance Summary:") 
    print(openmp_df.groupby(['Threads', 'Matrix_Size'])['Execution_Time_us'].describe()) 
 
    # Generate all plots
    print("\nGenerating performance plots...")
    
    plot_serial_performance(serial_df)
    speedup_df = plot_speedup_analysis(openmp_df)
    plot_comparison_analysis(serial_df, openmp_df)
    
    # Generate detailed report
    print("\nGenerating detailed analysis report...")
    generate_detailed_report(serial_df, openmp_df, speedup_df)
    
    print("\nAnalysis complete! Generated files:")
    print("- serial_performance_analysis.png")
    print("- speedup_analysis.png") 
    print("- efficiency_analysis.png")
    print("- comparison_analysis.png")
    print("- performance_analysis_report.md")
 
if __name__ == "__main__": 
    analyze_and_plot()