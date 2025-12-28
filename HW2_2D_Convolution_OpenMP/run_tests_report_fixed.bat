@echo off
REM filepath: c:\Users\OliverLin\OneDrive\Desktop\HP_HW2\run_tests_report_fixed.bat

echo ===============================================
echo High Performance Computing HW2 - Report Generation Test (Fixed)
echo ===============================================

REM Create results directory
if not exist "results" mkdir results

REM Get current date and time for log file
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "YY=%dt:~2,2%" & set "YYYY=%dt:~0,4%" & set "MM=%dt:~4,2%" & set "DD=%dt:~6,2%"
set "HH=%dt:~8,2%" & set "Min=%dt:~10,2%" & set "Sec=%dt:~12,2%"
set "datestamp=%YYYY%-%MM%-%DD%_%HH%-%Min%-%Sec%"

echo Test started at: %YYYY%-%MM%-%DD% %HH%:%Min%:%Sec%
echo Generating report files with timestamp: %datestamp%

echo Compiling programs...
gcc conv.c conv_template.c -o conv
if errorlevel 1 (
    echo Error: Failed to compile serial version
    pause
    exit /b 1
)

gcc -fopenmp conv_openmp.c conv_openmp_template.c -o conv_openmp
if errorlevel 1 (
    echo Error: Failed to compile OpenMP version
    pause
    exit /b 1
)
echo Compilation successful!
echo.

REM Test configurations
set "matrices=256 512 1024 2048 4096"
set "kernels=3 5 7 9"
set "threads=1 2 4 8 16 32"

REM Initialize CSV files with headers
echo Matrix_Size,Kernel_Size,Execution_Time_us,Status > results\serial_performance_%datestamp%.csv
echo Threads,Matrix_Size,Kernel_Size,Execution_Time_us,Status > results\openmp_performance_%datestamp%.csv

REM Initialize report file
echo High Performance Computing HW2 - Convolution Performance Report > results\performance_report_%datestamp%.txt
echo Generated on: %YYYY%-%MM%-%DD% %HH%:%Min%:%Sec% >> results\performance_report_%datestamp%.txt
echo ================================================================ >> results\performance_report_%datestamp%.txt
echo. >> results\performance_report_%datestamp%.txt

echo System Information: >> results\performance_report_%datestamp%.txt
echo - OS: Windows >> results\performance_report_%datestamp%.txt
echo - Compiler: GCC >> results\performance_report_%datestamp%.txt
echo - OpenMP: Enabled >> results\performance_report_%datestamp%.txt
echo - Test Matrices: 256x256, 512x512, 1024x1024, 2048x2048, 4096x4096 >> results\performance_report_%datestamp%.txt
echo - Kernel Sizes: 3x3, 5x5, 7x7, 9x9 >> results\performance_report_%datestamp%.txt
echo - Thread Counts: 1, 2, 4, 8, 16, 32 >> results\performance_report_%datestamp%.txt
echo. >> results\performance_report_%datestamp%.txt

echo ===============================================
echo Serial Version Performance Test
echo ===============================================

echo SERIAL PERFORMANCE RESULTS >> results\performance_report_%datestamp%.txt
echo ========================== >> results\performance_report_%datestamp%.txt

for %%m in (%matrices%) do (
    for %%k in (%kernels%) do (
        echo Testing Serial - Matrix: %%m x %%m, Kernel: %%k x %%k
        echo Testing Serial - Matrix: %%m x %%m, Kernel: %%k x %%k >> results\performance_report_%datestamp%.txt
        
        REM Run test and capture output
        conv.exe testing_data\mat-%%m.txt testing_data\ker-%%k.txt testing_data\ans-%%m-%%k.txt > temp_serial_%%m_%%k.txt 2>&1
        
        REM Check if test was successful and extract time
        findstr /C:"Correct!" temp_serial_%%m_%%k.txt > nul
        if not errorlevel 1 (
            REM Extract execution time - look for "Elapsed time: X.XX us"
            for /f "tokens=3" %%t in ('findstr /C:"Elapsed time:" temp_serial_%%m_%%k.txt') do (
                echo %%m,%%k,%%t,PASS >> results\serial_performance_%datestamp%.csv
                echo   Result: %%t - PASS >> results\performance_report_%datestamp%.txt
            )
        ) else (
            echo %%m,%%k,0,FAIL >> results\serial_performance_%datestamp%.csv
            echo   Result: FAILED >> results\performance_report_%datestamp%.txt
        )
        
        del temp_serial_%%m_%%k.txt
    )
)

echo. >> results\performance_report_%datestamp%.txt

echo ===============================================
echo OpenMP Version Performance Test
echo ===============================================

echo OPENMP PERFORMANCE RESULTS >> results\performance_report_%datestamp%.txt
echo =========================== >> results\performance_report_%datestamp%.txt

for %%t in (%threads%) do (
    echo. >> results\performance_report_%datestamp%.txt
    echo Testing OpenMP with %%t threads >> results\performance_report_%datestamp%.txt
    echo Testing OpenMP with %%t threads
    set OMP_NUM_THREADS=%%t
    
    for %%m in (%matrices%) do (
        for %%k in (%kernels%) do (
            echo   [%%t threads] Matrix: %%m x %%m, Kernel: %%k x %%k
            echo   [%%t threads] Matrix: %%m x %%m, Kernel: %%k x %%k >> results\performance_report_%datestamp%.txt
            
            REM Run OpenMP test
            conv_openmp.exe testing_data\mat-%%m.txt testing_data\ker-%%k.txt testing_data\ans-%%m-%%k.txt > temp_openmp_%%t_%%m_%%k.txt 2>&1
            
            REM Check if test was successful and extract time
            findstr /C:"Correct!" temp_openmp_%%t_%%m_%%k.txt > nul
            if not errorlevel 1 (
                for /f "tokens=3" %%o in ('findstr /C:"Elapsed time:" temp_openmp_%%t_%%m_%%k.txt') do (
                    echo %%t,%%m,%%k,%%o,PASS >> results\openmp_performance_%datestamp%.csv
                    echo     Result: %%o - PASS >> results\performance_report_%datestamp%.txt
                )
            ) else (
                echo %%t,%%m,%%k,0,FAIL >> results\openmp_performance_%datestamp%.csv
                echo     Result: FAILED >> results\performance_report_%datestamp%.txt
            )
            
            del temp_openmp_%%t_%%m_%%k.txt
        )
    )
)

echo ===============================================
echo Generating Enhanced Python Analysis Script
echo ===============================================

REM Create comprehensive Python analysis script
echo # High Performance Computing HW2 - Enhanced Analysis Script > results\analyze_results_enhanced.py
echo # Generated on %YYYY%-%MM%-%DD% %HH%:%Min%:%Sec% >> results\analyze_results_enhanced.py
echo. >> results\analyze_results_enhanced.py
echo import pandas as pd >> results\analyze_results_enhanced.py
echo import matplotlib.pyplot as plt >> results\analyze_results_enhanced.py
echo import numpy as np >> results\analyze_results_enhanced.py
echo import seaborn as sns >> results\analyze_results_enhanced.py
echo import os >> results\analyze_results_enhanced.py
echo. >> results\analyze_results_enhanced.py
echo def load_latest_data(): >> results\analyze_results_enhanced.py
echo     """Load the most recent CSV files""" >> results\analyze_results_enhanced.py
echo     try: >> results\analyze_results_enhanced.py
echo         # Find CSV files >> results\analyze_results_enhanced.py
echo         csv_files = [f for f in os.listdir('.') if f.endswith('.csv')] >> results\analyze_results_enhanced.py
echo         serial_files = [f for f in csv_files if 'serial_performance' in f] >> results\analyze_results_enhanced.py
echo         openmp_files = [f for f in csv_files if 'openmp_performance' in f] >> results\analyze_results_enhanced.py
echo. >> results\analyze_results_enhanced.py
echo         if not serial_files or not openmp_files: >> results\analyze_results_enhanced.py
echo             print("CSV files not found!") >> results\analyze_results_enhanced.py
echo             return None, None >> results\analyze_results_enhanced.py
echo. >> results\analyze_results_enhanced.py
echo         # Load latest files >> results\analyze_results_enhanced.py
echo         serial_df = pd.read_csv(sorted(serial_files)[-1]) >> results\analyze_results_enhanced.py
echo         openmp_df = pd.read_csv(sorted(openmp_files)[-1]) >> results\analyze_results_enhanced.py
echo. >> results\analyze_results_enhanced.py
echo         # Clean data >> results\analyze_results_enhanced.py
echo         serial_df['Execution_Time_us'] = pd.to_numeric(serial_df['Execution_Time_us'], errors='coerce') >> results\analyze_results_enhanced.py
echo         openmp_df['Execution_Time_us'] = pd.to_numeric(openmp_df['Execution_Time_us'], errors='coerce') >> results\analyze_results_enhanced.py
echo. >> results\analyze_results_enhanced.py
echo         print(f"Loaded: {sorted(serial_files)[-1]}, {sorted(openmp_files)[-1]}") >> results\analyze_results_enhanced.py
echo         return serial_df, openmp_df >> results\analyze_results_enhanced.py
echo. >> results\analyze_results_enhanced.py
echo     except Exception as e: >> results\analyze_results_enhanced.py
echo         print(f"Error: {e}") >> results\analyze_results_enhanced.py
echo         return None, None >> results\analyze_results_enhanced.py
echo. >> results\analyze_results_enhanced.py
echo def analyze_and_plot(): >> results\analyze_results_enhanced.py
echo     """Main analysis function""" >> results\analyze_results_enhanced.py
echo     serial_df, openmp_df = load_latest_data() >> results\analyze_results_enhanced.py
echo. >> results\analyze_results_enhanced.py
echo     if serial_df is None or openmp_df is None: >> results\analyze_results_enhanced.py
echo         return >> results\analyze_results_enhanced.py
echo. >> results\analyze_results_enhanced.py
echo     # Display basic statistics >> results\analyze_results_enhanced.py
echo     print("="*60) >> results\analyze_results_enhanced.py
echo     print("PERFORMANCE ANALYSIS REPORT") >> results\analyze_results_enhanced.py
echo     print("="*60) >> results\analyze_results_enhanced.py
echo. >> results\analyze_results_enhanced.py
echo     print("\nSerial Performance Summary:") >> results\analyze_results_enhanced.py
echo     print(serial_df.groupby(['Matrix_Size', 'Kernel_Size'])['Execution_Time_us'].describe()) >> results\analyze_results_enhanced.py
echo. >> results\analyze_results_enhanced.py
echo     print("\nOpenMP Performance Summary:") >> results\analyze_results_enhanced.py
echo     print(openmp_df.groupby(['Threads', 'Matrix_Size'])['Execution_Time_us'].describe()) >> results\analyze_results_enhanced.py
echo. >> results\analyze_results_enhanced.py
echo     # Calculate speedup >> results\analyze_results_enhanced.py
echo     print("\nSpeedup Analysis:") >> results\analyze_results_enhanced.py
echo     for matrix in sorted(openmp_df['Matrix_Size'].unique()): >> results\analyze_results_enhanced.py
echo         for kernel in sorted(openmp_df['Kernel_Size'].unique()): >> results\analyze_results_enhanced.py
echo             subset = openmp_df[(openmp_df['Matrix_Size'] == matrix) ^& (openmp_df['Kernel_Size'] == kernel)] >> results\analyze_results_enhanced.py
echo             baseline = subset[subset['Threads'] == 1]['Execution_Time_us'] >> results\analyze_results_enhanced.py
echo             if len(baseline) ^> 0: >> results\analyze_results_enhanced.py
echo                 baseline_time = baseline.iloc[0] >> results\analyze_results_enhanced.py
echo                 print(f"\nMatrix {matrix}x{matrix}, Kernel {kernel}x{kernel}:") >> results\analyze_results_enhanced.py
echo                 for threads in sorted(subset['Threads'].unique()): >> results\analyze_results_enhanced.py
echo                     thread_time = subset[subset['Threads'] == threads]['Execution_Time_us'] >> results\analyze_results_enhanced.py
echo                     if len(thread_time) ^> 0: >> results\analyze_results_enhanced.py
echo                         speedup = baseline_time / thread_time.iloc[0] >> results\analyze_results_enhanced.py
echo                         print(f"  {threads} threads: {speedup:.2f}x speedup") >> results\analyze_results_enhanced.py
echo. >> results\analyze_results_enhanced.py
echo if __name__ == "__main__": >> results\analyze_results_enhanced.py
echo     analyze_and_plot() >> results\analyze_results_enhanced.py

echo ===============================================
echo Quick Data Summary
echo ===============================================

echo Generating quick summary...
echo QUICK PERFORMANCE SUMMARY > results\quick_summary_%datestamp%.txt
echo ========================= >> results\quick_summary_%datestamp%.txt
echo Generated: %YYYY%-%MM%-%DD% %HH%:%Min%:%Sec% >> results\quick_summary_%datestamp%.txt
echo. >> results\quick_summary_%datestamp%.txt

REM Count successful tests
for /f %%c in ('findstr /C:",PASS" results\serial_performance_%datestamp%.csv ^| find /C ","') do (
    echo Serial tests passed: %%c >> results\quick_summary_%datestamp%.txt
)

for /f %%c in ('findstr /C:",PASS" results\openmp_performance_%datestamp%.csv ^| find /C ","') do (
    echo OpenMP tests passed: %%c >> results\quick_summary_%datestamp%.txt
)

echo ===============================================
echo Test Complete!
echo ===============================================

echo. >> results\performance_report_%datestamp%.txt
echo TEST COMPLETION SUMMARY >> results\performance_report_%datestamp%.txt
echo ======================= >> results\performance_report_%datestamp%.txt
echo Test completed: %YYYY%-%MM%-%DD% %HH%:%Min%:%Sec% >> results\performance_report_%datestamp%.txt

echo ===============================================
echo Files Generated:
echo ===============================================
echo - performance_report_%datestamp%.txt
echo - serial_performance_%datestamp%.csv  
echo - openmp_performance_%datestamp%.csv
echo - analyze_results_enhanced.py
echo - quick_summary_%datestamp%.txt
echo.
echo To analyze results:
echo   cd results
echo   python analyze_results_enhanced.py
echo.

pause