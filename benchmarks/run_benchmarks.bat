@echo off
REM Build CPU code using g++
g++ -O3 -fopenmp src\cpu\laplace_cpu.cpp -o laplace_cpu.exe

REM Build GPU code using nvcc (make sure nvcc is in your PATH)
nvcc -O3 src\gpu\laplace_cuda.cu -o laplace_cuda.exe

REM Clear previous performance logs if they exist
if exist benchmarks\performance.csv del benchmarks\performance.csv
echo Version,Time_sec > benchmarks\performance.csv

REM Run CPU simulation
laplace_cpu.exe

REM Run GPU simulation
laplace_cuda.exe

echo Benchmarking complete. See benchmarks\performance.csv for results.
pause
