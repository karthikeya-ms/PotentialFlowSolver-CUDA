// File: src/cpu/laplace_cpu.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <fstream>

int main() {
    // Grid and simulation parameters
    const int Nx = 1024, Ny = 1024;
    const double tol = 1e-6;
    const int maxIter = 10000;
    const double U = 1.0; // uniform flow speed

    // Allocate and initialize grid arrays (1D vector for 2D grid)
    std::vector<double> phi(Nx * Ny, 0.0), phi_new(Nx * Ny, 0.0);

    // Set boundary conditions for uniform flow: phi = U * x
    for (int i = 0; i < Nx; i++) {
        double x = double(i) / (Nx - 1);
        phi[i] = U * x;                             // Bottom row (j = 0)
        phi[(Ny - 1) * Nx + i] = U * x;              // Top row (j = Ny-1)
    }
    for (int j = 0; j < Ny; j++) {
        phi[j * Nx] = U * 0.0;                       // Left column (i = 0)
        phi[j * Nx + (Nx - 1)] = U * 1.0;             // Right column (i = Nx-1)
    }

    // Define a cylinder (obstacle) in the center of the domain
    const int cx = Nx / 2, cy = Ny / 2;
    const int r = Nx / 8;  // radius of cylinder
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            int dx = i - cx, dy = j - cy;
            if (dx * dx + dy * dy <= r * r) {
                phi[j * Nx + i] = 0.0;  // Dirichlet condition inside the cylinder
            }
        }
    }
    // Copy initial condition
    phi_new = phi;

    // Start timing the solver
    auto start = std::chrono::high_resolution_clock::now();

    // Jacobi iteration loop
    for (int iter = 0; iter < maxIter; iter++) {
        double maxDiff = 0.0;

        // Parallelize the update using OpenMP (collapse nested loops)
        #pragma omp parallel for reduction(max:maxDiff) collapse(2)
        for (int j = 1; j < Ny - 1; j++) {
            for (int i = 1; i < Nx - 1; i++) {
                // Skip updates for points inside the cylinder
                int dx = i - cx, dy = j - cy;
                if (dx * dx + dy * dy <= r * r) continue;

                // Update using the average of the four neighbors
                phi_new[j * Nx + i] = 0.25 * (phi[j * Nx + (i - 1)] +
                                              phi[j * Nx + (i + 1)] +
                                              phi[(j - 1) * Nx + i] +
                                              phi[(j + 1) * Nx + i]);
                double diff = fabs(phi_new[j * Nx + i] - phi[j * Nx + i]);
                if (diff > maxDiff)
                    maxDiff = diff;
            }
        }

        // Swap the pointers for the next iteration
        std::swap(phi, phi_new);

        if (maxDiff < tol) {
            std::cout << "Converged after " << iter << " iterations." << std::endl;
            break;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double execTime = std::chrono::duration<double>(end - start).count();
    std::cout << "CPU (OpenMP) Execution time: " << execTime << " seconds." << std::endl;

    // Optionally, write the final potential field to file for later visualization
    std::ofstream out("phi_cpu.dat");
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            out << phi[j * Nx + i] << " ";
        }
        out << "\n";
    }
    out.close();

    // Log performance data for visualization (append to CSV)
    std::ofstream perf("benchmarks/performance.csv", std::ios::app);
    perf << "CPU," << execTime << "\n";
    perf.close();

    return 0;
}
