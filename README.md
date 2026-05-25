# EXP-6---Matrix-multiplication-using-cuBLAS-in-CUDA-C-
## NAME :SANTHIYA B
## REG NO : 212224230247
# Objective
To implement matrix multiplication on the GPU using the cuBLAS library in CUDA C, and analyze the performance improvement over CPU-based matrix multiplication by leveraging GPU acceleration.

# AIM:
To utilize the cuBLAS library for performing matrix multiplication on NVIDIA GPUs, enhancing the performance of matrix operations by parallelizing computations and utilizing efficient GPU memory access.

Code Overview
In this experiment, you will work with the provided CUDA C code that performs matrix multiplication using the cuBLAS library. The code initializes two matrices (A and B) on the host, transfers them to the GPU device, and uses cuBLAS functions to compute the matrix product (C). The resulting matrix C is then transferred back to the host for verification and output.

# EQUIPMENTS REQUIRED:
Hardware:
PC with NVIDIA GPU
Google Colab with NVCC compiler
Software:
CUDA Toolkit (with cuBLAS library)
NVCC (NVIDIA CUDA Compiler)
Sample datasets for matrix multiplication (e.g., random matrices)

# PROCEDURE:
Tasks:
Initialize Host Memory:

Allocate memory for matrices A, B, and C on the host (CPU). Use random values for matrices A and B.
Allocate Device Memory:

Allocate corresponding memory on the GPU device for matrices A, B, and C using cudaMalloc().
Transfer the host matrices A and B to the GPU device using cudaMemcpy().
Matrix Multiplication using cuBLAS:

Initialize the cuBLAS library using cublasCreate().
Use the cublasSgemm() function to perform single-precision matrix multiplication on the GPU. This function computes the matrix product C = alpha * A * B + beta * C.
Retrieve and Print Results:

Copy the resulting matrix C from the device back to the host memory using cudaMemcpy().
Print the matrices A, B, and C to verify the correctness of the multiplication.
Clean Up Resources:

Free the allocated host and device memory using free() and cudaFree().
Shutdown the cuBLAS library using cublasDestroy().

Performance Analysis:
Measure the execution time of matrix multiplication using the cuBLAS library with different matrix sizes (e.g., 256x256, 512x512, 1024x1024).
Experiment with varying block sizes (e.g., 16, 32, 64 threads per block) and analyze their effect on execution time.
Compare the performance of the GPU-based matrix multiplication using cuBLAS with a standard CPU-based matrix multiplication implementation.
# PROGRAM:
```cu
cuda_code = r"""
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <time.h>

// Define matrix indexing for column-major order
#define index(i,j,ld) (((j)*(ld))+(i))

// Initialize matrices
void initializeMatrix(float *matrix, int size) {

    for (int i = 0; i < size; i++) {

        for (int j = 0; j < size; j++) {

            matrix[index(i, j, size)] =
                (float)(i + j) / size;
        }
    }
}

// CPU matrix multiplication
void cpuMatrixMultiplication(float *A,
                             float *B,
                             float *C,
                             int n) {

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < n; j++) {

            C[index(i, j, n)] = 0.0f;

            for (int k = 0; k < n; k++) {

                C[index(i, j, n)] +=
                    A[index(i, k, n)] *
                    B[index(k, j, n)];
            }
        }
    }
}

int main() {

    int sizes[] = {256, 512, 1024};

    int numSizes = 3;

    for (int s = 0; s < numSizes; s++) {

        int size = sizes[s];

        printf("\nRunning matrix multiplication for size: %d x %d\n",
               size, size);

        size_t bytes = size * size * sizeof(float);

        // Allocate host memory
        float *A =
            (float*)malloc(bytes);

        float *B =
            (float*)malloc(bytes);

        float *C_cpu =
            (float*)malloc(bytes);

        float *C_gpu =
            (float*)malloc(bytes);

        // Initialize matrices
        initializeMatrix(A, size);
        initializeMatrix(B, size);

        // ---------------- CPU Timing ----------------
        clock_t start_cpu = clock();

        cpuMatrixMultiplication(A,
                                B,
                                C_cpu,
                                size);

        clock_t end_cpu = clock();

        double time_cpu =
            ((double)(end_cpu - start_cpu))
            / CLOCKS_PER_SEC;

        printf("CPU Matrix Multiplication Time: %f seconds\n",
               time_cpu);

        // ---------------- Device Memory ----------------
        float *d_A, *d_B, *d_C;

        cudaMalloc((void**)&d_A, bytes);
        cudaMalloc((void**)&d_B, bytes);
        cudaMalloc((void**)&d_C, bytes);

        // Copy data to GPU
        cudaMemcpy(d_A,
                   A,
                   bytes,
                   cudaMemcpyHostToDevice);

        cudaMemcpy(d_B,
                   B,
                   bytes,
                   cudaMemcpyHostToDevice);

        // ---------------- cuBLAS Handle ----------------
        cublasHandle_t handle;

        cublasCreate(&handle);

        float alpha = 1.0f;
        float beta  = 0.0f;

        // ---------------- GPU Timing ----------------
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        // Matrix Multiplication using cuBLAS
        // Column-major order:
        // C = A × B

        cublasSgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    size,
                    size,
                    size,
                    &alpha,
                    d_B,
                    size,
                    d_A,
                    size,
                    &beta,
                    d_C,
                    size);

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float time_gpu;

        cudaEventElapsedTime(&time_gpu,
                             start,
                             stop);

        printf("GPU Matrix Multiplication Time (cuBLAS): %f milliseconds\n",
               time_gpu);

        // Copy result back to host
        cudaMemcpy(C_gpu,
                   d_C,
                   bytes,
                   cudaMemcpyDeviceToHost);

        // ---------------- Verification ----------------
        int errors = 0;

        float max_relative_error = 1e-4;

        for (int i = 0; i < size * size; i++) {

            float cpu_val = C_cpu[i];

            float gpu_val = C_gpu[i];

            float denominator =
                fmax(fabs(cpu_val),
                     fabs(gpu_val));

            if (denominator < 1e-6)
                denominator = 1.0f;

            float relative_error =
                fabs(cpu_val - gpu_val)
                / denominator;

            if (relative_error >
                max_relative_error) {

                errors++;
            }
        }

        if (errors == 0) {

            printf("Results verified successfully for size %d x %d\n",
                   size,
                   size);
        }
        else {

            printf("Discrepancies found in the results for size %d x %d\n",
                   size,
                   size);

            printf("Total Errors: %d\n",
                   errors);
        }

        // ---------------- Cleanup ----------------
        cublasDestroy(handle);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        free(A);
        free(B);
        free(C_cpu);
        free(C_gpu);
    }

    return 0;
}
"""

# Save CUDA code to file
with open("matrix_multiplication.cu", "w") as file:
    file.write(cuda_code)

print("CUDA code saved successfully.")
```

# OUTPUT:

<img width="604" height="267" alt="image" src="https://github.com/user-attachments/assets/566354da-aea5-452d-a2b2-d8c02cd5d1cc" />


# RESULT:

Thus, the matrix multiplication has been successfully implemented using the cuBLAS library in CUDA C, demonstrating the enhanced performance of GPU-based computation over CPU-based approaches.
