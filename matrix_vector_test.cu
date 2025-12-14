#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

__global__ void matrix_vector(float* d_a, float* d_b, float* d_c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float addition = 0.0f;
        for (int j = 0; j < n; j++) {
            addition += (d_b[i * n + j] + d_c[j]);
        }
        d_a[i] = addition;
    }
}

float matrix_vector_host(float* a, float* b, float* c, int n) {
    int size = n * n * sizeof(float);
    int size_vector = n * sizeof(float);
    float* d_a, * d_b, * d_c;

    cudaMalloc((void**)&d_a, size_vector);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size_vector);

    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size_vector, cudaMemcpyHostToDevice);

    int threads_block = 256;
    int threads_grid = (n + threads_block - 1) / threads_block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrix_vector<<<threads_grid, threads_block>>>(d_a, d_b, d_c, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(a, d_a, size_vector, cudaMemcpyDeviceToHost);

    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

void test_size(int n, int num_iterations) {
    printf("Testing with matrix size: %d x %d\n", n, n);

    float* a = (float*)malloc(n * sizeof(float));
    float* b = (float*)malloc(n * n * sizeof(float));
    float* c = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n * n; i++) {
        b[i] = (float)(i % 10);
    }
    for (int i = 0; i < n; i++) {
        c[i] = 1.0f;
    }

    float total_time = 0.0f;

    for (int iter = 0; iter < num_iterations; iter++) {
        for (int i = 0; i < n; i++) {
            a[i] = 0.0f;
        }

        float time = matrix_vector_host(a, b, c, n);
        total_time += time;
    }

    float avg_time = total_time / num_iterations;
    float gflops = (2.0f * n * n) / (avg_time * 1e6);
    float bandwidth = ((n * n + 2 * n) * sizeof(float)) / (avg_time * 1e6);

    printf("\nMatrix-Vector Operation:\n");
    printf("  Average Time: %.4f ms\n", avg_time);
    printf("  Performance: %.2f GFLOPS\n", gflops);
    printf("  Bandwidth: %.2f GB/s\n", bandwidth);
    printf("\n");

    free(a);
    free(b);
    free(c);
}

int main() {
    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_iterations = 10;

    printf("CUDA Matrix-Vector Performance Testing\n");
    printf("Number of iterations per test: %d\n", num_iterations);
    printf("========================================\n\n");

    for (int i = 0; i < num_sizes; i++) {
        test_size(sizes[i], num_iterations);
    }

    return 0;
}
