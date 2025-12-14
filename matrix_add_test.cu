#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

__global__ void matrix_add_element(float* d_c, float* d_a, float* d_b, int n);
__global__ void matrix_add_row(float* d_c, float* d_a, float* d_b, int n);
__global__ void matrix_add_column(float* d_c, float* d_a, float* d_b, int n);

float matrix_addition(float* c, float* a, float* b, int n, char solution) {
    int size = n * n * sizeof(float);
    float* d_a, * d_b, * d_c;

    cudaMalloc((void**) &d_a, size);
    cudaMalloc((void**) &d_b, size);
    cudaMalloc((void**) &d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 dim_block;
    dim3 dim_grid;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    if (solution == 'B') {
        dim_block = dim3(16, 16, 1);
        dim_grid = dim3((n + dim_block.x - 1) / dim_block.x, (n + dim_block.y - 1) / dim_block.y, 1);
        matrix_add_element<<<dim_grid, dim_block>>>(d_c, d_a, d_b, n);
    }
    else if (solution == 'C') {
        dim_block = dim3(256, 1, 1);
        dim_grid = dim3((n + dim_block.x - 1) / dim_block.x, 1, 1);
        matrix_add_row<<<dim_grid, dim_block>>>(d_c, d_a, d_b, n);
    }
    else if (solution == 'D') {
        dim_block = dim3(256, 1, 1);
        dim_grid = dim3((n + dim_block.x - 1) / dim_block.x, 1, 1);
        matrix_add_column<<<dim_grid, dim_block>>>(d_c, d_a, d_b, n);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

__global__ void matrix_add_element(float* d_c, float* d_a, float* d_b, int n) {
    int columns = blockIdx.x * blockDim.x + threadIdx.x;
    int rows = blockIdx.y * blockDim.y + threadIdx.y;

    if (columns < n && rows < n) {
        int index = rows * n + columns;
        d_c[index] = d_a[index] + d_b[index];
    }
}

__global__ void matrix_add_row(float* d_c, float* d_a, float* d_b, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        for (int i = 0; i < n; i++) {
            int index = row * n + i;
            d_c[index] = d_a[index] + d_b[index];
        }
    }
}

__global__ void matrix_add_column(float* d_c, float* d_a, float* d_b, int n) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (column < n) {
        for (int i = 0; i < n; i++) {
            int index = i * n + column;
            d_c[index] = d_a[index] + d_b[index];
        }
    }
}

void test_size(int n, int num_iterations) {
    printf("Testing with matrix size: %d x %d\n", n, n);

    float* a = (float*)malloc(n * n * sizeof(float));
    float* b = (float*)malloc(n * n * sizeof(float));
    float* c = (float*)malloc(n * n * sizeof(float));

    for (int i = 0; i < n * n; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    char solutions[] = {'B', 'C', 'D'};
    const char* names[] = {"Element-wise (1.B)", "Row-wise (1.C)", "Column-wise (1.D)"};

    for (int s = 0; s < 3; s++) {
        float total_time = 0.0f;
        bool passed = true;

        for (int iter = 0; iter < num_iterations; iter++) {
            for (int i = 0; i < n * n; i++) {
                c[i] = 0.0f;
            }

            float time = matrix_addition(c, a, b, n, solutions[s]);
            total_time += time;

            if (iter == 0) {
                for (int i = 0; i < n * n; i++) {
                    if (fabs(c[i] - 3.0f) > 0.001f) {
                        passed = false;
                        break;
                    }
                }
            }
        }

        float avg_time = total_time / num_iterations;
        float gflops = (2.0f * n * n) / (avg_time * 1e6);
        float bandwidth = (3.0f * n * n * sizeof(float)) / (avg_time * 1e6);

        printf("\n%s:\n", names[s]);
        printf("  Status: %s\n", passed ? "PASSED" : "FAILED");
        printf("  Average Time: %.4f ms\n", avg_time);
        printf("  Performance: %.2f GFLOPS\n", gflops);
        printf("  Bandwidth: %.2f GB/s\n", bandwidth);
    }

    free(a);
    free(b);
    free(c);
}

int main() {
    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_iterations = 10;

    printf("CUDA Matrix Addition Performance Testing\n");
    printf("Number of iterations per test: %d\n", num_iterations);

    for (int i = 0; i < num_sizes; i++) {
        test_size(sizes[i], num_iterations);
    }

    return 0;
}
