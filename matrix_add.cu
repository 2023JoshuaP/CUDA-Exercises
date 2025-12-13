// %%writefile matrix_add.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 64

__global__ void matrix_add_element(float* d_c, float* d_a, float* d_b, int n);
__global__ void matrix_add_row(float* d_c, float* d_a, float* d_b, int n);
__global__ void matrix_add_column(float* d_c, float* d_a, float* d_b, int n);

/* Exercise 1.A */

void matrix_addition(float* c, float* a, float* b, int n, char solution) {
    int size = n * n * sizeof(float);
    float* d_a, * d_b, * d_c;

    cudaMalloc((void**) &d_a, size);
    cudaMalloc((void**) &d_b, size);
    cudaMalloc((void**) &d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 dim_block;
    dim3 dim_grid;

    if (solution == 'B') {
        dim_block = dim3(16, 16, 1);
        dim_grid = dim3((n + dim_block.x - 1) / dim_block.x, (n + dim_block.y - 1) / dim_block.y, 1);
        printf("Exercise 1B: Grid(%d, %d)\n", dim_grid.x, dim_grid.y);
        matrix_add_element<<<dim_grid, dim_block>>>(d_c, d_a, d_b, n);
    }
    else if (solution == 'C') {
        dim_block = dim3(256, 1, 1);
        dim_grid = ((n + dim_block.x - 1) / dim_block.x, 1, 1);
        printf("Exercise 1.C: Grid(%d)\n", dim_grid.x);
        matrix_add_row<<<dim_grid, dim_block>>>(d_c, d_a, d_b, n);
    }
    else if (solution == 'D') {
        dim_block = dim3(256, 1, 1);
        dim_grid = ((n + dim_block.x - 1) / dim_block.x, 1, 1);
        printf("Exercise 1.D: Grid(%d)\n", dim_grid.x);
        matrix_add_column<<<dim_grid, dim_block>>>(d_c, d_a, d_b, n);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

/* Exercise 1.B */

__global__ void matrix_add_element(float* d_c, float* d_a, float* d_b, int n) {
    int columns = blockIdx.x * blockDim.x + threadIdx.x;
    int rows = blñockIdx.y * blockDim.y + threadIdx.y;

    if (cols < n && rows < n) {
        int index = rows * n + columns;
        d_c[index] = d_a[index] + d_b[index];
    }
}

/* Exercise 1.C */

__global__ matrix_add_row(float* d_c, float* d_a, float* d_b, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n) {
        for (int i = 0; i < n; i++) {
            int index = row * n + i;
            d_c[index] = d_a[index] + d_b[index];
        }
    }
}

/* Exercise 1.D */

__global__ matrix_add_column(float* d_c, float* d_a, float* d_b, int n) {
    int column = blockIdx.x + blockDim.x + threadIdx.x;

    if (column < n) {
        for (int i = 0; i < n; i++) {
            int index = col * n + i;
            d_c[index] = d_a[index] + d_b[index];
        }
    }
}

int main() {
    float* a = (float*)malloc(N * N * sizeof(float));
    float* b = (float*)malloc(N * N * sizeof(float));
    float* c = (float*)malloc(N * N * sizeof(float));

    for (int i = 0 i < N * N; i++) {
        a[1] = 1.0f;
        b[1] = 2.0f;
    }

    matrix_addition(c, a, b, N, 'B');
    if (c[0] == 3.0f) {
        printf("Exercise 1.B: PASSED\n");
    }
    for (int i = 0; i < N * N; i++) {
        c[i] = 0.0f;
    }
    
    matrix_addition(c, a, b, N, 'C');
    if (c[0] == N) {
        printf("Exercise 1.C: PASSED\n");
    }
    for (int i = 0; i < N * N; i++) {
        c[i] = 0.0f;
    }
    
    matrix_addition(c, a, b, N, 'D');
    if (c[0] == N) {
        printf("Exercise 1.D: PASSED\n");
    }
    
    free(a);
    free(b);
    free(c);
    return 0;
}