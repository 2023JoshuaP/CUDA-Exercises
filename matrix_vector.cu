%%writefile matrix_vector.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 64

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

void matrix_vector_host(float* a, float* b, float* c, int n) {
    int size = n * n * sizeof(float);
    int size_vector = n * sizeof(float);
    float* d_a, * d_b, * d_c;

    cudaMalloc((void**)&d_a, size_vector);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size_vector);

    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size_vector, cudaMemcpyHostToDevice);

    int threads_block = 64;
    int threads_grid = (n + threads_block - 1) / threads_block;

    printf(">>> [GPU] Lanzando kernel (Grid: %d, Block: %d)\n", threads_grid, threads_block);
    matrix_vector<<<threads_grid, threads_block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(a, d_a, size_vector, cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

void print_preview(const char* label, float* data, int rows, int cols, int limit) {
    printf("\n--- %s (Vista Previa %dx%d) ---\n", label, limit, limit);
    for (int i = 0; i < limit && i < rows; i++) {
        printf("Fila %02d: [ ", i);
        for (int j = 0; j < limit && j < cols; j++) {
            if (cols > 1) printf("%5.1f ", data[i * cols + j]); 
            else printf("%5.1f ", data[i]);
        }
        printf("... ]\n");
    }
    printf("...\n");
}

int main() {
    size_t size_mat = N * N * sizeof(float);
    size_t size_vec = N * sizeof(float);

    float* a = (float*)malloc(size_vec);
    float* b = (float*)malloc(size_mat);
    float* c = (float*)malloc(size_vec);

    for(int i = 0; i < N * N; i++) b[i] = (float)(i % 10);
    for(int i = 0; i < N; i++) c[i] = 1.0f;

    print_preview("INPUT MATRIZ B", b, N, N, 8);
    print_preview("INPUT VECTOR C", c, N, 1, 8);

    matrix_vector_host(a, b, c, N);

    print_preview("OUTPUT VECTOR A", a, N, 1, 8);

    printf("\nVerificación rápida A[0]: %.2f\n", a[0]);

    free(a); free(b); free(c);
    return 0;
}