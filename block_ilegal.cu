%%writefile error_demo.cu
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel() {}

void intentarLanzamiento(int hiloX, int hiloY, const char* descripcion) {
    dim3 blockSize(hiloX, hiloY);
    dim3 gridSize(1, 1);
    
    int totalHilos = hiloX * hiloY;
    
    printf("\n--- %s ---\n", descripcion);
    printf("Intentando lanzar bloque de %d x %d = %d hilos...\n", hiloX, hiloY, totalHilos);

    dummyKernel<<<gridSize, blockSize>>>();
    
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        printf("FALLÓ. La GPU dice: '%s'\n", cudaGetErrorString(error));
        printf("   (Código de error: %d)\n", error);
    }
    else {
        printf("ÉXITO. El kernel se ejecutó correctamente.\n");
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Tu GPU es: %s\n", prop.name);
    printf("Límite Físico de tu GPU (Max Threads per Block): %d\n", prop.maxThreadsPerBlock);
    printf("--------------------------------------------------\n");

    intentarLanzamiento(32, 32, "CASO 1: Código del Estudiante (1024 hilos)");

    intentarLanzamiento(32, 64, "CASO 2: Simulando el error (2048 hilos)");

    return 0;
}