# CUDA Parallel Programming Exercises

Este repositorio contiene ejercicios prácticos de programación paralela con CUDA.

## Contenido

### 1. Suma de Matrices
- **Archivos:** 
  - `matrix_add.cu` - Implementación básica
  - `matrix_add_test.cu` - Testing de rendimiento con múltiples estrategias
- **Descripción:** Implementa tres estrategias de suma de matrices:
  - Element-wise (1.B): Un thread por elemento
  - Row-wise (1.C): Un thread por fila
  - Column-wise (1.D): Un thread por columna

### 2. Multiplicación Matriz-Vector
- **Archivos:**
  - `matrix_vector.cu` - Implementación básica con visualización
  - `matrix_vector_test.cu` - Benchmark de rendimiento
- **Descripción:** Operación A[i] = Σ(B[i][j] + C[j])
- **Métricas:** Tiempo de ejecución, GFLOPS, Bandwidth

## Entorno de Desarrollo

Estos ejercicios fueron desarrollados en **Google Colab** utilizando la GPU integrada del entorno.

### Compilación en Colab

```python
!nvcc archivo.cu -o ejecutable
```

**Ejemplos:**
```python
!nvcc matrix_add_test.cu -o matrix_add_test
!nvcc matrix_vector_test.cu -o matrix_vector_test
```

### Ejecución en Colab

```python
!./ejecutable
```

## Requisitos

- Google Colab con GPU habilitada (Runtime → Change runtime type → GPU)
- O alternativamente: CUDA Toolkit y GPU NVIDIA local

## Resultados

Los programas de testing muestran:
- ✅ Tiempo promedio de ejecución
- ✅ Rendimiento en GFLOPS
- ✅ Ancho de banda en GB/s

---
*Ejercicios de programación paralela con CUDA*
