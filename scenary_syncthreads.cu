%%writefile case_warning

/* Escenario del programador */

__global__ void warp_race_condition(int *out) {
    __shared__ int temp[32];
    int tid = threadIdx.x;

    temp[tid] = tid;

    int val = temp[(tid + 1) % 32];

    out[tid] = val;
}

/* Sin omision de __syncthreads */

__global__ void warp_safe(int *out) {
    __shared__ int temp[32];
    int tid = threadIdx.x;

    temp[tid] = tid;

    __syncthreads();

    int val = temp[(tid + 1) % 32];

    out[tid] = val;
}