#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_WIDTH 32

__global__ void tiled_mm_kernel(int32_t *a, int32_t *b, int32_t *c, int M, int K, int N) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * BLOCK_WIDTH + ty;
    int Col = bx * BLOCK_WIDTH + tx;

    int32_t value = 0;

    __shared__ int32_t A_block[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ int32_t B_block[BLOCK_WIDTH][BLOCK_WIDTH];

    for (int index = 0; index < ceil(K / (float)BLOCK_WIDTH); index++) {
        if ((Row < M) && (index * BLOCK_WIDTH + tx < K))
            A_block[ty][tx] = a[Row * K + index * BLOCK_WIDTH + tx];
        else
            A_block[ty][tx] = 0;

        if ((Col < N) && (index * BLOCK_WIDTH + ty) < K)
            B_block[ty][tx] = b[(index * BLOCK_WIDTH + ty) * N + Col];
        else
            B_block[ty][tx] = 0;

        __syncthreads();

        for (int k = 0; k < BLOCK_WIDTH; k++) {
            value += A_block[ty][k] * B_block[k][tx];
        }

        __syncthreads();

        if ((Row < M) && (Col < N))
            c[Row * N + Col] = value;
    }
}

int Test_mm(int32_t *A, int32_t *B, int32_t *C, int M, int K, int N) {
    printf("Verifying all results...\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t val = 0;
            for (int k = 0; k < K; k++) {
                val += A[i * K + k] * B[k * N + j];
            }

            if (val - C[i * N + j] != 0) {
                printf("%d != %d\n", C[i * N + j], val);
                fprintf(stderr, "Result verification failed at (%d, %d)\n", i, j);
                fprintf(stderr, "Test FAILED\n");
                return -1;
            }

            // printf("%5d", val);
        }
        // printf("\n");
    }
    printf(".....\n");
    printf("Test PASSED\n");

    return 0;
}

int Just_mm(int32_t *A, int32_t *B, int32_t *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t val = 0;
            for (int k = 0; k < K; k++) {
                val += A[i * K + k] * B[k * N + j];
            }
        }
    }
    return 0;
}

int print_matrix(int32_t *C, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%5d", C[i * N + j]);
        }
        printf("\n");
    }
    printf(".....\n");

    return 0;
}

int main(void) {
    int32_t *a, *b, *c;  // memory in host
    int *d_a, *d_b, *d_c;
    int M, N, K;
    int size = sizeof(int32_t);

    M = 1024;
    N = 1024;
    K = 1024;

    a = (int32_t *)malloc(M * K * size);
    b = (int32_t *)malloc(K * N * size);
    c = (int32_t *)malloc(M * N * size);

    // initialize matrix
    for (int i = 0; i < M * K; i++) {
        a[i] = i;
    }
    for (int i = 0; i < K * N; i++) {
        b[i] = i;
    }

    cudaMalloc((void **)&d_a, M * K * size);
    cudaMalloc((void **)&d_b, K * N * size);
    cudaMalloc((void **)&d_c, M * N * size);

    dim3 threadsPerBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 blockPerGrid(ceil(M / (float)BLOCK_WIDTH), ceil(N / (float)BLOCK_WIDTH));

    int start = clock();

    cudaMemcpy(d_a, a, M * K * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K * N * size, cudaMemcpyHostToDevice);

    tiled_mm_kernel<<<blockPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, M, K, N);

    cudaMemcpy(c, d_c, M * N * size, cudaMemcpyDeviceToHost);

    int end = clock();
    int tiling_time = end - start;

    start = clock();
    Just_mm(a, b, c, M, K, N);
    end = clock();

    int just_mm_time = end - start;
    Test_mm(a, b, c, M, K, N);

    printf("Compare Time\n");
    printf("ijk mm time : %d\n", just_mm_time);
    printf("Tiling mm time : %d\n", tiling_time);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
