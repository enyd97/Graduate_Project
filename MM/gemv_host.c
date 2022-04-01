#include <stdio.h>
#include <dpu.h>
#include <assert.h>
#include <stdint.h>
#include "gemv.h"
#include <stdlib.h>
#include <time.h>

#ifndef DPU_BINARY
#define DPU_BINARY "gemv_kernel"
#endif

void print(T* addr, unsigned row, unsigned col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%2d ", addr[i * col + j]);
        }
        printf("\n");
    }
}

void mm(T* A, T* B, T* C, unsigned M, unsigned N, unsigned K) {
    T temp;
    for (unsigned i = 0; i < M; i++) {
        for (unsigned k = 0; k < K;k++) {
            temp = A[i * K + k];
            for (unsigned j = 0; j < N; j++) {
                C[N * i + j] += temp * B[M * k + j];
            }
        }
    }
}

int check_mm(T* A, T* B, T* C, unsigned M, unsigned N, unsigned K) {
    printf("check the result of mm\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            T val = 0;
            for (int k = 0; k < K; k++) {
                val += A[i * K + k] * B[k * N + j];
            }

            if ((val - C[i * N + j]) != 0) {
                printf("%d != %d\n", C[i * N + j], val);
                fprintf(stderr, "Result verification failed at (%d, %d)\n", i, j);
                fprintf(stderr, "Test FAILED\n");
                return 1;
            }
        }
    }
    printf(".....\n");
    printf("MM success\n");
    return 0;
}

int main() {
    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus = NR_DPUS;
    unsigned rowA = 32;
    unsigned colA = 32;
    unsigned rowB = 32;
    unsigned colB = 32;
    uint32_t start;
    uint32_t end;
    unsigned i;

    T* bufferA; //buffer on main memory
    T* bufferB; //buffer on main memory
    T* bufferC;
    T* bufferD;

    unsigned row_per_dpu = rowA / NR_DPUS;   // required memory size per 1 dpu


    bufferA = (T*)malloc(rowA * colA * sizeof(T));
    bufferB = (T*)malloc(rowB * colB * sizeof(T));
    bufferC = (T*)malloc(rowA * colB * sizeof(T));
    bufferD = (T*)malloc(rowA * colB * sizeof(T));
    //initialize matrix
    for (int i = 0; i < rowA * rowB; i++) {
        bufferA[i] = i%4;
        bufferB[i] = i%4;
        bufferC[i] = 0;
    }
    //print two vector
    /*
    printf("matrix A \n");
    print(bufferA, rowA, rowB);
    printf("matrix B\n");
    print(bufferB, rowB, colB);
    printf("matrix C\n");*/
    start = clock();
    mm(bufferA, bufferB, bufferC, rowA, colA, rowB);
    end= clock();
    printf("Execution time without DPU : %d\n", end - start);
    
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));


    //printf("check point1\n");
    dpu_arguments_t input_arguments[nr_of_dpus];
    for (int i = 0; i < nr_of_dpus; i++) {
        input_arguments[i].m_size = row_per_dpu;
        input_arguments[i].n_size = colA;
    }

    unsigned sliced_matrix = row_per_dpu * colA;
    unsigned matrix_size = rowA * colA;

    //printf("check point2\n");
    start = clock();
    i = 0;
    //transfer arguments
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i])); //enter index
    }
    //enter byte address
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(input_arguments[0]), DPU_XFER_DEFAULT));
    //printf("check point3\n");
    //transfer divided A matrix
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, bufferA + sliced_matrix * i));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, sliced_matrix * sizeof(T), DPU_XFER_DEFAULT));
    //printf("check poin4\n");
    //transfer whole B matrix 
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, bufferB));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, sliced_matrix * sizeof(T), matrix_size * sizeof(T), DPU_XFER_DEFAULT));

    //printf("check point5\n");
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    //printf("check point6\n");
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, bufferD + sliced_matrix * i));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, (sliced_matrix + matrix_size) * sizeof(T), sliced_matrix * sizeof(T), DPU_XFER_DEFAULT));
    //printf("check point7\n");
    end = clock();

    //print two vector
    print(bufferC, rowA, colA);
    printf("matrix D\n");
    print(bufferD, rowA, colA);
    if(!check_mm(bufferA, bufferB, bufferD, rowA, rowB, colB)){
        printf("Excecution time with DPU : %d\n", end - start);

    }
    DPU_ASSERT(dpu_free(dpu_set));

    free(bufferA);
    free(bufferB);
    free(bufferC);
    free(bufferD);
}