#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdint.h>
#include <unistd.h>

#define T int32_t
#define SIZE 2048

#define NUM_OF_THREADS 16
#define TILING_LENGTH 128

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
typedef struct {
    unsigned a;
    unsigned b;
    unsigned c;
} matrix_argument;

T matrixA[SIZE][SIZE];
T matrixB[SIZE][SIZE];
T matrixC[SIZE][SIZE];
T verifying_matrix[SIZE][SIZE];

void matrix_initialize();
void result_matrix_initialize();
void ijk_matrix_multiplication();
void ikj_matrix_multiplication();
void verify_matrix();
void* threads_ijk_matrix_multiplication();
void print_matrix();
void* threads_block_multiplication();
void* threads_tiled_block_multiplication();

void main() {
    struct timeval start;
    struct timeval end;

    //initialize matrix
    matrix_initialize();
    result_matrix_initialize();

    //ijk multiplication
    gettimeofday(&start, NULL);
    ijk_matrix_multiplication();
    gettimeofday(&end, NULL);
    uint64_t result0 = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
    printf("ijk multiplication : %ld\n", result0);

    //matrix to verify the result.
    for (unsigned i = 0; i < SIZE; i++) for (unsigned j = 0; j < SIZE; j++) verifying_matrix[i][j] = matrixC[i][j];

    //ikj multiplication
    result_matrix_initialize();
    gettimeofday(&start, NULL);
    ikj_matrix_multiplication();
    gettimeofday(&end, NULL);
    verify_matrix(matrixC);
    uint64_t result1 = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
    printf("ikj multiplication : %ld\n", result1);
    printf("speed-up from ijk %.2f\n", (double)result0 / (double)result1);

    //multi-threading ijk mm
    gettimeofday(&start, NULL);
    pthread_t threads1[NUM_OF_THREADS];
    T argument_buffer1[NUM_OF_THREADS];
    result_matrix_initialize();
    unsigned i;
    for (i = 0; i < NUM_OF_THREADS; i++) {
        argument_buffer1[i] = i;
        pthread_create(&threads1[i], NULL, threads_ijk_matrix_multiplication, (void*)&argument_buffer1[i]);
    }
    for (unsigned i = 0; i < NUM_OF_THREADS; i++) {
        pthread_join(threads1[i], NULL);
    }
    gettimeofday(&end, NULL);
    verify_matrix(matrixC);
    uint64_t result2 = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
    printf("multi-threading devied A multiplication : %ld\n", result2);
    printf("speed-up from ijk %.2f\n", (double)result0 / (double)result2);

    //multi-threading with block
    result_matrix_initialize();
    gettimeofday(&start, NULL);
    pthread_t threads2[NUM_OF_THREADS * 4];
    matrix_argument* argument_buffer2;
    for (unsigned j = 0; j < 4; j++) { //interate 4 times.
        for (unsigned i = 0; i < NUM_OF_THREADS; i++) {
            argument_buffer2 = (matrix_argument*)malloc(sizeof(matrix_argument));
            argument_buffer2->a = ((i / 4) * (SIZE / 4) * SIZE) + ((j % 4) * (SIZE / 4));
            argument_buffer2->b = ((j % 4) * (SIZE / 4) * SIZE) + ((i % 4) * (SIZE / 4));
            argument_buffer2->c = ((i / 4) * (SIZE / 4) * SIZE) + ((i % 4) * (SIZE / 4));
            pthread_create(&threads2[j * 16 + i], NULL, threads_block_multiplication, (void*)argument_buffer2);
        }
        for (unsigned i = 0; i < NUM_OF_THREADS; i++) pthread_join(threads2[j * 16 + i], NULL);
    }
    gettimeofday(&end, NULL);
    verify_matrix(matrixC);
    uint64_t result3 = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
    printf("multi-threading 16 blocks multiplication : %ld\n", result3);
    printf("speed-up from ijk %.2f\n", (double)result0 / (double)result3);


    //multi-threading with block and tiling the block
    result_matrix_initialize();
    gettimeofday(&start, NULL);
    pthread_t threads3[NUM_OF_THREADS * 4];
    matrix_argument* argument_buffer3;
    for (unsigned j = 0; j < 4; j++) { //interate 4 times.
        for (unsigned i = 0; i < NUM_OF_THREADS; i++) {
            argument_buffer3 = (matrix_argument*)malloc(sizeof(matrix_argument));
            argument_buffer3->a = ((i / 4) * (SIZE / 4) * SIZE) + ((j % 4) * (SIZE / 4));
            argument_buffer3->b = ((j % 4) * (SIZE / 4) * SIZE) + ((i % 4) * (SIZE / 4));
            argument_buffer3->c = ((i / 4) * (SIZE / 4) * SIZE) + ((i % 4) * (SIZE / 4));
            pthread_create(&threads3[j * 16 + i], NULL, threads_tiled_block_multiplication, (void*)argument_buffer3);
        }
        for (unsigned i = 0; i < NUM_OF_THREADS; i++) pthread_join(threads3[j * 16 + i], NULL);
    }
    gettimeofday(&end, NULL);
    verify_matrix(matrixC);
    uint64_t result4 = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
    printf("16 threads multi-threading blocking and tiling multiplication : %ld\n", result4);
    printf("speed-up from ijk %.2f\n", (double)result0 / (double)result4);

}

void matrix_initialize() {
    for (unsigned i = 0; i < SIZE; i++) {
        for (unsigned j = 0; j < SIZE; j++) {
            matrixA[i][j] = rand() % 4;
            matrixB[i][j] = rand() % 4;
        }
    }
}

void result_matrix_initialize() {
    for (unsigned i = 0; i < SIZE; i++) {
        for (unsigned j = 0; j < SIZE; j++) {
            matrixC[i][j] = 0;
        }
    }
}

void ijk_matrix_multiplication() {
    for (unsigned i = 0; i < SIZE; i++) {
        for (unsigned j = 0; j < SIZE; j++) {
            for (unsigned k = 0; k < SIZE; k++) {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
}

void ikj_matrix_multiplication() {
    T temp;
    for (unsigned i = 0; i < SIZE; i++) {
        for (unsigned k = 0; k < SIZE;k++) {
            temp = matrixA[i][k];
            for (unsigned j = 0; j < SIZE; j++) {
                matrixC[i][j] += temp * matrixB[k][j];
            }
        }
    }
}

void verify_matrix(T current[][SIZE]) {
    for (unsigned i = 0; i < SIZE; i++) {
        for (unsigned j = 0; j < SIZE; j++) {
            if (current[i][j] != verifying_matrix[i][j]) {
                printf("verification fail\n");
                return;
            }
        }
    }
    printf("verifying OK\n");
}

void print_matrix(T current[][SIZE]) {
    for (unsigned i = 0; i < SIZE; i++) {
        for (unsigned j = 0; j < SIZE; j++) {
            printf("%4d", current[i][j]);
        }
    }
    printf("\n");
}

void* threads_ijk_matrix_multiplication(void* arg) {
    unsigned* tid = (unsigned*)arg;
    unsigned start = (unsigned)(*tid) * SIZE / NUM_OF_THREADS;

    for (unsigned i = start; i < start + SIZE / NUM_OF_THREADS; i++) {
        for (unsigned j = 0; j < SIZE; j++) {
            T temp = 0;
            for (unsigned k = 0; k < SIZE; k++) {
                temp += matrixA[i][k] * matrixB[k][j];
            }
            pthread_mutex_lock(&lock);
            matrixC[i][j] = temp;
            pthread_mutex_unlock(&lock);
        }
    }

}

void* threads_block_multiplication(void* arg) {
    matrix_argument *arguments = (matrix_argument*)arg;
    T temp;
    unsigned a_i = arguments->a / SIZE;
    unsigned a_j = arguments->a % SIZE;
    unsigned b_i = arguments->b / SIZE;
    unsigned b_j = arguments->b % SIZE;
    unsigned c_i = arguments->c / SIZE;
    unsigned c_j = arguments->c % SIZE;

    for (unsigned i = 0; i < SIZE / 4; i++) {
        for (unsigned k = 0; k < SIZE / 4;k++) {
            temp = matrixA[a_i + i][a_j + k];
            for (unsigned j = 0; j < SIZE / 4; j++) {
                matrixC[c_i + i][c_j + j] += temp * matrixB[b_i + k][b_j + j];
            }
        }
    }
}

void* threads_tiled_block_multiplication(void* arg) {
    matrix_argument* arguments = (matrix_argument*)arg;
    T temp;
    unsigned a_i = arguments->a / SIZE;
    unsigned a_j = arguments->a % SIZE;
    unsigned b_i = arguments->b / SIZE;
    unsigned b_j = arguments->b % SIZE;
    unsigned c_i = arguments->c / SIZE;
    unsigned c_j = arguments->c % SIZE;

    for (unsigned k = 0; k < SIZE / 4; k += TILING_LENGTH) {
        for (unsigned i = 0; i < SIZE / 4; i+= TILING_LENGTH) {
            for (unsigned j = 0; j < SIZE / 4; j+= TILING_LENGTH) {
                for(unsigned q = 0; q < TILING_LENGTH; q++){
                    for(unsigned p = 0; p < TILING_LENGTH; p++){
                        temp = matrixA[a_i + i + p][a_j + k + q];
                        for(unsigned r = 0; r< TILING_LENGTH; r++){
                            matrixC[c_i + i + p][c_j + j + r] += temp * matrixB[b_i + k + q][b_j + j + r];
                        }
                    }
                }
            }
        }
    }
}

