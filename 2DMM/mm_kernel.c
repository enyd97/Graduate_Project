#include <stdio.h>
#include <stdint.h>
#include <mram.h>
#include "mm.h"
#include <defs.h>
#include <alloc.h>
#include <barrier.h>

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

BARRIER_INIT(barrier, NR_TASKLETS);
//__host 는 호스트에서 바로 접근 가능함.

void matrix_multiplication(T *A, T *B, T *C, uint32_t row, uint32_t col){
    for(unsigned i = 0; i < col; i++){
        C[i] = 0;
        for(unsigned j = 0; j < row; j++){
            C[i] += A[j] * B[col * j + i]; 
        }
    }
}


 
int main(){
    unsigned int tasklet_id = me();
    
    uint32_t input_size = DPU_INPUT_ARGUMENTS.size;
    uint32_t input_row = DPU_INPUT_ARGUMENTS.row;
    uint32_t input_col = DPU_INPUT_ARGUMENTS.col;

    uint32_t A_PER_TASKLETS = input_row * sizeof(T) / NR_TASKLETS;
    uint32_t B_PER_TASKLETS = input_row * input_col * sizeof(T) / NR_TASKLETS;
    uint32_t C_PER_TASKLETS = input_col * sizeof(T) / NR_TASKLETS;

    uint32_t mram_base_addr_A = (uint32_t)(DPU_MRAM_HEAP_POINTER + (tasklet_id * A_PER_TASKLETS));
    uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER + input_row *sizeof(T) + (tasklet_id * B_PER_TASKLETS));

    T *cache_A = (T *)mem_alloc(A_PER_TASKLETS);
    T *cache_B = (T *)mem_alloc(B_PER_TASKLETS);
    T *cache_C = (T *)mem_alloc(C_PER_TASKLETS);
    //for (unsigned int byte_index = 0; byte_index < BLOCK_SIZE; byte_index += BLOCK_SIZE_PER_TASKLET){
    
    mram_read((__mram_ptr void const *)(mram_base_addr_A), cache_A, A_PER_TASKLETS);
    mram_read((__mram_ptr void const *)(mram_base_addr_B), cache_B, B_PER_TASKLETS);


    //matrix_multiplication(cache_A, cache_B, cache_C, input_row, input_col);

    barrier_wait(&barrier);

    //if(tasklet_id == 0){
    mram_write(cache_A, (__mram_ptr void *)(mram_base_addr_A), A_PER_TASKLETS);
    //}
    //}
}