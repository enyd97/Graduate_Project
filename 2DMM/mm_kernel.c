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


uint32_t BLOCK_SIZE_PER_TASKLET = NR_ELEMENTS * sizeof(T) / NR_DPUS / NR_TASKLETS;

void vector_addition(T *bufferB, T *bufferA){
    for (unsigned int i = 0; i < BLOCK_SIZE_PER_TASKLET / sizeof(T); i++){
        bufferB[i] += bufferA[i];
    }
}
 
int main(){
    unsigned int tasklet_id = me();
    

    uint32_t mram_base_addr_A = (uint32_t)(DPU_MRAM_HEAP_POINTER + (tasklet_id * BLOCK_SIZE_PER_TASKLET));
    uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER + NR_ELEMENTS *sizeof(T) / NR_DPUS + (tasklet_id * BLOCK_SIZE_PER_TASKLET));

    T *cache_A = (T *)mem_alloc(BLOCK_SIZE_PER_TASKLET);
    T *cache_B = (T *)mem_alloc(BLOCK_SIZE_PER_TASKLET);

    //for (unsigned int byte_index = 0; byte_index < BLOCK_SIZE; byte_index += BLOCK_SIZE_PER_TASKLET){
    
    mram_read((__mram_ptr void const *)(mram_base_addr_A), cache_A, BLOCK_SIZE_PER_TASKLET);
    mram_read((__mram_ptr void const *)(mram_base_addr_B), cache_B, BLOCK_SIZE_PER_TASKLET);


    vector_addition(cache_B, cache_A);

    barrier_wait(&barrier);

    //if(tasklet_id == 0){
    mram_write(cache_A, (__mram_ptr void *)(mram_base_addr_A), BLOCK_SIZE_PER_TASKLET);
    mram_write(cache_B, (__mram_ptr void *)(mram_base_addr_B), BLOCK_SIZE_PER_TASKLET);
    //}
    //}
}