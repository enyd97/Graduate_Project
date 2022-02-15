#include <stdio.h>
#include <dpu.h>
#include <assert.h>
#include <stdint.h>
#include "vector_addition.h"
#include <stdlib.h>

#ifndef DPU_BINARY
#define DPU_BINARY "vector_kernel"
#endif


int main(){
    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;
    T* bufferA; //buffer on main memory
    T* bufferB; //buffer on main memory

    uint32_t input_size_dpu = NR_ELEMENTS * sizeof(T) / NR_DPUS;   // required memory size per 1 dpu


    bufferA = (T*)malloc(NR_ELEMENTS * sizeof(T));
    bufferB = (T*)malloc(NR_ELEMENTS * sizeof(T));

    //make two vectors
    int k = 0;
    for(int i = 0; i < NR_ELEMENTS * sizeof(T); i += sizeof(T)){
        bufferA[k] = k;
        bufferB[k] = k+50;
        k++;
    }
    //print two vector
    printf("Before addition\nA : ");
    k=0;
    for(int i = 0; i < NR_ELEMENTS * sizeof(T); i += sizeof(T)){
        printf("%3d ",bufferA[k]);
        k++;
    }
    printf("\nB : ");
    k=0;
    for(int i = 0; i < NR_ELEMENTS * sizeof(T); i += sizeof(T)){
        printf("%3d ",bufferB[k]);
        k++;
    }
    printf("\n");
    uint32_t i = 0;
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_FOREACH(dpu_set, dpu){
        DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, 0, bufferA + input_size_dpu * i / sizeof(T), NR_ELEMENTS * sizeof(T) / NR_DPUS));
        DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, NR_ELEMENTS * sizeof(T) / NR_DPUS, bufferB + input_size_dpu * i / sizeof(T), NR_ELEMENTS * sizeof(T) / NR_DPUS));
        i++;
    }
    
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    k = 0;
    for(int i = 0; i < NR_ELEMENTS * sizeof(T); i += sizeof(T)){
        bufferA[k] = 0;
        bufferB[k] = 0;
        k++;
    }


    i=0;
    DPU_FOREACH(dpu_set, dpu){
        DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, 0, bufferA + input_size_dpu * i / sizeof(T), NR_ELEMENTS * sizeof(T) / NR_DPUS));
        DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, NR_ELEMENTS * sizeof(T) / NR_DPUS, bufferB + input_size_dpu * i / sizeof(T), NR_ELEMENTS * sizeof(T) / NR_DPUS));
        i++;
    }

    

    //print two vector
    printf("After addition\nA : ");
    k=0;
    for(int i = 0; i < NR_ELEMENTS * sizeof(T); i += sizeof(T)){
        printf("%3d ",bufferA[k]);
        k++;
    }
    printf("\nB : ");
    k=0;
    for(int i = 0; i < NR_ELEMENTS * sizeof(T); i += sizeof(T)){
        printf("%3d ",bufferB[k]);
        k++;
    }
    printf("\n");

    DPU_ASSERT(dpu_free(dpu_set));
    free(bufferA);
    free(bufferB);
}