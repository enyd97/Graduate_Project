#include <stdio.h>
#include <stdint.h>

#define T int32_t
#define NR_TASKLETS 2

typedef struct {
    uint32_t size;
    uint32_t row;
    uint32_t col;
} dpu_arguments_t;