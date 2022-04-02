#include <stdio.h>
#include <stdint.h>

#define T int64_t
#define NR_DPUS 16
#define NR_TASKLETS 1
typedef struct {
    unsigned m_size;
    unsigned n_size;
} dpu_arguments_t;