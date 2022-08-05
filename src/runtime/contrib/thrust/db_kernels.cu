// #ifdef USE_CUDA
#include <iostream>

#include "db_kernels.h"

#define CEIL(a, b) (((a) + (b)-1) / (b))
#define FLOOR(a, b) ((a) / (b))
#define REDUCE_SUM_THREAD_COUNT 512
#define CONCAT_COPY_THREAD_COUNT 1024

__global__ void reduce_sum(float** input, float** output, int* input_indices, int batch_size,
                           int hidden_size) {
  int b = blockIdx.x;
  int ii = threadIdx.x;

  int input_start = input_indices[b];
  int input_end = input_indices[b + 1];
  float* output_b = output[b];
  for (int io = 0; io < FLOOR(hidden_size, REDUCE_SUM_THREAD_COUNT); ++io) {
    int i = io * REDUCE_SUM_THREAD_COUNT + ii;
    output_b[i] = 0;
    for (int input_index = input_start; input_index < input_end; ++input_index) {
      output_b[i] += input[input_index][i];
    }
  }

  int io = CEIL(hidden_size, REDUCE_SUM_THREAD_COUNT) - 1;
  int i = io * REDUCE_SUM_THREAD_COUNT + ii;
  if (i < hidden_size) {
    output_b[i] = 0;
    for (int input_index = input_start; input_index < input_end; ++input_index) {
      output_b[i] += input[input_index][i];
    }
  }
}

__global__ void db_concat_copy(float** input, float* output, int batch_size, int flat_size) {
  int b = blockIdx.x;
  int ii = threadIdx.x;
  for (int io = 0; io < FLOOR(flat_size, CONCAT_COPY_THREAD_COUNT); ++io) {
    int i = io * CONCAT_COPY_THREAD_COUNT + ii;
    output[b * flat_size + i] = input[b][i];
  }

  int io = CEIL(flat_size, CONCAT_COPY_THREAD_COUNT) - 1;
  int i = io * CONCAT_COPY_THREAD_COUNT + ii;
  if (i < flat_size) {
    output[b * flat_size + i] = input[b][i];
  }
}

namespace tvm {
namespace contrib {

void reduce_sum_wrapper(float** input, float** output, int* input_indices, int batch_size,
                        int hidden_size) {
  reduce_sum<<<batch_size, REDUCE_SUM_THREAD_COUNT>>>(input, output, input_indices, batch_size,
                                                      hidden_size);
}

void db_concat_copy_wrapper(float** input, float* output, int batch_size, int flat_size) {
  db_concat_copy<<<batch_size, CONCAT_COPY_THREAD_COUNT>>>(input, output, batch_size, flat_size);
}

#undef CEIL
#undef FLOOR
#undef CONCAT_COPY_THREAD_COUNT
#undef REDUCE_SUM_THREAD_COUNT

}  // namespace contrib
}  // namespace tvm
// #endif
