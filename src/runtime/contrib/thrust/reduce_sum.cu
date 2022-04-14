#ifdef USE_CUDA
#include "reduce_sum.h"

#define CEIL(a, b) (((a) + (b)-1) / (b))

#define FLOOR(a, b) ((a) / (b))

__global__ void reduce_sum(float** input, float** output, int* input_indices, int batch_size,
                           int hidden_size) {
  int b = blockIdx.x;
  int ii = threadIdx.x;

  int input_start = input_indices[b];
  int input_end = input_indices[b + 1];
  for (int io = 0; io < FLOOR(hidden_size, 256); ++io) {
    int i = io * 256 + ii;
    output[b][i] = 0;
    for (int input_index = input_start; input_index < input_end; ++input_index) {
      output[b][i] += input[input_index][i];
    }
  }

  int io = CEIL(hidden_size, 256) - 1;
  int i = io * 256 + ii;
  if (i < hidden_size) {
    output[b][i] = 0;
    for (int input_index = input_start; input_index < input_end; ++input_index) {
      output[b][i] += input[input_index][i];
    }
  }
}

namespace tvm {
namespace contrib {

void reduce_sum_wrapper(float** input, float** output, int* input_indices, int batch_size,
                        int hidden_size) {
  reduce_sum<<<batch_size, 256>>>(input, output, input_indices, batch_size, hidden_size);
}

}  // namespace contrib
}  // namespace tvm
#endif
