/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file Use external Thrust library call
 */

#ifndef TVM_RUNTIME_CONTRIB_REDUCE_SUM_H_
#define TVM_RUNTIME_CONTRIB_REDUCE_SUM_H_

namespace tvm {
namespace contrib {

// #ifdef USE_CUDA
// void reduce_sum_wrapper(float** input, float** output, int* input_indices, int batch_size,
// int hidden_size);

// void db_concat_copy_wrapper(float** input, float* output, int batch_size, int flat_size);
// #else
inline void reduce_sum_wrapper(float** input, float** output, int* input_indices, int batch_size,
                               int hidden_size) {
  ICHECK(false);
}

inline void db_concat_copy_wrapper(float** input, float* output, int batch_size, int flat_size) {
  ICHECK(false);
}
// #endif
}  // namespace contrib
}  // namespace tvm

#endif
