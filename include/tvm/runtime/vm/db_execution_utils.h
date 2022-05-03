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
 *
 * \file db_execution_utils.h
 * \brief Arena allocator that allocates memory chunks and frees them all during destruction time.
 */
#ifndef TVM_RUNTIME_VM_DB_EXECUTION_UTILS_H_
#define TVM_RUNTIME_VM_DB_EXECUTION_UTILS_H_

#include <stddef.h>
#include <tvm/runtime/ndarray.h>

#include <iostream>
#include <memory>
#include <random>
#include <utility>

namespace tvm {
namespace runtime {
namespace vm {

inline int max(std::initializer_list<int> numbers) {
  int res = std::numeric_limits<int>::min();
  for (auto& num : numbers) {
    res = (res > num) ? res : num;
  }
  return res;
}

inline int64_t NDToInt64(const NDArray& nd) {
  static auto int64_dtype = DataType::Int(64);
  DLDevice cpu_ctx{kDLCPU, 0};
  NDArray cpu_array = nd.CopyTo(cpu_ctx);
  ICHECK_EQ(DataType(cpu_array->dtype), int64_dtype);
  return reinterpret_cast<int64_t*>(cpu_array->data)[0];
}

inline int32_t GetRandom(int32_t lo, int32_t hi) {
  static std::random_device rd;
  /* static std::mt19937 gen(rd()); */
  static std::mt19937 gen(4);
  return std::uniform_int_distribution<>(lo, hi)(gen);
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_VM_DB_EXECUTION_UTILS_H_
