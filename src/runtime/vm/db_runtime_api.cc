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
 * \file c_runtime_api.cc
 * \brief Device specific implementations
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/db_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <string>

#include "../runtime_base.h"

using namespace tvm::runtime;
using namespace tvm::runtime::vm;

#ifdef __cplusplus
extern "C" {
#endif

void TVMDBLoadConstant(int64_t const_index, NDArray* out) {
  auto db_runtime = DynBatchRuntime::Current();
  *out = db_runtime->LoadConstant(const_index);
}

void TVMDBInvokePacked(int64_t packed_index, int64_t arity, int64_t output_size,
                       const tvm::runtime::NDArray* args, int64_t num_args) {
  auto db_runtime = DynBatchRuntime::Current();
  db_runtime->InvokePacked(packed_index, arity, output_size, args, num_args);
}

tvm::runtime::vm::Storage TVMDBAllocateStorage(int64_t allocation_size, int64_t alignment,
                                               DLDataType dtype, int64_t device_index) {
  auto db_runtime = DynBatchRuntime::Current();
  // *out = db_runtime->AllocateStorage(allocation_size, alignment, dtype, device_index);
  return db_runtime->AllocateStorage(allocation_size, alignment, dtype, device_index);
}

void TVMDBAllocateTensor(const Storage& storage, int64_t offset, uint32_t ndim, int64_t* shape,
                         DLDataType dtype, NDArray* out) {
  auto db_runtime = DynBatchRuntime::Current();
  *out = db_runtime->AllocTensor(storage, offset, ndim, shape, dtype);
}

void TVMDBAllocateTensorReg(const tvm::runtime::vm::Storage& storage, int64_t offset,
                            tvm::runtime::NDArray shape, DLDataType dtype,
                            tvm::runtime::NDArray* out) {
  auto db_runtime = DynBatchRuntime::Current();
  *out = db_runtime->AllocTensorReg(storage, offset, shape, dtype);
}

void TVMDBDeviceCopy(const NDArray& src_data, const int64_t src_device_index,
                     const int64_t dst_device_index, NDArray* out) {
  auto db_runtime = DynBatchRuntime::Current();
  *out = db_runtime->DeviceCopy(src_data, src_device_index, dst_device_index);
}

void TVMDBReshapeTensor(NDArray& tensor_arr, const NDArray& shape_tensor, NDArray* out) {
  auto db_runtime = DynBatchRuntime::Current();
  *out = db_runtime->ReshapeTensor(tensor_arr, shape_tensor);
}

void TVMDBShapeOf(const NDArray& input_array, NDArray* out) {
  auto db_runtime = DynBatchRuntime::Current();
  *out = db_runtime->ShapeOf(input_array);
}

#ifdef __cplusplus
}
#endif