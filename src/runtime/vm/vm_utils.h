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
 * \file src/runtime/vm/vm.cc
 * \brief The Relay virtual machine runtime.
 */

#include <dmlc/memory_io.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/vm/lazy_executor.h>
#include <tvm/runtime/vm/vm.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../file_utils.h"

//////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CALL_HEADER(func)                                \
  {                                                           \
    cudaError_t e = (func);                                   \
    ICHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                 \
  }
//////////////////////////////////////////////////

using namespace tvm::runtime;

namespace tvm {
namespace runtime {
namespace vm {

/* Other utils */

/*!
 * \brief Convert NDArray to vector
 *
 * \param shape_tensor The source NDArray.
 *
 * \return The vector.
 */
std::vector<int64_t> ToShape(NDArray shape_tensor);

/*!
 * \brief Copy objects (arrays or lists of arrays) across devices
 *
 * \param src The source object.
 * \param dev The destinationd device.
 *
 * \return The copied object.
 */
ObjectRef CopyTo(ObjectRef src, const DLDevice& dev);

/*!
 * \brief A simple procedure to write to all locations of an
 * NDArray/DLTensor to check for invalid accesses.
 *
 * \param array The array to be tested.
 */
void TestNDArray(const NDArray& array);
void TestNDArray(DLTensor* array);

/*!
 * \brief Create a gathered/copied NDArray from a batch of scattered tensors.
 *
 * \param nodes The OpNodes that contain the scattered tensors.
 * \param arg_num The index of the scattered argument tensors.
 * \param allocator The allocator to allocate from.
 * \param gpu_execution Whether or not we're executing on the GPU.
 *
 * \return The created gathered array
 */
void FillInPointersScatter(void** host_raw_ptrs, size_t size,
                           const std::vector<OpNode<DLTensor*>*>& nodes, int arg_num,
                           Allocator* allocator);

NDArray CreatePointerNDArray(const std::vector<OpNode<NDArray>*>& nodes, int arg_num);
NDArray CreatePointerNDArray(const std::vector<OpNode<DLTensor*>*>& nodes, int arg_num,
                             Allocator* allocator);
DLTensor* CreatePointerDLTensor(const std::vector<OpNode<DLTensor*>*>& nodes, int arg_num,
                                Allocator* allocator);
DLTensor* CreateConcatenatedDLTensor(const std::vector<OpNode<DLTensor*>*>& nodes, int arg_num,
                                     Allocator* allocator);
/*!
 * \brief A simple procedure to write to all locations of an gathered
 * NDArray/DLTensor to check for invalid accesses.
 *
 * \param ptr_array The array to be tested.
 * \param sample A sample scattered array to compute size .
 * \param batch_size The number of gathered tensors.
 */
void TestPointerNDArray(const NDArray& ptr_array, const NDArray& sample, int64_t batch_size);

/*!
 * \brief Async memory copy function.
 *
 * \param handle The destination array.
 * \param data The source on the host .
 * \param nbytes The number of bytes to be copied.
 */
// void ArrayCopyFromBytesAsync(DLTensor* handle, const void* data, size_t nbytes);
inline void ArrayCopyFromBytesAsync(void* dst, const void* src, size_t nbytes) {
  // CUDA_CALL_HEADER(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice));
  cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice);
}

/* Invoking packed functions */

/*!
 * \brief Invoke a PackedFunction (refactored out to avoid code
 * duplication). This functions assumes that all ADT args have already
 * been unrolled into their constituent NDArrays
 *
 * \param func The PackedFunction to be invoked.
 * \param arg_count The number of arguments to the PackedFunction.
 * \param args Arguments to the PackedFunction.
 *
 * \note The return value will be stored in the last output_size slots of args.
 */
template <typename TensorType>
void InvokePackedFnUnrolled(const size_t func_idx, const PackedFunc& func, TensorType* args,
                            int arity);

/*!
 * \brief Invoke a batch PackedFunction (refactored out to avoid code
 * duplication) on a batch of OpNodes . This functions assumes that
 * all ADT args have already been unrolled into their constituent
 * NDArrays
 *
 * \param func The PackedFunction to be invoked.
 * \param arg_count The number of arguments to the PackedFunction.
 * \param output_size The number of outputs of the PackedFunction.
 * \param arg_modes The argument modes of the PackedFunction.
 * \param nodes The OpNodes as a batch.
 *
 * \note The return value will be stored in the last output_size slots of args.
 */
template <typename TensorType>
void InvokePackedFnBatchedUnrolled(const size_t func_idx, const PackedFunc& func, Index arg_count,
                                   const std::vector<DBBatchedArgMode>& arg_modes,
                                   const std::vector<OpNode<TensorType>*>& nodes);

/*!
 * \brief Invoke a PackedFunction (refactored out to avoid code duplication)
 *
 * \param func The PackedFunction to be invoked.
 * \param arg_count The number of arguments to the PackedFunction.
 * \param output_size The number of outputs of the PackedFunction.
 * \param args Arguments to the PackedFunction.
 * \param num_args number of arguments to the PackedFunction.
 *
 * \note The return value will be stored in the last output_size slots of args.
 */
void InvokePackedFn(const PackedFunc& func, Index arg_count, Index output_size,
                    const ObjectRef* args, int64_t num_args,
                    const std::vector<DBBatchedArgMode>& arg_modes, bool batched = false,
                    bool scattered_kernels = false);

/*!
 * \brief Print a DLTensor for debugging purposes
 *
 * \param tensor The DLTensor.
 *
 * \note The return value will be represent a debug string for the input tensor.
 */
inline std::string GetDLTensorInfo(const DLTensor* tensor) {
  std::stringstream ss;
  ss << "Tensor (" << tensor->dtype << ") [";
  for (int i = 0; i < tensor->ndim; ++i) {
    ss << tensor->shape[i] << " ";
  }
  ss << "] on device (" << tensor->device.device_type << ", " << tensor->device.device_id << ")";
  if (tensor->data == nullptr) {
    ss << " with null data";
  }
  return ss.str();
}

inline bool CheckEqualShape(const DLTensor& t1, const DLTensor& t2) {
  if (t1.ndim != t2.ndim) {
    return false;
  }

  for (int i = 0; i < t1.ndim; ++i) {
    if (t1.shape[i] != t2.shape[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
