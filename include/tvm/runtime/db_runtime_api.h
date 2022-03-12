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

/*
 * \file tvm/runtime/c_runtime_api.h
 * \brief TVM runtime library.
 *
 *  The philosophy of TVM project is to customize the compilation
 *  stage to generate code that can used by other projects transparently.
 *  So this is a minimum runtime code gluing, and some limited
 *  memory management code to enable quick testing.
 *
 *  The runtime API is independent from TVM compilation stack and can
 *  be linked via libtvm_runtime.
 *
 *  The common flow is:
 *   - Use TVMFuncListGlobalNames to get global function name
 *   - Use TVMFuncCall to call these functions.
 *
 *  Possible return values of the API functions:
 *  * 0: success
 *  * -1: the error can be retrieved through TVMGetLastError.
 *  * -2: a frontend error occurred and recorded in the frontend.
 */
#ifndef TVM_RUNTIME_DB_RUNTIME_API_H_
#define TVM_RUNTIME_DB_RUNTIME_API_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/vm/arena.h>
#include <tvm/runtime/vm/memory_manager.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Load constant from the executable.
 * \param const_index The constant index in the executable.
 * \param out The result constant array
 *
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMDBLoadConstant(int64_t const_index, tvm::runtime::NDArray* out);

/*!
 * \brief Invoke a packed function.
 * \param packed_index The index of the callee in the executable.
 * \param arity The arity of the function
 * \param output_size The number of outputs of the function
 * \param args Pointer to args
 * \param num_args number of arguments
 *
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMDBInvokePacked(int64_t packed_index, int64_t arity, int64_t output_size,
                              const tvm::runtime::NDArray* args, int64_t num_args);

/*!
 * \brief Allocate a memory storage object.
 * \param allocation_size The size of the storage to be allocated.
 * \param alignment The alignment of the storage to be allocated.
 * \param dtype The datatype of the storage to be allocated.
 * \param device_index The device the memory is to be allocated on
 * \param out The allocated storage object
 *
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMDBAllocateStorage(int64_t allocation_size, int64_t alignment, DLDataType dtype,
                                 int64_t device_index, tvm::runtime::vm::Storage* out);

/*!
 * \brief Allocate a tensor.
 * \param storage The storage to allocate from.
 * \param offset The offset into the storage to allocate from
 * \param ndim The number of dimensions.
 * \param shape The shape of tensor.
 * \param dtype The datatype of tensor to be allocated
 * \param out The allocated tensor object
 *
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMDBAllocateTensor(const tvm::runtime::vm::Storage& storage, int64_t offset,
                                uint32_t ndim, int64_t* shape, DLDataType dtype,
                                tvm::runtime::NDArray* out);

/*!
 * \brief Allocate a tensor.
 * \param storage The storage to allocate from.
 * \param offset The offset into the storage to allocate from
 * \param shape The shape of tensor.
 * \param dtype The datatype of tensor to be allocated
 * \param out The allocated tensor object
 *
 * \return 0 when success, nonzero when failure happens
 */
TVM_DLL int TVMDBAllocateTensorReg(const tvm::runtime::vm::Storage& storage, int64_t offset,
                                   tvm::runtime::NDArray shape, DLDataType dtype,
                                   tvm::runtime::NDArray* out);

/*!
 * \brief Copy a tensor between devices.
 * \param src_data The tensor to copy from.
 * \param src_device_index The index of the source device
 * \param dst_device_index The index of the destination device
 * \param out The copied tensor object
 */
TVM_DLL int TVMDBDeviceCopy(const tvm::runtime::NDArray& src_data, const int64_t src_device_index,
                            const int64_t dst_device_index, tvm::runtime::NDArray* out);

/*!
 * \brief Rehape a tensor.
 * \param tensor_arr The tensor to be reshaped.
 * \param shape_tensor The new shape
 * \param out The reshaped tensor object
 */
TVM_DLL int TVMDBReshapeTensor(tvm::runtime::NDArray& tensor_arr,
                               const tvm::runtime::NDArray& shape_tensor,
                               tvm::runtime::NDArray* out);

/*!
 * \brief Obtain the shape of a tensor.
 * \param input_array The input tensor.
 * \param out The shape of the input tensor
 */
TVM_DLL int TVMDBShapeOf(const tvm::runtime::NDArray& input_array, tvm::runtime::NDArray* out);

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
#endif  // TVM_RUNTIME_DB_RUNTIME_API_H_
