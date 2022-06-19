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

#include "vm_utils.h"

#include <dmlc/memory_io.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/vm/arena.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/runtime/vm/vm_profiling.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../contrib/thrust/db_kernels.h"
#include "../file_utils.h"

// //////////////////////////////////////////////////
// #include <cuda.h>
// #include <cuda_runtime.h>

// #include "../cuda/cuda_common.h"
// //////////////////////////////////////////////////

using namespace tvm::runtime;

namespace tvm {
namespace runtime {
namespace vm {

/* Other utils */
std::string ShapeToString(const ShapeTuple& st) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < st.size(); ++i) {
    ss << st[i];
    if (i < st.size() - 1) ss << ", ";
  }
  ss << "]";
  return ss.str();
}

std::vector<int64_t> ToShape(NDArray shape_tensor) {
  std::vector<int64_t> shape;
  auto rank = shape_tensor.Shape().size();
  auto dtype = shape_tensor.DataType();

  // For 0-rank shapes we need to allocate a single scalar.
  if (rank == 0) {
    return shape;
  }

  // Otherwise we should be rank-1, and we will extract the number of dimensions
  // for the output vector.
  ICHECK_EQ(rank, 1U) << "shape tensor should be a k-length vector, found " << rank;
  int64_t ndim = shape_tensor.Shape().at(0);
  shape.resize(ndim);

  const DLTensor* dl_tensor = shape_tensor.operator->();
  if (dtype.is_int() && dtype.bits() == 32 && dtype.lanes() == 1) {
    int32_t* dims = reinterpret_cast<int32_t*>(dl_tensor->data);
    shape.assign(dims, dims + ndim);
  } else if (dtype.is_int() && dtype.bits() == 64 && dtype.lanes() == 1) {
    int64_t* dims = reinterpret_cast<int64_t*>(dl_tensor->data);
    shape.assign(dims, dims + ndim);
  } else {
    LOG(FATAL) << "invalid shape tensor datatype: " << dtype;
  }

  return shape;
}

ObjectRef CopyTo(ObjectRef src, const DLDevice& dev) {
  if (src->IsInstance<NDArray::ContainerType>()) {
    auto nd_array = Downcast<NDArray>(src);
    // TODO(mbs): Should respect device id also.
    if (nd_array->device.device_type != dev.device_type) {
      VLOG(2) << "copying from " << nd_array->device.device_type << " to " << dev.device_type;
      return nd_array.CopyTo(dev);
    }
    return src;
  } else {
    ICHECK(src->IsInstance<ADTObj>())
        << "VM data must be NDArray or a list of NDArray, but received: " << src->_type_key;
    std::vector<ObjectRef> ret;
    ADT adt = Downcast<ADT>(src);
    for (size_t i = 0; i < adt.size(); i++) {
      ret.push_back(CopyTo(adt[i], dev));
    }
    return ADT(adt->tag, ret.begin(), ret.end());
  }
}

/* Utilities for invoking packed functions */
void TestNDArray(DLTensor* array) {
  size_t total_nums = 1;
  for (int i = 0; i < array->ndim; ++i) {
    total_nums *= array->shape[i];
  }

  for (size_t j = 0; j < total_nums; ++j) {
    static_cast<float*>(array->data)[j] = 1.0;
  }
}

void TestNDArray(const NDArray& array) { TestNDArray(const_cast<DLTensor*>(array.operator->())); }

void TestPointerNDArray(const NDArray& ptr_array, int64_t total_nums, int64_t batch_size) {
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < total_nums; ++j) {
      static_cast<int32_t**>(ptr_array->data)[i][j] = 999;
    }
  }
}

void TestPointerNDArray(const NDArray& ptr_array, const NDArray& sample, int64_t batch_size) {
  size_t total_nums = 1;
  for (auto d : sample.Shape()) {
    total_nums *= d;
  }
  TestPointerNDArray(ptr_array, total_nums, batch_size);
}

bool CheckEqualShape(const DLTensor& t1, const DLTensor& t2) {
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

void FillInPointers(void** host_raw_ptrs, size_t size, const std::vector<OpNode<DLTensor*>*>& nodes,
                    int arg_num, Allocator* allocator) {
  auto first_arg = nodes[0]->args_[arg_num];
  auto data_size = GetDataSize(*first_arg);
  if (first_arg->data != nullptr) {
#pragma GCC ivdep
    for (size_t j = 0; j < size; ++j) {
#ifdef DEBUG_CHECKS
      ICHECK(nodes[j]->args_[arg_num]->data != nullptr) << arg_num << " " << j;
      ICHECK(CheckEqualShape(*first_arg, *(nodes[j]->args_[arg_num])));
#endif
      host_raw_ptrs[j] = nodes[j]->args_[arg_num]->data;
    }
  } else {
    void* start = allocator->ArenaAlloc(size * data_size, 256, first_arg->dtype).data;
#pragma GCC ivdep
    for (size_t j = 0; j < size; ++j) {
      auto ptr = static_cast<char*>(start) + j * data_size;
      host_raw_ptrs[j] = ptr;
      nodes[j]->args_[arg_num]->data = ptr;
#ifdef DEBUG_CHECKS
      ICHECK(CheckEqualShape(*first_arg, *(nodes[j]->args_[arg_num])));
#endif
    }
  }
}

NDArray CreatePointerNDArray(const std::vector<OpNode<NDArray>*>& nodes, int arg_num) {
  size_t size = nodes.size();
  NDArray result = NDArray::Empty(ShapeTuple({static_cast<int64_t>(size)}),
                                  DLDataType{kDLOpaqueHandle, 8 * sizeof(void*), 1},
                                  nodes[0]->args_[arg_num]->device);

  void** raw_data = static_cast<void**>(result->data);
  for (size_t j = 0; j < size; ++j) {
    raw_data[j] = nodes[j]->args_[arg_num]->data;
  }

  // TestPointerNDArray(result, nodes[0]->args_[arg_num], size);

  return result;
}

bool CheckEqualShape(DLTensor& t1, DLTensor& t2) {
  if (t1.ndim != t2.ndim) {
    return false;
  }
  for (int i = 0; i < t2.ndim; ++i) {
    if (t1.shape[i] != t2.shape[i]) {
      return false;
    }
  }
  return true;
}

NDArray CreatePointerNDArray(const std::vector<OpNode<DLTensor*>*>& nodes, int arg_num,
                             Allocator* allocator) {
  int64_t size = nodes.size();
  auto& accelerator_device = nodes[0]->args_[arg_num]->device;
  NDArray result =
      NDArray::Empty(ShapeTuple({static_cast<int64_t>(size)}),
                     DLDataType{kDLOpaqueHandle, 8 * sizeof(void*), 1}, accelerator_device);
  if (accelerator_device.device_type == kDLCUDA) {
    void** raw_data = static_cast<void**>(Arena::Current()->allocate_<void*>(size));
    FillInPointers(raw_data, size, nodes, arg_num, allocator);
    result.CopyFromBytes(raw_data, size * sizeof(void*));
  } else {
    void** raw_data = static_cast<void**>(result->data);
    FillInPointers(raw_data, size, nodes, arg_num, allocator);
#ifdef DEBUG_CHECKS
    TestPointerNDArray(result, GetDataSize(*(nodes[0]->args_[arg_num])) / sizeof(float), size);
#endif
  }
  return result;
}

// void ArrayCopyFromBytesAsync(DLTensor* handle, const void* data, size_t nbytes) {
// #ifdef DEBUG_CHECKS
//   size_t arr_size = GetDataSize(*handle);
//   ICHECK_EQ(arr_size, nbytes) << "ArrayCopyFromBytes: size mismatch " << arr_size << " " <<
//   nbytes; ICHECK(IsContiguous(*handle)) << "ArrayCopyFromBytes only support contiguous array for
//   now";
// #endif

//   DLTensor from;
//   from.data = const_cast<void*>(data);
//   from.device = Device{kDLCPU, 0};
//   from.ndim = handle->ndim;
//   from.dtype = handle->dtype;
//   from.shape = handle->shape;
//   from.strides = nullptr;
//   from.byte_offset = 0;
//   // std::cout << "[COPY] " << handle->data << " " << handle->ndim << " " << handle->shape[0]
//   //           << std::endl;
//   // std::cout << "[COPY]  " << from.data << " " << from.ndim << " " << from.shape[0] <<
//   std::endl; DeviceAPI::Get(handle->device)->CopyDataFromTo(&from, handle, nullptr);
// #ifdef DEBUG_CHECKS
//   // Synchronize in case data become unavailable later.
//   DeviceAPI::Get(handle->device)->StreamSync(handle->device, nullptr);
// #endif
// }

DLTensor* CreatePointerDLTensor(const std::vector<OpNode<DLTensor*>*>& nodes, int arg_num,
                                Allocator* allocator) {
  int64_t size = nodes.size();
  auto& accelerator_device = nodes[0]->args_[arg_num]->device;

  DLTensor* result = Arena::Current()->allocate_<DLTensor>();
  {
    int64_t* shape_data = Arena::Current()->allocate_<int64_t>();
    shape_data[0] = size;
    auto dtype = DLDataType{kDLOpaqueHandle, 8 * sizeof(void*), 1};
    result->device = accelerator_device;
    result->data = allocator->ArenaAlloc(size * sizeof(void*), 256, dtype).data;
    result->strides = nullptr;
    result->ndim = 1;
    result->dtype = dtype;
    result->shape = shape_data;
    result->byte_offset = 0;
  }

  if (accelerator_device.device_type == kDLCUDA) {
    void** raw_data = static_cast<void**>(Arena::Current()->allocate_<void*>(size));
    FillInPointers(raw_data, size, nodes, arg_num, allocator);
    ArrayCopyFromBytesAsync(result->data, raw_data, size * sizeof(void*));
  } else {
    void** raw_data = static_cast<void**>(result->data);
    FillInPointers(raw_data, size, nodes, arg_num, allocator);
#ifdef DEBUG_CHECKS
    // TestPointerNDArray(result, GetDataSize(*(nodes[0]->args_[arg_num])) / sizeof(float), size);
#endif
  }
  return result;
}

NDArray CreateConcatenatedNDArray(std::vector<NDArray>& arrays) {
  auto unbatched_shape = arrays[0].Shape();
  std::vector<int64_t> batched_shape(unbatched_shape.size() + 1);
  batched_shape[0] = static_cast<int64_t>(arrays.size());
  for (size_t i = 0; i < unbatched_shape.size(); ++i) {
    batched_shape[i + 1] = unbatched_shape[i];
  }
  // std::cout << "[VMU]     concated size " << ShapeToString(batched_shape) << std::endl;
  auto result = NDArray::Empty(ShapeTuple(batched_shape), arrays[0].DataType(), arrays[0]->device);
  DLTensor* mutable_internal = const_cast<DLTensor*>(result.operator->());
  auto old_offset = mutable_internal->byte_offset;
  auto offset_delta = arrays[0].DataType().bytes();
  for (auto dim : unbatched_shape) {
    offset_delta *= dim;
  }
  mutable_internal->shape[0] = 1;
  for (size_t i = 0; i < arrays.size(); ++i) {
    arrays[i].CopyTo(result);
    mutable_internal->byte_offset += offset_delta;
  }

  mutable_internal->shape[0] = arrays.size();
  mutable_internal->byte_offset = old_offset;
  return result;
}

DLTensor* CreateConcatenatedDLTensor(const std::vector<OpNode<DLTensor*>*>& nodes, int arg_num,
                                     Allocator* allocator) {
  auto& first_arg = nodes[0]->args_[arg_num];
  auto ub_ndim = first_arg->ndim;
  int64_t size = nodes.size();
  auto& accelerator_device = first_arg->device;

  DLTensor* result = Arena::Current()->allocate_<DLTensor>();
  int64_t ub_flat_size = 1;
  auto& dtype = first_arg->dtype;
  auto dtype_bytes = (dtype.lanes * dtype.bits + 7) / 8;
  {
    int64_t* shape_data = Arena::Current()->allocate_<int64_t>(ub_ndim + 1);
    shape_data[0] = size;
    for (int i = 0; i < ub_ndim; ++i) {
      auto dim_ext = first_arg->shape[i];
      shape_data[i + 1] = dim_ext;
      ub_flat_size *= dim_ext;
    }
    int64_t b_flat_bytes = ub_flat_size * dtype_bytes * size;

    result->device = accelerator_device;
    result->data = allocator->ArenaAlloc(b_flat_bytes, 256, dtype).data;
    result->strides = nullptr;
    result->ndim = ub_ndim + 1;
    result->dtype = dtype;
    result->shape = shape_data;
    result->byte_offset = 0;
  }
  int64_t ub_flat_bytes = ub_flat_size * dtype_bytes;

  if (first_arg->data == nullptr) {
    void* start = result->data;
#pragma GCC ivdep
    for (int j = 0; j < size; ++j) {
      nodes[j]->args_[arg_num]->data = static_cast<char*>(start) + j * ub_flat_bytes;
    }
  } else {
    void** input_raw_ptrs = static_cast<void**>(Arena::Current()->allocate_<void*>(size));
#pragma GCC ivdep
    for (int j = 0; j < size; ++j) {
      input_raw_ptrs[j] = nodes[j]->args_[arg_num]->data;
    }

    // DLTensor* input_ptrs_device = Arena::Current()->allocate_<DLTensor>();
    // {
    //   int64_t* shape_data = Arena::Current()->allocate_<int64_t>();
    //   shape_data[0] = size;
    //   auto dtype = DLDataType{kDLOpaqueHandle, 8 * sizeof(void*), 1};
    //   input_ptrs_device->device = accelerator_device;
    //   input_ptrs_device->data = allocator->ArenaAlloc(size * sizeof(void*), 256, dtype).data;
    //   input_ptrs_device->strides = nullptr;
    //   input_ptrs_device->ndim = 1;
    //   input_ptrs_device->dtype = dtype;
    //   input_ptrs_device->shape = shape_data;
    //   input_ptrs_device->byte_offset = 0;
    // }

    void* input_ptrs_device = allocator->ArenaAlloc(size * sizeof(void*), 256, dtype).data;
    ArrayCopyFromBytesAsync(input_ptrs_device, input_raw_ptrs, sizeof(void*) * size);
    contrib::db_concat_copy_wrapper(static_cast<float**>(input_ptrs_device),
                                    static_cast<float*>(result->data), size, ub_flat_size);
  }

  return result;
}

/* Invoking packed functions */
template <typename TensorType>
void InvokePackedFnUnrolled(const size_t func_idx, const PackedFunc& func, TensorType* args,
                            int arity) {
#ifdef DB_PROFILING
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStartCall("arg_prep_unbatched");
  }
#endif

  std::vector<TVMValue> values(arity);
  std::vector<int> codes(arity);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  for (int i = 0; i < arity; i++) {
    setter(i, args[i]);
  }

#ifdef DB_PROFILING
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStopCall();
    VMDBProfiler::ProfileDeviceStartCall("Kernel_" + std::to_string(func_idx));
  }
#endif
  TVMRetValue rv;
  func.CallPacked(TVMArgs(values.data(), codes.data(), arity), &rv);
#ifdef DB_PROFILING
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileDeviceStopCall();
  }
#endif
}

template void InvokePackedFnUnrolled<NDArray>(const size_t func_idx, const PackedFunc& func,
                                              NDArray* args, int arity);

template void InvokePackedFnUnrolled<DLTensor*>(const size_t func_idx, const PackedFunc& func,
                                                DLTensor** args, int arity);

template <typename TensorType>
void InvokePackedFnBatchedUnrolled(const size_t func_idx, const PackedFunc& func, Index arity,
                                   const std::vector<DBBatchedArgMode>& arg_modes,
                                   const std::vector<OpNode<TensorType>*>& nodes) {
#ifdef DB_PROFILING
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStartCall("arg_prep_batched");
  }
#endif
  bool print = false;
  ICHECK_EQ(arity, arg_modes.size());
  int32_t batch_size = nodes.size();

  std::vector<TVMValue> values(arity + 1);
  std::vector<int> codes(arity + 1);
  static std::vector<NDArray> arg_holder;
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  setter(0, batch_size);
  if (print) {
    std::cout << "[VMU]    BatchSize 0 " << batch_size << std::endl;
  }
  int ctr = 1;
  for (Index i = 0; i < arity; ++i) {
    switch (arg_modes[i]) {
      case kIgnore: {
        if (print) {
          std::cout << "[VMU]    Ignoring " << i << std::endl;
        }
        break;
      }
      case kReuse: {
        setter(ctr, nodes[0]->args_[i]);
        if (print) {
          std::cout << "[VMU]    ArgReuse " << ctr << std::endl;
        }
        ctr += 1;
        break;
      }
      case kScatter: {
        arg_holder.push_back(CreatePointerNDArray(nodes, i));
        setter(ctr, arg_holder.back());
        if (print) {
          std::cout << "[VMU]    ArgScatter " << ctr << " "
                    << GetDLTensorInfo(arg_holder.back().operator->()) << std::endl;
        }
        ctr += 1;
        break;
      }
      case kConcat: {
        std::vector<NDArray> to_concat(batch_size);
        for (size_t j = 0; j < static_cast<size_t>(batch_size); ++j) {
          to_concat[j] = nodes[j]->args_[i];
        }

        if (print) {
          std::cout << "[VMU]    Concating " << to_concat.size() << " arays." << std::endl;
        }
        NDArray concat_array = CreateConcatenatedNDArray(to_concat);
        arg_holder.push_back(concat_array);
        setter(ctr, concat_array);
        if (print) {
          std::cout << "[VMU]    ArgConcat " << ctr << " " << ShapeToString(concat_array.Shape())
                    << std::endl;
        }
        ctr += 1;

        // ICHECK(false) << "Concat not implemented yet!";
        // break;
      }
    }
  }

  // std::cout << "[VMU]    Calling " << ctr << " " << arity << std::endl;
#ifdef DB_PROFILING
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStopCall();
    VMDBProfiler::ProfileDeviceStartCall("Kernel_" + std::to_string(func_idx));
  }
#endif
  TVMRetValue rv;
  func.CallPacked(TVMArgs(values.data(), codes.data(), ctr), &rv);
#ifdef DB_PROFILING
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileDeviceStopCall();
  }
#endif
}

template void InvokePackedFnBatchedUnrolled<NDArray>(const size_t func_idx, const PackedFunc& func,
                                                     Index arity,
                                                     const std::vector<DBBatchedArgMode>& arg_modes,
                                                     const std::vector<OpNode<NDArray>*>& nodes);

// template void InvokePackedFnBatchedUnrolled<DLTensor*>(
//     const size_t func_idx, const PackedFunc& func, Index arity,
//     const std::vector<DBBatchedArgMode>& arg_modes, const std::vector<OpNode<DLTensor*>*>&
//     nodes);

void InvokePackedFn(const PackedFunc& func, Index arg_count, Index output_size,
                    const ObjectRef* args, int64_t num_args,
                    const std::vector<DBBatchedArgMode>& arg_modes, bool batched,
                    bool scattered_kernels) {
  size_t arity = 0;
  for (Index i = 0; i < arg_count; i++) {
    if (const auto* obj = args[i].as<ADTObj>()) {
      arity += obj->size;
    } else {
      ++arity;
    }
  }

  std::vector<TVMValue> values(arity);
  std::vector<int> codes(arity);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  int idx = 0;
  bool is_empty_output = false;
  for (Index i = 0; i < arg_count; i++) {
    if (const auto* dt_cell = args[i].as<ADTObj>()) {
      for (size_t fi = 0; fi < dt_cell->size; ++fi) {
        auto obj = (*dt_cell)[fi];
        auto nd_array = Downcast<NDArray>(obj);
        setter(idx++, nd_array);
      }
    } else {
      auto nd_array = Downcast<NDArray>(args[i]);
      // We can safely skip CallPacked if there is only one
      // output and it is empty.
      if (i == arg_count - 1 && output_size == 1) {
        for (const auto& dim : nd_array.Shape()) {
          if (!dim) {
            is_empty_output = true;
            break;
          }
        }
      }
      setter(idx++, nd_array);
    }
  }

  if (!is_empty_output) {
    TVMRetValue rv;
    func.CallPacked(TVMArgs(values.data(), codes.data(), arity), &rv);
  }
}
}  // namespace vm
}  // namespace runtime
}  // namespace tvm
