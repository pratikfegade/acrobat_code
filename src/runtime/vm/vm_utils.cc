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
#include <tvm/runtime/vm/vm.h>
#include <tvm/runtime/vm/vm_profiling.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../file_utils.h"

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
  for (size_t i = 0; i < array->ndim; ++i) {
    total_nums *= array->shape[i];
  }

  for (size_t j = 0; j < total_nums; ++j) {
    static_cast<float*>(array->data)[j] = 1.0;
  }
}

void TestNDArray(const NDArray& array) { TestNDArray(const_cast<DLTensor*>(array.operator->())); }

void TestPointerNDArray(const NDArray& ptr_array, const NDArray& sample, int64_t batch_size) {
  size_t total_nums = 1;
  for (auto d : sample.Shape()) {
    total_nums *= d;
  }
  for (size_t i = 0; i < batch_size; ++i) {
    for (size_t j = 0; j < total_nums; ++j) {
      static_cast<float**>(ptr_array->data)[i][j] = 1.0;
    }
  }
}

NDArray CreatePointerNDArray(const std::vector<OpNode<NDArray>*>& nodes, int arg_num) {
  size_t size = nodes.size();
  NDArray result = NDArray::Empty(ShapeTuple({static_cast<int64_t>(size)}),
                                  DLDataType{kDLOpaqueHandle, 8 * sizeof(void*), 1},
                                  nodes[0]->args_[arg_num]->device);
  void** raw_data = static_cast<void**>(result->data);

  constexpr size_t unroll_factor = 16;
  size_t preloop_bound = size / unroll_factor;

  // #pragma omp parallel for
  //   for (size_t jo = 0; jo < preloop_bound; ++jo) {
  // #pragma GCC unroll unroll_factor
  //     for (size_t ji = 0; ji < unroll_factor; ++ji) {
  //       size_t j = jo * unroll_factor + ji;
  //       raw_data[j] = nodes[j]->args_[arg_num]->data;
  //     }
  //   }

  for (size_t j = 0; j < size; ++j) {
    raw_data[j] = nodes[j]->args_[arg_num]->data;
  }

  // TestPointerNDArray(result, nodes[0]->args_[arg_num], size);

  return result;
}

NDArray CreatePointerNDArray(const std::vector<OpNode<DLTensor*>*>& nodes, int arg_num,
                             Allocator* allocator) {
  size_t size = nodes.size();
  NDArray result = NDArray::Empty(ShapeTuple({static_cast<int64_t>(size)}),
                                  DLDataType{kDLOpaqueHandle, 8 * sizeof(void*), 1},
                                  nodes[0]->args_[arg_num]->device);
  void** raw_data = static_cast<void**>(result->data);

  constexpr size_t unroll_factor = 16;
  size_t preloop_bound = size / unroll_factor;

  auto first_arg = nodes[0]->args_[arg_num];
  auto data_size = GetDataSize(*first_arg);
  if (first_arg->data != nullptr) {
    for (size_t j = 0; j < size; ++j) {
      // ICHECK(nodes[j]->args_[arg_num]->data != nullptr) << arg_num << " " << j;
      raw_data[j] = nodes[j]->args_[arg_num]->data;
    }
  } else {
    void* start = allocator->ArenaAlloc(size * data_size, 256, first_arg->dtype).data;
    for (size_t j = 0; j < size; ++j) {
      raw_data[j] = start + j * data_size;
      nodes[j]->args_[arg_num]->data = start + j * data_size;
    }
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

/* Invoking packed functions */
template <typename TensorType>
void InvokePackedFnUnrolled(const size_t func_idx, const PackedFunc& func, Index output_size,
                            TensorType* args, int arity) {
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStartCall("arg_prep_unbatched");
  }

  std::vector<TVMValue> values(arity);
  std::vector<int> codes(arity);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  for (size_t i = 0; i < arity; i++) {
    setter(i, args[i]);
  }

  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStopCall();
    VMDBProfiler::ProfileDeviceStartCall("Kernel_" + std::to_string(func_idx));
  }
  TVMRetValue rv;
  func.CallPacked(TVMArgs(values.data(), codes.data(), arity), &rv);
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileDeviceStopCall();
  }
}

template void InvokePackedFnUnrolled<NDArray>(const size_t func_idx, const PackedFunc& func,
                                              Index output_size, NDArray* args, int arity);

template void InvokePackedFnUnrolled<DLTensor*>(const size_t func_idx, const PackedFunc& func,
                                                Index output_size, DLTensor** args, int arity);

template <typename TensorType>
void InvokePackedFnBatchedUnrolled(const size_t func_idx, const PackedFunc& func, Index arity,
                                   Index output_size,
                                   const std::vector<DBBatchedArgMode>& arg_modes,
                                   const std::vector<OpNode<TensorType>*>& nodes) {
  // if (VMDBProfiler::DoProfile()) {
  //   VMDBProfiler::ProfileHostStartCall("arg_prep_batched");
  // }
  // // std::cout << "[UMA] Executing" << std::endl;
  // bool print = false;
  // ICHECK_EQ(arity, arg_modes.size());
  // int32_t batch_size = nodes.size();

  // std::vector<TVMValue> values(arity + 1);
  // std::vector<int> codes(arity + 1);
  // std::vector<NDArray> arg_holder(arity);
  // runtime::TVMArgsSetter setter(values.data(), codes.data());
  // setter(0, batch_size);
  // if (print) {
  //   std::cout << "[VMU]    BatchSize 0 " << batch_size << std::endl;
  // }
  // int ctr = 1;
  // for (Index i = 0; i < arity; ++i) {
  //   switch (arg_modes[i]) {
  //     case kIgnore: {
  //       if (print) {
  //         std::cout << "[VMU]    Ignoring " << i << std::endl;
  //       }
  //       break;
  //     }
  //     case kReuse: {
  //       // arg_holder[i] = nodes[0]->args_[i];
  //       setter(ctr, nodes[0]->args_[i]);
  //       if (print) {
  //         std::cout << "[VMU]    ArgReuse " << ctr << " "
  //                   << ShapeToString(nodes[0]->args_[i].Shape()) << std::endl;
  //       }
  //       ctr += 1;
  //       break;
  //     }
  //     case kScatter: {
  //       arg_holder[i] = CreatePointerNDArray(nodes, i);
  //       setter(ctr, arg_holder[i]);
  //       if (print) {
  //         std::cout << "[VMU]    ArgScatter " << ctr << " " <<
  //         ShapeToString(arg_holder[i].Shape())
  //                   << std::endl;
  //       }
  //       ctr += 1;
  //       break;
  //     }
  //     case kConcat: {
  //       std::vector<NDArray> to_concat(batch_size);
  //       for (size_t j = 0; j < static_cast<size_t>(batch_size); ++j) {
  //         to_concat[j] = nodes[j]->args_[i];
  //       }

  //       if (print) {
  //         std::cout << "[VMU]    Concating " << to_concat.size() << " arays." << std::endl;
  //       }
  //       NDArray concat_array = CreateConcatenatedNDArray(to_concat);
  //       arg_holder[i] = concat_array;
  //       setter(ctr, concat_array);
  //       if (print) {
  //         std::cout << "[VMU]    ArgConcat " << ctr << " " << ShapeToString(concat_array.Shape())
  //                   << std::endl;
  //       }
  //       ctr += 1;

  //       // ICHECK(false) << "Concat not implemented yet!";
  //       // break;
  //     }
  //   }
  // }

  // // std::cout << "[VMU]    Calling " << ctr << " " << arity << std::endl;
  // if (VMDBProfiler::DoProfile()) {
  //   VMDBProfiler::ProfileHostStopCall();
  //   VMDBProfiler::ProfileDeviceStartCall("Kernel_" + std::to_string(func_idx));
  // }
  // TVMRetValue rv;
  // func.CallPacked(TVMArgs(values.data(), codes.data(), ctr), &rv);
  // if (VMDBProfiler::DoProfile()) {
  //   VMDBProfiler::ProfileDeviceStopCall();
  // }
}

template void InvokePackedFnBatchedUnrolled<NDArray>(const size_t func_idx, const PackedFunc& func,
                                                     Index arity, Index output_size,
                                                     const std::vector<DBBatchedArgMode>& arg_modes,
                                                     const std::vector<OpNode<NDArray>*>& nodes);

template void InvokePackedFnBatchedUnrolled<DLTensor*>(
    const size_t func_idx, const PackedFunc& func, Index arity, Index output_size,
    const std::vector<DBBatchedArgMode>& arg_modes, const std::vector<OpNode<DLTensor*>*>& nodes);

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
