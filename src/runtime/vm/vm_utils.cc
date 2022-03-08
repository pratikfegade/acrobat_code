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

void TestNDArray(const NDArray& array) {
  size_t total_nums = 1;
  ICHECK(array.defined());
  ICHECK(array.Shape().defined());
  for (auto d : array.Shape()) {
    total_nums *= d;
  }

  std::cout << "[VMU] Array size " << total_nums << std::endl;

  for (size_t j = 0; j < total_nums; ++j) {
    static_cast<float*>(array->data)[j] = 1.0;
  }
}

NDArray CreatePointerNDArray(std::vector<NDArray>& arrays) {
  size_t size = arrays.size();
  std::vector<void*> raw_ptrs(size);
  for (size_t i = 0; i < size; ++i) {
    raw_ptrs[i] = arrays[i]->data;
  }
  auto result =
      NDArray::Empty(ShapeTuple({static_cast<int64_t>(size)}),
                     DLDataType{kDLOpaqueHandle, 8 * sizeof(void*), 1}, arrays[0]->device);
  result.CopyFromBytes(raw_ptrs.data(), size * sizeof(void*));

  // {
  //   size_t total_nums = 1;
  //   for (auto d : arrays[0].Shape()) {
  //     total_nums *= d;
  //   }

  //   std::cout << "[VMU] Array size " << total_nums << std::endl;

  //   for (size_t i = 0; i < size; ++i) {
  //     for (size_t j = 0; j < total_nums; ++j) {
  //       static_cast<float**>(result->data)[i][j] = 1.0;
  //     }
  //   }
  // }

  return result;
}

NDArray CreateConcatenatedNDArray(std::vector<NDArray>& arrays) {
  auto unbatched_shape = arrays[0].Shape();
  std::vector<int64_t> batched_shape(unbatched_shape.size() + 1);
  batched_shape[0] = static_cast<int64_t>(arrays.size());
  for (size_t i = 0; i < unbatched_shape.size(); ++i) {
    batched_shape[i + 1] = unbatched_shape[i];
  }
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
  mutable_internal->shape[0] = batched_shape.size();

  mutable_internal->byte_offset = old_offset;
  return result;
}

void InvokePackedFnUnrolled(const PackedFunc& func, Index arg_count, Index output_size,
                            const std::vector<NDArray>& args) {
  // std::cout << "[UMA] Executing" << std::endl;
  size_t arity = arg_count;

  std::vector<TVMValue> values(arity);
  std::vector<int> codes(arity);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  for (size_t i = 0; i < arity; i++) {
    setter(i, args[i]);
  }

  TVMRetValue rv;
  func.CallPacked(TVMArgs(values.data(), codes.data(), arity), &rv);
}

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

void InvokePackedFnBatchedUnrolled(const PackedFunc& func, Index arity, Index output_size,
                                   const std::vector<DBBatchedArgMode>& arg_modes,
                                   const std::vector<OpNode*>& nodes) {
  // std::cout << "[UMA] Executing" << std::endl;
  bool print = false;
  ICHECK_EQ(arity, arg_modes.size());
  int32_t batch_size = nodes.size();

  std::vector<TVMValue> values(arity + 1);
  std::vector<int> codes(arity + 1);
  std::vector<NDArray> arg_holder(arity);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  setter(0, batch_size);
  if (print) {
    std::cout << "[VMU]   BatchSize 0 " << batch_size << std::endl;
  }
  int ctr = 1;
  for (Index i = 0; i < arity; ++i) {
    switch (arg_modes[i]) {
      case kIgnore: {
        if (print) {
          std::cout << "[VMU]   Ignoring " << i << std::endl;
        }
        break;
      }
      case kReuse: {
        arg_holder[i] = nodes[0]->args_[i];
        setter(ctr, nodes[0]->args_[i]);
        if (print) {
          std::cout << "[VMU]   ArgReuse " << ctr << " "
                    << ShapeToString(nodes[0]->args_[i].Shape()) << std::endl;
        }
        ctr += 1;
        break;
      }
      case kScatter: {
        std::vector<NDArray> to_scatter(batch_size);
        for (size_t j = 0; j < static_cast<size_t>(batch_size); ++j) {
          to_scatter[j] = nodes[j]->args_[i];
        }
        auto ptr_array = CreatePointerNDArray(to_scatter);
        arg_holder[i] = ptr_array;
        setter(ctr, ptr_array);
        if (print) {
          std::cout << "[VMU]   ArgScatter " << ctr << " " << ShapeToString(ptr_array.Shape())
                    << std::endl;
        }
        ctr += 1;
        break;
      }
      case kConcat: {
        std::vector<NDArray> to_concat(batch_size);
        for (size_t j = 0; j < static_cast<size_t>(batch_size); ++j) {
          to_concat[j] = nodes[j]->args_[i];
        }

        auto concat_array = CreateConcatenatedNDArray(to_concat);
        arg_holder[i] = concat_array;
        setter(ctr, concat_array);
        if (print) {
          std::cout << "[VMU]   ArgConcat " << ctr << " " << ShapeToString(concat_array.Shape())
                    << std::endl;
        }
        ctr += 1;

        // ICHECK(false) << "Concat not implemented yet!";
        // break;
      }
    }
  }

  // std::cout << "[VMU] Calling " << ctr << " " << arity << std::endl;
  TVMRetValue rv;
  func.CallPacked(TVMArgs(values.data(), codes.data(), ctr), &rv);
}

void InvokePackedFn(const PackedFunc& func, Index arg_count, Index output_size,
                    const std::vector<ObjectRef>& args,
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
        // TestNDArray(nd_array);
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
      // TestNDArray(nd_array);
      setter(idx++, nd_array);
    }
  }

  if (!is_empty_output) {
    TVMRetValue rv;
    // std::cout << "[VMU] Calling " << arity << std::endl;
    func.CallPacked(TVMArgs(values.data(), codes.data(), arity), &rv);
  }
}
}  // namespace vm
}  // namespace runtime
}  // namespace tvm
