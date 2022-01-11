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

NDArray CreatePointerNDArray(std::vector<NDArray> arrays) {
  int size = arrays.size();
  std::vector<void*> raw_ptrs(size);
  for (size_t i = 0; i < size; ++i) {
    raw_ptrs[i] = arrays[i]->data;
  }
  auto result =
      NDArray::Empty(ShapeTuple({size}), DLDataType{kDLOpaqueHandle, 64, 1}, arrays[0]->device);
  result.CopyFromBytes(raw_ptrs.data(), size * sizeof(void*));
  return result;
}

void InvokePackedFnUnrolled(const PackedFunc& func, Index arg_count, Index output_size,
                            const std::vector<NDArray>& args) {
  size_t arity = arg_count;

  std::vector<TVMValue> values(arity);
  std::vector<int> codes(arity);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  // std::cout << "Executing " << arity << std::endl;
  for (Index i = 0; i < arity; i++) {
    setter(i, args[i]);
  }

  TVMRetValue rv;
  func.CallPacked(TVMArgs(values.data(), codes.data(), arity), &rv);
}

void InvokePackedFn(const PackedFunc& func, Index arg_count, Index output_size,
                    const std::vector<ObjectRef>& args,
                    const std::vector<DBBatchedArgMode>& arg_modes, bool batched) {
  size_t arity = 0;
  if (batched) {
    arity++;
  }
  for (Index i = 0; i < arg_count; i++) {
    if (const auto* obj = args[i].as<ADTObj>()) {
      arity += obj->size;
    } else {
      ++arity;
    }
  }

  if (batched) {
    ICHECK_EQ(arg_modes.size() + 1, arity) << arg_modes.size() << " " << arity << " " << arg_count;
  }

  if (batched) {
    arity = 2 * arity - 1;
  }
  std::vector<ObjectRef> new_args;
  new_args.reserve(arity);
  std::vector<TVMValue> values(arity);
  std::vector<int> codes(arity);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  bool is_empty_output = false;
  int32_t batch_size = 1;
  int32_t counter = 0;
  if (batched) {
    setter(counter++, batch_size);
  }
  int32_t idx = 0;
  for (Index i = 0; i < arg_count; i++) {
    if (const auto* dt_cell = args[i].as<ADTObj>()) {
      for (size_t fi = 0; fi < dt_cell->size; ++fi) {
        if (batched && arg_modes.size() > 0 && arg_modes[idx++] == kIgnore) {
          continue;
        }

        auto obj = (*dt_cell)[fi];
        auto nd_array = Downcast<NDArray>(obj);

        if (batched) {
          auto old_shape = nd_array.Shape();
          std::vector<Index> new_shape_vec;
          new_shape_vec.push_back(batch_size);
          for (auto dim : old_shape) {
            new_shape_vec.push_back(dim);
          }
          nd_array = nd_array.CreateView(ShapeTuple(new_shape_vec), nd_array.DataType());
        }
        new_args.push_back(nd_array);
        setter(counter++, nd_array);
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

      if (batched && arg_modes.size() > 0 && arg_modes[idx++] == kIgnore) {
        continue;
      }

      if (batched) {
        auto old_shape = nd_array.Shape();
        std::vector<Index> new_shape_vec;
        new_shape_vec.push_back(batch_size);
        for (auto dim : old_shape) {
          new_shape_vec.push_back(dim);
        }
        nd_array = nd_array.CreateView(ShapeTuple(new_shape_vec), nd_array.DataType());
      }
      new_args.push_back(nd_array);
      setter(counter++, nd_array);
    }
  }

  if (batched) {
    int total_args = new_args.size();
    for (auto i = 0; i < total_args; ++i) {
      auto nd_array = Downcast<NDArray>(new_args[i]);
      auto ptr_nd_array = CreatePointerNDArray({nd_array});
      setter(counter++, ptr_nd_array);
      new_args.push_back(ptr_nd_array);
    }
  }

  std::cout << "[VMU]   Invoke " << counter << std::endl;

  if (!is_empty_output) {
    TVMRetValue rv;
    func.CallPacked(TVMArgs(values.data(), codes.data(), counter), &rv);
  }
}
}  // namespace vm
}  // namespace runtime
}  // namespace tvm
