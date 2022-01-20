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
  for (auto d : array.Shape()) {
    total_nums *= d;
  }

  std::cout << "[VMU] Array size " << total_nums << std::endl;

  for (size_t j = 0; j < total_nums; ++j) {
    static_cast<float*>(array->data)[j] = 1.0;
  }
}

NDArray CreatePointerNDArray(std::vector<NDArray>& arrays) {
  int size = arrays.size();
  std::vector<void*> raw_ptrs(size);
  for (size_t i = 0; i < size; ++i) {
    raw_ptrs[i] = arrays[i]->data;
  }
  auto result = NDArray::Empty(
      ShapeTuple({size}), DLDataType{kDLOpaqueHandle, 8 * sizeof(void*), 1}, arrays[0]->device);
  result.CopyFromBytes(raw_ptrs.data(), size * sizeof(void*));

  {
    size_t total_nums = 1;
    for (auto d : arrays[0].Shape()) {
      total_nums *= d;
    }

    std::cout << "[VMU] Array size " << total_nums << std::endl;

    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < total_nums; ++j) {
        static_cast<float**>(result->data)[i][j] = 1.0;
      }
    }
  }

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
  ICHECK_EQ(arity, arg_modes.size());
  int32_t batch_size = nodes.size();

  std::vector<TVMValue> values(arity + 1);
  std::vector<int> codes(arity + 1);
  std::vector<NDArray> arg_holder(arity);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  // std::cout << "Executing " << arity << std::endl;
  setter(0, batch_size);
  for (Index i = 0; i < arity; i++) {
    switch (arg_modes[i]) {
      case kIgnore: {
        break;
      }
      case kReuse: {
        arg_holder[i] = nodes[0]->args_[i];
        setter(i + 1, nodes[0]->args_[i]);
        std::cout << "[VMU]  ArgReuse " << ShapeToString(nodes[0]->args_[i].Shape()) << std::endl;
        break;
      }
      case kScatter: {
        std::vector<NDArray> to_scatter(batch_size);
        for (size_t j = 0; j < batch_size; ++j) {
          to_scatter[j] = nodes[j]->args_[i];
        }
        auto ptr_array = CreatePointerNDArray(to_scatter);
        arg_holder[i] = ptr_array;
        setter(i + 1, ptr_array);
        std::cout << "[VMU]  ArgScatter " << ShapeToString(ptr_array.Shape()) << std::endl;
        break;
      }
      case kConcat: {
        ICHECK(false) << "Concat not implemented yet!";
        break;
      }
    }
  }

  TVMRetValue rv;
  func.CallPacked(TVMArgs(values.data(), codes.data(), arity + 1), &rv);
}

// void InvokePackedFn(const PackedFunc& func, Index arg_count, Index output_size,
//                     const std::vector<ObjectRef>& args,
//                     const std::vector<DBBatchedArgMode>& arg_modes, bool batched,
//                     bool scattered_kernels) {
//   size_t arity = 0;
//   if (batched) {
//     arity++;
//   }
//   for (Index i = 0; i < arg_count; i++) {
//     if (const auto* obj = args[i].as<ADTObj>()) {
//       arity += obj->size;
//     } else {
//       ++arity;
//     }
//   }

//   if (batched) {
//     ICHECK_EQ(arg_modes.size() + 1, arity) << arg_modes.size() << " " << arity << " " <<
//     arg_count;
//   }

//   if (batched) {
//     arity = 2 * arity - 1;
//   }
//   std::vector<ObjectRef> new_args;
//   new_args.reserve(arity);
//   std::vector<TVMValue> values(arity);
//   std::vector<int> codes(arity);
//   runtime::TVMArgsSetter setter(values.data(), codes.data());
//   bool is_empty_output = false;
//   int32_t batch_size = 1;
//   int32_t counter = 0;
//   if (batched) {
//     setter(counter++, batch_size);
//   }
//   int32_t idx = 0;
//   for (Index i = 0; i < arg_count; i++) {
//     if (const auto* dt_cell = args[i].as<ADTObj>()) {
//       for (size_t fi = 0; fi < dt_cell->size; ++fi) {
//         if (batched && arg_modes.size() > 0 && arg_modes[idx++] == kIgnore) {
//           continue;
//         }

//         auto obj = (*dt_cell)[fi];
//         auto nd_array = Downcast<NDArray>(obj);

//         if (batched) {
//           auto old_shape = nd_array.Shape();
//           std::vector<Index> new_shape_vec;
//           new_shape_vec.push_back(batch_size);
//           for (auto dim : old_shape) {
//             new_shape_vec.push_back(dim);
//           }
//           nd_array = nd_array.CreateView(ShapeTuple(new_shape_vec), nd_array.DataType());
//         }
//         new_args.push_back(nd_array);
//         setter(counter++, nd_array);
//       }
//     } else {
//       auto nd_array = Downcast<NDArray>(args[i]);
//       // We can safely skip CallPacked if there is only one
//       // output and it is empty.
//       if (i == arg_count - 1 && output_size == 1) {
//         for (const auto& dim : nd_array.Shape()) {
//           if (!dim) {
//             is_empty_output = true;
//             break;
//           }
//         }
//       }

//       if (batched && arg_modes.size() > 0 && arg_modes[idx++] == kIgnore) {
//         continue;
//       }

//       if (batched) {
//         auto old_shape = nd_array.Shape();
//         std::vector<Index> new_shape_vec;
//         new_shape_vec.push_back(batch_size);
//         for (auto dim : old_shape) {
//           new_shape_vec.push_back(dim);
//         }
//         nd_array = nd_array.CreateView(ShapeTuple(new_shape_vec), nd_array.DataType());
//       }
//       new_args.push_back(nd_array);
//       setter(counter++, nd_array);
//     }
//   }

//   // if (scattered_kernels) {
//   //   int total_args = new_args.size();
//   //   for (auto i = 0; i < total_args; ++i) {
//   //     auto nd_array = Downcast<NDArray>(new_args[i]);
//   //     auto ptr_nd_array = CreatePointerNDArray({nd_array});
//   //     setter(counter++, ptr_nd_array);
//   //     new_args.push_back(ptr_nd_array);
//   //   }
//   // }

//   std::cout << "[VMU]   Invoke " << counter << " " << batched << std::endl;

//   if (!is_empty_output) {
//     TVMRetValue rv;
//     func.CallPacked(TVMArgs(values.data(), codes.data(), counter), &rv);
//   }
// }

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
    std::cout << "YELLO " << std::endl;
    TVMRetValue rv;
    func.CallPacked(TVMArgs(values.data(), codes.data(), arity), &rv);
  }
}
}  // namespace vm
}  // namespace runtime
}  // namespace tvm
