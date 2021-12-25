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
#include "vm_utils.h"

using namespace tvm::runtime;

namespace tvm {
namespace runtime {
namespace vm {
void OpNode::Execute() { InvokePackedFnUnrolled(func_, arg_count_, output_size_, args_); }

void LazyExecutor::AddPackedCall(const PackedFunc& func, const Index arg_count,
                                 const Index output_size, const std::vector<ObjectRef> args) {
  std::vector<NDArray> args_copy;
  bool is_empty_output = false;

  Index rolled_output_size = output_size;
  Index rolled_input_size = arg_count - rolled_output_size;
  Index unrolled_input_size = 0;
  Index unrolled_output_size = 0;
  for (Index i = 0; i < arg_count; i++) {
    if (const auto* dt_cell = args[i].as<ADTObj>()) {
      for (size_t fi = 0; fi < dt_cell->size; ++fi) {
        auto obj = (*dt_cell)[fi];
        auto nd_array = Downcast<NDArray>(obj);
        args_copy.push_back(nd_array);
        if (i < rolled_input_size) {
          unrolled_input_size++;
        } else {
          unrolled_output_size++;
        }
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
      args_copy.push_back(nd_array);
      if (i < rolled_input_size) {
        unrolled_input_size++;
      } else {
        unrolled_output_size++;
      }
    }
  }

  if (is_empty_output) {
    return;
  }

  std::cout << "Op " << nodes_.size() << std::endl;
  size_t i = 0;
  for (; i < unrolled_input_size; ++i) {
    std::cout << " I " << args_copy[i].get() << std::endl;
  }

  for (; i < unrolled_input_size + unrolled_output_size; ++i) {
    std::cout << " O " << args_copy[i].get() << std::endl;
  }

  OpNode node(nodes_.size(), func, unrolled_input_size + unrolled_output_size, unrolled_output_size,
              args_copy);
  nodes_.push_back(node);
}

void LazyExecutor::Execute() {
  std::cout << "Executing nodes" << std::endl;
  for (OpNode& node : nodes_) {
    node.Execute();
  }
  nodes_.clear();
}

void LazyExecutor::BatchedExecute() {
  std::cout << "Batch Executing" << std::endl;
  std::unordered_map<NDArray, int, ObjectPtrHash, ObjectPtrEqual> output_tensor_to_node;
  for (OpNode& node : nodes_) {
    for (size_t i = node.OutputStart(); i < node.OutputEnd(); ++i) {
      output_tensor_to_node[node.args_[i]] = node.id_;
    }
  }

  size_t num_nodes = nodes_.size();
  std::vector<int> node_to_depth(num_nodes);
  for (size_t i = 0; i < num_nodes; ++i) {
    OpNode& node = nodes_[i];
    int max_depth = 0;
    std::cout << " Node " << i << std::endl;
    for (size_t j = node.InputStart(); j < node.InputEnd(); ++j) {
      std::cout << "   Tensor " << node.args_[j].get() << std::endl;
      auto it = output_tensor_to_node.find(node.args_[j]);
      if (it != output_tensor_to_node.end()) {
        auto input_node_id = it->second;
        max_depth = std::max(max_depth, node_to_depth[input_node_id]);
      }
    }
    node_to_depth[i] = max_depth + 1;
    std::cout << "  Depth " << max_depth << std::endl;
  }

  std::cout << " Executing nodes" << std::endl;
  for (OpNode& node : nodes_) {
    node.Execute();
  }
  nodes_.clear();
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
