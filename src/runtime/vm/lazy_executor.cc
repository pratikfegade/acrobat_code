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
void LazyExecutor::AddPackedCall(const Index func_idx, const Index arg_count,
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

  // std::cout << "Op " << nodes_.size() << std::endl;
  // size_t i = 0;
  // for (; i < unrolled_input_size; ++i) {
  // std::cout << " I " << args_copy[i].get() << std::endl;
  // }

  // for (; i < unrolled_input_size + unrolled_output_size; ++i) {
  // std::cout << " O " << args_copy[i].get() << std::endl;
  // }

  OpNode node(nodes_.size(), func_idx, unrolled_input_size + unrolled_output_size,
              unrolled_output_size, args_copy);
  nodes_.push_back(node);
}

void LazyExecutor::Execute() {
  for (OpNode& node : nodes_) {
    InvokePackedFnUnrolled(vm_->shared_state_->packed_funcs_[node.func_idx_], node.arg_count_,
                           node.output_size_, node.args_);
  }
  nodes_.clear();
}

void LazyExecutor::BatchedExecute() {
  std::unordered_map<NDArray, int, ObjectPtrHash, ObjectPtrEqual> output_tensor_to_node;
  for (OpNode& node : nodes_) {
    for (size_t i = node.OutputStart(); i < node.OutputEnd(); ++i) {
      output_tensor_to_node[node.args_[i]] = node.id_;
    }
  }

  size_t num_nodes = nodes_.size();
  std::vector<int> node_to_depth(num_nodes);
  std::vector<std::vector<int>> depth_to_node(num_nodes);
  int graph_depth = -1;
  for (size_t i = 0; i < num_nodes; ++i) {
    OpNode& node = nodes_[i];
    int max_depth = 0;
    for (size_t j = node.InputStart(); j < node.InputEnd(); ++j) {
      auto it = output_tensor_to_node.find(node.args_[j]);
      if (it != output_tensor_to_node.end()) {
        auto input_node_id = it->second;
        max_depth = std::max(max_depth, node_to_depth[input_node_id]);
      }
    }
    int node_depth = max_depth + 1;
    node_to_depth[i] = node_depth;
    depth_to_node[node_depth].push_back(i);
    graph_depth = std::max(graph_depth, node_depth);
  }

  for (int i = 0; i <= graph_depth; ++i) {
    auto& depth_nodes = depth_to_node[i];
    std::unordered_map<int, std::vector<OpNode*>> func_to_node;
    for (auto& node_id : depth_nodes) {
      func_to_node[nodes_[node_id].func_idx_].push_back(&nodes_[node_id]);
    }

    for (auto& pair : func_to_node) {
      auto& func_idx = pair.first;
      auto& nodes = pair.second;

      for (size_t i = 0; i < nodes.size(); ++i) {
        ICHECK_EQ(func_idx, nodes[i]->func_idx_);
        ICHECK_EQ(nodes[0]->arg_count_, nodes[i]->arg_count_);
        ICHECK_EQ(nodes[0]->output_size_, nodes[i]->output_size_);
        ICHECK_EQ(nodes[0]->args_.size(), nodes[i]->args_.size());
      }

      // if (nodes.size() == 1) {
      //   // std::cout << "[VMU] Executing " << func_idx << " " << nodes.size() << std::endl;
      //   InvokePackedFnUnrolled(vm_->shared_state_->packed_funcs_[func_idx], nodes[0]->arg_count_,
      //                          nodes[0]->output_size_, nodes[0]->args_);
      // } else {
      auto batched_func_idx = vm_->shared_state_->batched_funcs_[func_idx];
      // std::cout << "[VMU] Executing " << batched_func_idx << " " << nodes.size() << std::endl;
      // for (auto i : vm_->shared_state_->batched_func_arg_mode_[batched_func_idx]) {
      //   std::cout << "[VMU]   ArgMode " << i << std::endl;
      // }
      InvokePackedFnBatchedUnrolled(vm_->shared_state_->packed_funcs_[batched_func_idx],
                                    nodes[0]->arg_count_, nodes[0]->output_size_,
                                    vm_->shared_state_->batched_func_arg_mode_[batched_func_idx],
                                    nodes);
      // }
    }
  }
  nodes_.clear();
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
