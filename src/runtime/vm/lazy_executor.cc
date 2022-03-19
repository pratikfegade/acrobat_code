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
#include <tvm/runtime/vm/vm_profiling.h>

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
VMDBProfiler* VMDBProfiler::instance_{nullptr};

void LazyExecutor::AddPackedCall(const Index func_idx, const Index arg_count,
                                 const Index output_size, const ObjectRef* args, int num_args) {
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

  if (!is_empty_output) {
    OpNode node(nodes_.size(), func_idx, unrolled_input_size + unrolled_output_size,
                unrolled_output_size, args_copy);
    nodes_.push_back(node);
  }
}

void LazyExecutor::AddPackedCallUnrolled(const Index func_idx, const Index arg_count,
                                         const Index output_size, const NDArray* args,
                                         int num_args) {
  nodes_.emplace_back(nodes_.size(), func_idx, num_args, output_size, args, num_args);
}

void LazyExecutor::Execute() {
  for (OpNode& node : nodes_) {
    InvokePackedFnUnrolled(node.func_idx_, vm_shared_state_->packed_funcs_[node.func_idx_],
                           node.output_size_, node.args_.data(), node.args_.size());
  }
  nodes_.clear();
}

void LazyExecutor::ExecuteOpNodeBatch(
    const std::unordered_map<int, std::vector<OpNode*>>& func_to_node) {
  for (auto& pair : func_to_node) {
    auto& func_idx = pair.first;
    auto& func_nodes = pair.second;

    // for (size_t i = 0; i < func_nodes.size(); ++i) {
    //   ICHECK_EQ(func_idx, func_nodes[i]->func_idx_);
    //   ICHECK_EQ(func_nodes[0]->arg_count_, func_nodes[i]->arg_count_);
    //   ICHECK_EQ(func_nodes[0]->output_size_, func_nodes[i]->output_size_);
    //   ICHECK_EQ(func_nodes[0]->args_.size(), func_nodes[i]->args_.size());
    // }

    if (VMDBProfiler::DoProfile()) {
      VMDBProfiler::ProfileHostStopCall();
    }
    // std::cout << "[LE]  Executing " << func_idx << " " << func_nodes.size() << std::endl;
    if (func_nodes.size() == 1) {
      InvokePackedFnUnrolled(func_idx, vm_shared_state_->packed_funcs_[func_idx],
                             func_nodes[0]->output_size_, func_nodes[0]->args_.data(),
                             func_nodes[0]->args_.size());
    } else {
      auto batched_func_idx = vm_shared_state_->batched_funcs_[func_idx];
      InvokePackedFnBatchedUnrolled(
          batched_func_idx, vm_shared_state_->packed_funcs_[batched_func_idx],
          func_nodes[0]->arg_count_, func_nodes[0]->output_size_,
          vm_shared_state_->batched_func_arg_mode_[batched_func_idx], func_nodes);
    }
    if (VMDBProfiler::DoProfile()) {
      VMDBProfiler::ProfileHostStartCall("batched_execution");
    }
  }
}

void LazyExecutor::BatchedExecute(bool coarsened_execution, bool all_nodes_same_depth) {
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStartCall("batched_execution");
  }
  if (all_nodes_same_depth) {
    std::unordered_map<int, std::vector<OpNode*>> func_to_node;
    for (auto& node : nodes_) {
      func_to_node[node.func_idx_].push_back(&node);
    }
    ExecuteOpNodeBatch(func_to_node);
  } else {
    std::unordered_map<const Object*, int> output_tensor_to_node;
    size_t num_nodes = nodes_.size();
    output_tensor_to_node.reserve(num_nodes * 2);
    int graph_depth = -1;
    std::vector<std::vector<OpNode*>> depth_to_node(num_nodes);
    std::vector<int> node_to_depth(num_nodes, -1);

    if (coarsened_execution) {
      for (size_t i = 0; i < num_nodes; ++i) {
        OpNode& node = nodes_[i];
        auto& access_modes = vm_shared_state_->prim_func_arg_access_mode_[node.func_idx_];
        int max_depth = 0;

        for (size_t i = 0; i < node.args_.size(); ++i) {
          if (access_modes[i] == kInput || access_modes[i] == kInputOutput) {
            auto it = output_tensor_to_node.find(node.args_[i].get());
            if (it != output_tensor_to_node.end()) {
              auto input_node_id = it->second;
              ICHECK(node_to_depth[input_node_id] >= 0);
              max_depth = std::max(max_depth, node_to_depth[input_node_id]);
            }
          }
          if (access_modes[i] == kOutput || access_modes[i] == kInputOutput) {
            output_tensor_to_node[node.args_[i].get()] = node.id_;
          }
        }

        int node_depth = max_depth + 1;
        node_to_depth[i] = node_depth;
        depth_to_node[node_depth].push_back(&node);
        graph_depth = std::max(graph_depth, node_depth);
      }
    } else {
      for (OpNode& node : nodes_) {
        for (Index i = node.OutputStart(); i < node.OutputEnd(); ++i) {
          output_tensor_to_node[node.args_[i].get()] = node.id_;
        }
      }

      for (size_t i = 0; i < num_nodes; ++i) {
        OpNode& node = nodes_[i];
        int max_depth = 0;
        for (Index j = node.InputStart(); j < node.InputEnd(); ++j) {
          auto it = output_tensor_to_node.find(node.args_[j].get());
          if (it != output_tensor_to_node.end()) {
            auto input_node_id = it->second;
            // ICHECK(node_to_depth[input_node_id] >= 0);
            max_depth = std::max(max_depth, node_to_depth[input_node_id]);
          }
        }
        int node_depth = max_depth + 1;
        node_to_depth[i] = node_depth;
        depth_to_node[node_depth].push_back(&node);
        graph_depth = std::max(graph_depth, node_depth);
      }
    }

    std::unordered_map<int, std::vector<OpNode*>> func_to_node;
    for (int i = 0; i <= graph_depth; ++i) {
      auto& depth_nodes = depth_to_node[i];
      for (auto& node : depth_nodes) {
        func_to_node[node->func_idx_].push_back(node);
      }
      ExecuteOpNodeBatch(func_to_node);
      func_to_node.clear();
    }
  }
  nodes_.clear();
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStopCall();
  }
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
