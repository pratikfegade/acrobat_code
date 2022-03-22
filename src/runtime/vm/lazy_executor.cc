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

template <>
void EagerAllocationLazyExecutor::AddPackedCall(const Index func_idx, const Index arg_count,
                                                const Index output_size, const ObjectRef* args,
                                                int num_args) {
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
    nodes_.emplace_back(nodes_.size(), func_idx, unrolled_input_size + unrolled_output_size,
                        unrolled_output_size, args_copy);
  }
}

template <>
void LazyAllocationLazyExecutor::AddPackedCall(const Index func_idx, const Index arg_count,
                                               const Index output_size, const ObjectRef* args,
                                               int num_args) {
  ICHECK(false) << "Un-unrolled packed calls not allowed in the lazy allocation executor";
}

template <>
void EagerAllocationLazyExecutor::AddPackedCallUnrolled(const Index func_idx, const Index arg_count,
                                                        const Index output_size, NDArray* args,
                                                        int num_args) {
  nodes_.emplace_back(nodes_.size(), func_idx, num_args, output_size, args, num_args);
}

template <>
void LazyAllocationLazyExecutor::AddPackedCallUnrolled(const Index func_idx, const Index arg_count,
                                                       const Index output_size, DLTensor** args,
                                                       int num_args) {
  nodes_.emplace_back(nodes_.size(), func_idx, num_args, output_size, args, num_args);
}

template <>
void EagerAllocationLazyExecutor::Execute() {
  for (EagerOpNode& node : nodes_) {
    InvokePackedFnUnrolled(node.func_idx_, vm_shared_state_->packed_funcs_[node.func_idx_],
                           node.output_size_, node.args_.data(), node.args_.size());
  }
  nodes_.clear();
}

template <>
void LazyAllocationLazyExecutor::Execute() {
  // TODO
  nodes_.clear();
}

template <>
void EagerAllocationLazyExecutor::ExecuteOpNodeBatch(
    const std::unordered_map<int, std::vector<EagerOpNode*>>& func_to_node) {
  for (auto& pair : func_to_node) {
    auto& func_idx = pair.first;
    auto& func_nodes = pair.second;

    if (VMDBProfiler::DoProfile()) {
      VMDBProfiler::ProfileHostStopCall();
    }
    // std::cout << "[LE]  Executing " << func_idx << " " << func_nodes.size() << std::endl;
    // if (func_nodes.size() == 1) {
    //   InvokePackedFnUnrolled(func_idx, vm_shared_state_->packed_funcs_[func_idx],
    //                          func_nodes[0]->output_size_, func_nodes[0]->args_.data(),
    //                          func_nodes[0]->args_.size());
    // } else {
    //   auto batched_func_idx = vm_shared_state_->batched_funcs_[func_idx];
    //   InvokePackedFnBatchedUnrolled(
    //       batched_func_idx, vm_shared_state_->packed_funcs_[batched_func_idx],
    //       func_nodes[0]->arg_count_, func_nodes[0]->output_size_,
    //       vm_shared_state_->batched_func_arg_mode_[batched_func_idx], func_nodes);
    // }
    if (VMDBProfiler::DoProfile()) {
      VMDBProfiler::ProfileHostStartCall("batched_execution");
    }
  }
}

template <>
void LazyAllocationLazyExecutor::ExecuteOpNodeBatch(
    const std::unordered_map<int, std::vector<LazyOpNode*>>& func_to_node) {
  for (auto& kv : func_to_node) {
    auto& func_idx = kv.first;
    auto& func_nodes = kv.second;
    auto& batched_func_idx = vm_shared_state_->batched_funcs_[func_idx];
    auto& arg_modes = vm_shared_state_->batched_func_arg_mode_[batched_func_idx];

    int32_t batch_size = func_nodes.size();
    auto arity = func_nodes[0]->arg_count_;
    std::vector<TVMValue> values(arity + 1);
    std::vector<int> codes(arity + 1);
    std::vector<NDArray> arg_holder(arity);
    runtime::TVMArgsSetter setter(values.data(), codes.data());
    setter(0, batch_size);
    int ctr = 1;

    // std::cout << "[LZ] Executing " << batched_func_idx << " " << arity << " "
    // << arg_modes.size() << std::endl;
    for (Index i = 0; i < arity; ++i) {
      switch (arg_modes[i]) {
        case kIgnore: {
          break;
        }
        case kReuse: {
          auto& arg = func_nodes[0]->args_[i];
          if (arg->data == nullptr) {
            auto ptr = vm_shared_state_->allocators_[0]
                           ->ArenaAlloc(GetDataSize(*arg), 256, arg->dtype)
                           .data;
            for (size_t j = 0; j < batch_size; ++j) {
              func_nodes[j]->args_[i]->data = ptr;
            }
          }
          setter(ctr, func_nodes[0]->args_[i]);
          ctr += 1;
          break;
        }
        case kScatter: {
          arg_holder[i] = CreatePointerNDArray(func_nodes, i, vm_shared_state_->allocators_[0]);
          setter(ctr, arg_holder[i]);
          ctr += 1;
          break;
        }
        case kConcat: {
          // std::vector<NDArray> to_concat(batch_size);
          // for (size_t j = 0; j < static_cast<size_t>(batch_size); ++j) {
          //   to_concat[j] = func_nodes[j]->args_[i];
          // }

          // NDArray concat_array = CreateConcatenatedNDArray(to_concat);
          // arg_holder[i] = concat_array;
          // setter(ctr, concat_array);
          // ctr += 1;
        }
      }
    }

    TVMRetValue rv;
    vm_shared_state_->packed_funcs_[batched_func_idx].CallPacked(
        TVMArgs(values.data(), codes.data(), ctr), &rv);
  }
}

template <typename T>
std::string PrintVector(std::vector<T> vector) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < vector.size(); ++i) {
    ss << " " << vector[i];
  }
  ss << "]";
  return ss.str();
}

template <>
void LazyAllocationLazyExecutor::BatchedExecute(bool coarsened_execution,
                                                bool all_nodes_same_depth) {
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStartCall("batched_execution");
  }
  if (all_nodes_same_depth) {
    std::unordered_map<int, std::vector<LazyOpNode*>> func_to_node;
    for (auto& node : nodes_) {
      func_to_node[node.func_idx_].push_back(&node);
    }
    ExecuteOpNodeBatch(func_to_node);
  } else {
    std::unordered_map<DLTensor*, int> output_tensor_to_node;
    size_t num_nodes = nodes_.size();
    output_tensor_to_node.reserve(num_nodes * 2);
    int graph_depth = -1;
    std::vector<std::vector<LazyOpNode*>> depth_to_node(num_nodes);
    std::vector<int> node_to_depth(num_nodes, -1);

    if (coarsened_execution) {
      for (size_t i = 0; i < num_nodes; ++i) {
        LazyOpNode& node = nodes_[i];
        auto& access_modes = vm_shared_state_->prim_func_arg_access_mode_[node.func_idx_];
        // std::cout << "[LZ] Access modes for " << node.func_idx_ << " " << access_modes.size()
        // << std::endl;
        int max_depth = -1;

        for (size_t j = 0; j < node.args_.size(); ++j) {
          auto& access_mode = access_modes[j];
          auto arg_ptr = node.args_[j];
          if (access_mode == kInput || access_mode == kInputOutput) {
            auto it = output_tensor_to_node.find(arg_ptr);
            if (it != output_tensor_to_node.end()) {
              auto input_node_id = it->second;
              auto input_node_depth = node_to_depth[input_node_id];
              max_depth = max_depth < input_node_depth ? input_node_depth : max_depth;
            }
          }
          if (access_mode == kOutput || access_mode == kInputOutput) {
            output_tensor_to_node[arg_ptr] = node.id_;
          }
        }

        int node_depth = max_depth + 1;
        node_to_depth[i] = node_depth;
        depth_to_node[node_depth].push_back(&node);
        graph_depth = std::max(graph_depth, node_depth);
      }
    } else {
      for (LazyOpNode& node : nodes_) {
        for (Index i = node.OutputStart(); i < node.OutputEnd(); ++i) {
          output_tensor_to_node[node.args_[i]] = node.id_;
        }
      }

      for (size_t i = 0; i < num_nodes; ++i) {
        LazyOpNode& node = nodes_[i];
        int max_depth = -1;
        for (Index j = node.InputStart(); j < node.InputEnd(); ++j) {
          auto it = output_tensor_to_node.find(node.args_[j]);
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

    std::vector<std::unordered_map<int, std::vector<LazyOpNode*>>> func_to_node_vecs;
    for (int j = 0; j <= graph_depth; ++j) {
      auto& depth_nodes = depth_to_node[j];
      std::unordered_map<int, std::vector<LazyOpNode*>> func_to_node;
      for (auto& node : depth_nodes) {
        func_to_node[node->func_idx_].push_back(node);
      }

      ExecuteOpNodeBatch(func_to_node);
    }
  }
  nodes_.clear();
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStopCall();
  }
}

template <>
void EagerAllocationLazyExecutor::BatchedExecute(bool coarsened_execution,
                                                 bool all_nodes_same_depth) {
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStartCall("batched_execution");
  }
  if (all_nodes_same_depth) {
    std::unordered_map<int, std::vector<EagerOpNode*>> func_to_node;
    for (auto& node : nodes_) {
      func_to_node[node.func_idx_].push_back(&node);
    }
    ExecuteOpNodeBatch(func_to_node);
  } else {
    std::unordered_map<const Object*, int> output_tensor_to_node;
    size_t num_nodes = nodes_.size();
    output_tensor_to_node.reserve(num_nodes * 2);
    int graph_depth = -1;
    std::vector<std::vector<EagerOpNode*>> depth_to_node(num_nodes);
    std::vector<int> node_to_depth(num_nodes, -1);

    if (coarsened_execution) {
      for (size_t i = 0; i < num_nodes; ++i) {
        EagerOpNode& node = nodes_[i];
        auto& access_modes = vm_shared_state_->prim_func_arg_access_mode_[node.func_idx_];
        int max_depth = 0;

        for (size_t j = 0; j < node.args_.size(); ++j) {
          auto& access_mode = access_modes[j];
          auto arg_ptr = node.args_[j].get();
          if (access_mode == kInput || access_mode == kInputOutput) {
            auto it = output_tensor_to_node.find(arg_ptr);
            if (it != output_tensor_to_node.end()) {
              auto input_node_id = it->second;
              // ICHECK(node_to_depth[input_node_id] >= 0);
              auto input_node_depth = node_to_depth[input_node_id];
              max_depth = max_depth < input_node_depth ? input_node_depth : max_depth;
            }
          }
          if (access_mode == kOutput || access_mode == kInputOutput) {
            output_tensor_to_node[arg_ptr] = node.id_;
          }
        }

        int node_depth = max_depth + 1;
        node_to_depth[i] = node_depth;
        depth_to_node[node_depth].push_back(&node);
        graph_depth = std::max(graph_depth, node_depth);
      }
    } else {
      for (EagerOpNode& node : nodes_) {
        for (Index i = node.OutputStart(); i < node.OutputEnd(); ++i) {
          output_tensor_to_node[node.args_[i].get()] = node.id_;
        }
      }

      for (size_t i = 0; i < num_nodes; ++i) {
        EagerOpNode& node = nodes_[i];
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

    std::unordered_map<int, std::vector<EagerOpNode*>> func_to_node;
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
