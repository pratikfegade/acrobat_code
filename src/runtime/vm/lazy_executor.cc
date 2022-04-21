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
#include <tvm/runtime/vm/arena.h>
#include <tvm/runtime/vm/dynamic_batching.h>
#include <tvm/runtime/vm/lazy_executor.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/runtime/vm/vm_profiling.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../../support/utils.h"
#include "../contrib/thrust/db_kernels.h"
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
    // nodes_.emplace_back(nodes_.size(), func_idx, unrolled_input_size + unrolled_output_size,
    // unrolled_output_size, args_copy);
    nodes_.emplace_back(nodes_.size(), func_idx, args_copy);
  }
}

template <>
void LazyAllocationLazyExecutor::AddPackedCall(const Index func_idx, const Index arg_count,
                                               const Index output_size, const ObjectRef* args,
                                               int num_args) {
  ICHECK(false) << "Un-unrolled packed calls not allowed in the lazy allocation executor";
}

template <>
void EagerAllocationLazyExecutor::AddPackedCallUnrolled(const Index func_idx, NDArray* args,
                                                        int num_args) {
  nodes_.emplace_back(nodes_.size(), func_idx, args, num_args);
}

template <>
void LazyAllocationLazyExecutor::AddPackedCallUnrolled(const Index func_idx, DLTensor** args,
                                                       int num_args) {
  nodes_.emplace_back(nodes_.size(), func_idx, args, num_args);
}

template <>
void EagerAllocationLazyExecutor::Execute() {
  for (EagerOpNode& node : nodes_) {
    InvokePackedFnUnrolled(node.func_idx_, vm_shared_state_->packed_funcs_[node.func_idx_],
                           node.args_.data(), node.args_.size());
  }
  nodes_.clear();
}

template <>
void LazyAllocationLazyExecutor::Execute() {
  // TODO
  nodes_.clear();
}

std::string GetDLTensorInfo(const DLTensor* tensor) {
  std::stringstream ss;
  ss << "Tensor [";
  for (int i = 0; i < tensor->ndim; ++i) {
    ss << tensor->shape[i] << " ";
  }
  ss << "] on device (" << tensor->device.device_type << ", " << tensor->device.device_id << ")";
  if (tensor->data == nullptr) {
    ss << " with null data";
  }
  return ss.str();
}

template <typename ConcreteExecutorType>
void ExecuteReduceSum(const ConcreteExecutorType& executor,
                      const std::vector<LazyOpNode*>& func_nodes) {
  static std::vector<NDArray> arg_holder;

  int num_nodes = func_nodes.size();
  void** output_raw_ptrs = static_cast<void**>(Arena::Current()->allocate_<void*>(num_nodes));
  int* input_indices = static_cast<int*>(Arena::Current()->allocate_<int>(num_nodes + 1));
  std::vector<void*> input_raw_ptrs;
  input_raw_ptrs.reserve(num_nodes * 2);

  auto allocator = executor.vm_shared_state_->allocators_[executor.accelerator_device_];
  auto& first_output = *(func_nodes[0]->args_.back());
  size_t output_size = GetDataSize(first_output);
  void* output_start = allocator->ArenaAlloc(num_nodes * output_size, 256, first_output.dtype).data;
  auto& accelerator_device = first_output.device;

  int ctr = 0;
  input_indices[0] = 0;
  for (int i = 0; i < num_nodes; ++i) {
    auto& node = *(func_nodes[i]);
    int num_reduce_tensors = node.args_.size() - 1;
    for (int j = 0; j < num_reduce_tensors; ++j) {
#ifdef DEBUG_CHECKS
      ICHECK(node.args_[j]->data != nullptr);
#endif
      input_raw_ptrs.push_back(node.args_[j]->data);
    }
    ctr += num_reduce_tensors;

    auto output_data_ptr = static_cast<char*>(output_start) + output_size;
    output_raw_ptrs[i] = output_data_ptr;
    node.args_[num_reduce_tensors]->data = output_data_ptr;
    input_indices[i + 1] = input_indices[i] + num_reduce_tensors;
  }

  auto num_inputs = input_raw_ptrs.size();
#ifdef DEBUG_CHECKS
  ICHECK_EQ(num_inputs, ctr);
#endif

  DLTensor* input_ptrs_device = Arena::Current()->allocate_<DLTensor>();
  {
    int64_t* shape_data = Arena::Current()->allocate_<int64_t>();
    shape_data[0] = num_inputs;
    auto dtype = DLDataType{kDLOpaqueHandle, 8 * sizeof(void*), 1};
    input_ptrs_device->device = accelerator_device;
    input_ptrs_device->data = allocator->ArenaAlloc(num_inputs * sizeof(void*), 256, dtype).data;
    input_ptrs_device->strides = nullptr;
    input_ptrs_device->ndim = 1;
    input_ptrs_device->dtype = dtype;
    input_ptrs_device->shape = shape_data;
    input_ptrs_device->byte_offset = 0;
  }

  DLTensor* output_ptrs_device = Arena::Current()->allocate_<DLTensor>();
  {
    int64_t* shape_data = Arena::Current()->allocate_<int64_t>();
    shape_data[0] = num_nodes;
    auto dtype = DLDataType{kDLOpaqueHandle, 8 * sizeof(void*), 1};
    output_ptrs_device->device = accelerator_device;
    output_ptrs_device->data = allocator->ArenaAlloc(num_nodes * sizeof(void*), 256, dtype).data;
    output_ptrs_device->strides = nullptr;
    output_ptrs_device->ndim = 1;
    output_ptrs_device->dtype = dtype;
    output_ptrs_device->shape = shape_data;
    output_ptrs_device->byte_offset = 0;
  }

  DLTensor* input_indices_device = Arena::Current()->allocate_<DLTensor>();
  {
    int64_t* shape_data = Arena::Current()->allocate_<int64_t>();
    shape_data[0] = (num_nodes + 1);
    auto dtype = DLDataType{kDLOpaqueHandle, 8 * sizeof(void*), 1};
    input_indices_device->device = accelerator_device;
    input_indices_device->data =
        allocator->ArenaAlloc((num_nodes + 1) * sizeof(void*), 256, dtype).data;
    input_indices_device->strides = nullptr;
    input_indices_device->ndim = 1;
    input_indices_device->dtype = dtype;
    input_indices_device->shape = shape_data;
    input_indices_device->byte_offset = 0;
  }

  ArrayCopyFromBytesAsync(input_ptrs_device, &(input_raw_ptrs[0]), sizeof(void*) * num_inputs);
  ArrayCopyFromBytesAsync(output_ptrs_device, output_raw_ptrs, sizeof(void*) * num_nodes);
  ArrayCopyFromBytesAsync(input_indices_device, input_indices, sizeof(void*) * (num_nodes + 1));

#ifdef DB_PROFILING
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStopCall();
    VMDBProfiler::ProfileDeviceStartCall("kernel");
  }
#endif
  tvm::contrib::reduce_sum_wrapper(
      static_cast<float**>(input_ptrs_device->data), static_cast<float**>(output_ptrs_device->data),
      static_cast<int*>(input_indices_device->data), num_nodes, output_size / sizeof(float));
#ifdef DB_PROFILING
  if (VMDBProfiler::DoProfile()) {
    DeviceAPI::Get(accelerator_device)->StreamSync(accelerator_device, nullptr);
    VMDBProfiler::ProfileDeviceStopCall();
    VMDBProfiler::ProfileHostStartCall("scheduling");
  }
#endif
}

template <typename ConcreteExecutorType>
void LazyAllocationExecuteOpNodeBatch(const ConcreteExecutorType& executor, const Index func_idx,
                                      const std::vector<LazyOpNode*>& func_nodes) {
  const VMSharedState<ConcreteExecutorType>& vm_shared_state = *(executor.vm_shared_state_);
  int32_t batch_size = func_nodes.size();
  if (batch_size == 1) {
    auto arity = executor.GetArity(func_idx);
    std::vector<TVMValue> values(arity);
    std::vector<int> codes(arity);
    runtime::TVMArgsSetter setter(values.data(), codes.data());

    // std::cout << "[LZ]  ExecutingUB " << func_idx << " " << arity << " " << func_nodes.size()
    // << std::endl;

    auto& allocator = vm_shared_state.allocators_[executor.accelerator_device_];
    auto& args = func_nodes[0]->args_;
    for (size_t i = 0; i < arity; ++i) {
      auto& arg = args[i];
      if (arg->data == nullptr) {
        arg->data = allocator->ArenaAlloc(GetDataSize(*arg), 256, arg->dtype).data;
      }
      // std::cout << "[LZ]   Arg " << i << " " << GetDLTensorInfo(arg) << std::endl;
      setter(i, arg);
    }

#ifdef DB_PROFILING
    if (VMDBProfiler::DoProfile()) {
      VMDBProfiler::ProfileHostStopCall();
      VMDBProfiler::ProfileDeviceStartCall("kernel");
    }
#endif
    TVMRetValue rv;
    vm_shared_state.packed_funcs_[func_idx].CallPacked(TVMArgs(values.data(), codes.data(), arity),
                                                       &rv);
#ifdef DB_PROFILING
    if (VMDBProfiler::DoProfile()) {
      auto& gpu_device = executor.vm_shared_state_->devices_[executor.accelerator_device_];
      DeviceAPI::Get(gpu_device)->StreamSync(gpu_device, nullptr);
      VMDBProfiler::ProfileDeviceStopCall();
      VMDBProfiler::ProfileHostStartCall("scheduling");
    }
#endif
  } else {
    auto& batched_func_idx = vm_shared_state.batched_funcs_[func_idx];

    auto& arg_modes = vm_shared_state.batched_func_arg_mode_[batched_func_idx];
    auto arity = executor.GetArity(func_idx);
    std::vector<TVMValue> values(arity + 1);
    std::vector<int> codes(arity + 1);
    runtime::TVMArgsSetter setter(values.data(), codes.data());
    setter(0, batch_size);
    int ctr = 1;

    // std::cout << "[LZ]  ExecutingB " << batched_func_idx << " " << arity << " " <<
    // func_nodes.size()
    // << std::endl;
    auto& allocator = vm_shared_state.allocators_[executor.accelerator_device_];
    for (size_t i = 0; i < arity; ++i) {
      switch (arg_modes[i]) {
        case kIgnore: {
          break;
        }
        case kReuse: {
          auto& arg = func_nodes[0]->args_[i];
          if (arg->data == nullptr) {
            auto ptr = allocator->ArenaAlloc(GetDataSize(*arg), 256, arg->dtype).data;
#pragma GCC ivdep
            for (int j = 0; j < batch_size; ++j) {
              func_nodes[j]->args_[i]->data = ptr;
            }
          }
          setter(ctr, arg);

          // std::cout << "[LZ]   Arg1 " << ctr << " " << GetDLTensorInfo(func_nodes[0]->args_[i])
          // << std::endl;

          ctr += 1;
          break;
        }
        case kScatter: {
          auto tensor = CreatePointerDLTensor(func_nodes, i, allocator);
          setter(ctr, tensor);

          // std::cout << "[LZ]   Arg2 " << ctr << " " << GetDLTensorInfo(tensor) << " "
          // << GetDLTensorInfo(func_nodes[0]->args_[i]) << std::endl;

          ctr += 1;
          break;
        }
        case kConcat: {
          auto tensor = CreateConcatenatedDLTensor(func_nodes, i, allocator);
          setter(ctr, tensor);

          // std::cout << "[LZ]   Arg3 " << ctr << " " << GetDLTensorInfo(tensor) << " "
          // << GetDLTensorInfo(func_nodes[0]->args_[i]) << std::endl;

          ctr += 1;
          break;
        }
      }
    }

#ifdef DB_PROFILING
    if (VMDBProfiler::DoProfile()) {
      VMDBProfiler::ProfileHostStopCall();
      VMDBProfiler::ProfileDeviceStartCall("kernel");
    }
#endif
    TVMRetValue rv;
    vm_shared_state.packed_funcs_[batched_func_idx].CallPacked(
        TVMArgs(values.data(), codes.data(), ctr), &rv);
#ifdef DB_PROFILING
    if (VMDBProfiler::DoProfile()) {
      auto& gpu_device = executor.vm_shared_state_->devices_[executor.accelerator_device_];
      DeviceAPI::Get(gpu_device)->StreamSync(gpu_device, nullptr);
      VMDBProfiler::ProfileDeviceStopCall();
      VMDBProfiler::ProfileHostStartCall("scheduling");
    }
#endif
  }
}

template <>
void EagerAllocationLazyExecutor::ExecuteOpNodeBatch(const Index func_idx,
                                                     const std::vector<EagerOpNode*>& func_nodes) {
#ifdef DB_PROFILING
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStopCall();
  }
#endif
  // std::cout << "[LE]  Executing " << func_idx << " " << func_nodes.size() << std::endl;
  // if (func_nodes.size() == 1) {
  //   InvokePackedFnUnrolled(func_idx, vm_shared_state_->packed_funcs_[func_idx],
  //                          func_nodes[0]->args_.data(),
  //                          func_nodes[0]->args_.size());
  // } else {
  //   auto batched_func_idx = vm_shared_state_->batched_funcs_[func_idx];
  //   InvokePackedFnBatchedUnrolled(
  //       batched_func_idx, vm_shared_state_->packed_funcs_[batched_func_idx],
  //       func_nodes[0]->arg_count_, func_nodes[0]->output_size_,
  //       vm_shared_state_->batched_func_arg_mode_[batched_func_idx], func_nodes);
  // }
#ifdef DB_PROFILING
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStartCall("batched_execution");
  }
#endif
}

template <>
void LazyAllocationLazyExecutor::ExecuteOpNodeBatch(const Index func_idx,
                                                    const std::vector<LazyOpNode*>& func_nodes) {
  if (func_idx == REDUCE_SUM_FUNC_INDEX) {
    ExecuteReduceSum<LazyAllocationLazyExecutor>(*this, func_nodes);
  } else {
    LazyAllocationExecuteOpNodeBatch<LazyAllocationLazyExecutor>(*this, func_idx, func_nodes);
  }
}

template <typename TensorType, typename TensorPtrType>
inline TensorPtrType GetPtr(TensorType& tensor);

template <>
inline DLTensor* GetPtr(DLTensor*& tensor) {
  return tensor;
}

template <>
inline const Object* GetPtr(NDArray& tensor) {
  return tensor.get();
}

template <typename TensorType, typename TensorPtrType>
void BatchedExecuteImpl(LazyExecutor<TensorType>* executor, bool coarsened_execution,
                        bool all_nodes_same_depth) {
#ifdef DB_PROFILING
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStartCall("batched_execution");
  }
#endif
  if (all_nodes_same_depth) {
    std::unordered_map<int, std::vector<OpNode<TensorType>*>> func_to_node;
    for (auto& node : executor->nodes_) {
      func_to_node[node.func_idx_].push_back(&node);
    }
    for (auto kv : func_to_node) {
      executor->ExecuteOpNodeBatch(kv.first, kv.second);
    }
  } else if (false) {
    std::unordered_map<TensorPtrType, int> output_tensor_to_node;
    size_t num_nodes = executor->nodes_.size();
    output_tensor_to_node.reserve(num_nodes * 2);
    int graph_depth = -1;
    std::vector<std::vector<OpNode<TensorType>*>> depth_to_node(num_nodes);
    std::vector<int> node_to_depth(num_nodes, -1);

    for (size_t i = 0; i < num_nodes; ++i) {
      OpNode<TensorType>& node = executor->nodes_[i];
      int max_depth = -1;
      if (node.func_idx_ == REDUCE_SUM_FUNC_INDEX) {
        for (size_t j = 0; j < node.args_.size() - 1; ++j) {
          auto it = output_tensor_to_node.find(GetPtr<TensorType, TensorPtrType>(node.args_[j]));
          if (it != output_tensor_to_node.end()) {
            auto input_node_id = it->second;
            max_depth = std::max(max_depth, node_to_depth[input_node_id]);
          }
        }
        output_tensor_to_node[GetPtr<TensorType, TensorPtrType>(node.args_.back())] = node.id_;
      } else {
        for (size_t j = executor->InputStart(node.func_idx_);
             j < executor->InputEnd(node.func_idx_); ++j) {
          auto it = output_tensor_to_node.find(GetPtr<TensorType, TensorPtrType>(node.args_[j]));
          if (it != output_tensor_to_node.end()) {
            auto input_node_id = it->second;
            max_depth = std::max(max_depth, node_to_depth[input_node_id]);
          }
        }
        for (size_t j = executor->OutputStart(node.func_idx_);
             j < executor->InoutEnd(node.func_idx_); ++j) {
          output_tensor_to_node[GetPtr<TensorType, TensorPtrType>(node.args_[j])] = node.id_;
        }
      }
      int node_depth = max_depth + 1;
      node_to_depth[i] = node_depth;
      depth_to_node[node_depth].push_back(&node);
      graph_depth = std::max(graph_depth, node_depth);
    }

    // std::cout << "[LZ] Graph depth " << graph_depth << " " << num_nodes << std::endl;
    int nodes_executed = 0;
    std::vector<std::unordered_map<int, std::vector<OpNode<TensorType>*>>> func_to_node_vecs;
    for (int j = 0; j <= graph_depth; ++j) {
      auto& depth_nodes = depth_to_node[j];
      std::unordered_map<int, std::vector<OpNode<TensorType>*>> func_to_node;
      // std::cout << "[LZ] Depth " << j << " " << depth_nodes.size() << std::endl;
      nodes_executed += depth_nodes.size();
      for (auto& node : depth_nodes) {
        func_to_node[node->func_idx_].push_back(node);
      }

      for (auto kv : func_to_node) {
        executor->ExecuteOpNodeBatch(kv.first, kv.second);
      }
    }
    ICHECK_EQ(nodes_executed, num_nodes);
  } else {
    // std::cout << "Agenda scheduling" << std::endl;
    std::unordered_map<TensorPtrType, std::vector<OpNode<TensorType>*>> tensor_to_consumers;
    auto num_packed_funs = executor->vm_shared_state_->packed_funcs_.size();
    std::vector<int> func_idx_to_position(num_packed_funs);
    std::vector<int> position_to_func_idx(num_packed_funs);
    for (size_t i = 0; i < num_packed_funs; ++i) {
      func_idx_to_position[i] = i;
      position_to_func_idx[i] = i;
    }
    size_t num_nodes = executor->nodes_.size();
    tensor_to_consumers.reserve(num_nodes * 2);
    std::vector<uint16_t> node_to_inputs(num_nodes, 0);
    std::vector<std::vector<OpNode<TensorType>*>> agenda(num_packed_funs);
    int agenda_size = 0;
    for (size_t j = 0; j < num_nodes; ++j) {
      auto& node = executor->nodes_[j];
      uint16_t num_inputs = 0;
      std::vector<TensorPtrType> uncomputed_inputs;
      std::vector<int> uncomputed_input_ids;
      if (node.func_idx_ == REDUCE_SUM_FUNC_INDEX) {
        for (size_t k = 0; k < node.args_.size() - 1; ++k) {
          auto tensor = node.args_[k];
          if (tensor->data == nullptr) {
            num_inputs++;
            uncomputed_inputs.push_back(GetPtr<TensorType, TensorPtrType>(tensor));
            uncomputed_input_ids.push_back(k);
          }
          tensor_to_consumers[GetPtr<TensorType, TensorPtrType>(tensor)].push_back(&node);
        }
      } else {
        for (size_t k = executor->InputStart(node.func_idx_);
             k < executor->InputEnd(node.func_idx_); ++k) {
          auto tensor = node.args_[k];
          if (tensor->data == nullptr) {
            num_inputs++;
            uncomputed_inputs.push_back(GetPtr<TensorType, TensorPtrType>(tensor));
            uncomputed_input_ids.push_back(k);
          }
          tensor_to_consumers[GetPtr<TensorType, TensorPtrType>(tensor)].push_back(&node);
        }
      }
      // std::cout << " Node: " << j << " " << node.func_idx_ << " " << num_inputs << " "
      // << support::PrintVector(uncomputed_inputs) << " "
      // << support::PrintVector(uncomputed_input_ids) << std::endl;
      if (num_inputs == 0) {
        agenda[func_idx_to_position[node.func_idx_]].push_back(&node);
        agenda_size++;
      } else {
        node_to_inputs[j] = num_inputs;
      }
    }

    // std::cout << " Map size: " << tensor_to_consumers.size() << std::endl;
    for (size_t j = 0; j < num_nodes; ++j) {
      auto& node = executor->nodes_[j];
      std::vector<int> consumer_ids;
      if (node.func_idx_ == REDUCE_SUM_FUNC_INDEX) {
        auto tensor = node.args_.back();
        if (tensor_to_consumers.count(GetPtr<TensorType, TensorPtrType>(tensor))) {
          for (auto consumer : tensor_to_consumers.at(GetPtr<TensorType, TensorPtrType>(tensor))) {
            consumer_ids.push_back(consumer->id_);
          }
        }
      } else {
        for (size_t i = executor->OutputStart(node.func_idx_);
             i < executor->InoutEnd(node.func_idx_); ++i) {
          auto tensor = node.args_[i];
          if (tensor_to_consumers.count(GetPtr<TensorType, TensorPtrType>(tensor))) {
            for (auto consumer :
                 tensor_to_consumers.at(GetPtr<TensorType, TensorPtrType>(tensor))) {
              consumer_ids.push_back(consumer->id_);
            }
          }
        }
      }
      // std::cout << " Edge: " << j << " " << support::PrintVector(consumer_ids) << std::endl;
    }

    while (agenda_size > 0) {
      int func_pos = -1;
      for (size_t i = 0; i < num_packed_funs; ++i) {
        auto& func_nodes = agenda[i];
        if (func_nodes.size() > 0) {
          func_pos = i;
          break;
        }
      }

      if (func_pos == -1) {
        break;
      }
      auto func_idx = position_to_func_idx[func_pos];

      // Execute
      // std::cout << "Func " << func_idx << std::endl;
      // Make a copy here so we don't edit the vector we're iterating
      // upon below.
      std::vector<OpNode<TensorType>*> func_nodes(agenda[func_pos].begin(), agenda[func_pos].end());
      executor->ExecuteOpNodeBatch(func_idx, func_nodes);
      agenda_size -= func_nodes.size();
      agenda[func_pos].clear();
      // Reduce to evaluate input counts for out nodes
      for (auto& node : func_nodes) {
        // std::cout << " Node " << node->id_ << std::endl;
        if (func_idx == REDUCE_SUM_FUNC_INDEX) {
          auto tensor = node->args_.back();
          // std::cout << "  Computed tensor " << GetPtr<TensorType, TensorPtrType>(tensor)
          // << std::endl;
          auto it = tensor_to_consumers.find(GetPtr<TensorType, TensorPtrType>(tensor));
          if (it != tensor_to_consumers.end()) {
            for (auto& out_node : it->second) {
              if (node_to_inputs[out_node->id_] == 1) {
                agenda[func_idx_to_position[out_node->func_idx_]].push_back(out_node);
                // std::cout << "   Added to agenda " << out_node->id_ << std::endl;
                agenda_size++;
              } else {
                --node_to_inputs[out_node->id_];
              }
            }
          }
        } else {
          for (size_t i = executor->OutputStart(func_idx); i < executor->InoutEnd(func_idx); ++i) {
            auto tensor = node->args_[i];
            // std::cout << "  Computed tensor " << GetPtr<TensorType, TensorPtrType>(tensor)
            // << std::endl;
            auto it = tensor_to_consumers.find(GetPtr<TensorType, TensorPtrType>(tensor));
            if (it != tensor_to_consumers.end()) {
              for (auto& out_node : it->second) {
                if (node_to_inputs[out_node->id_] == 1) {
                  agenda[func_idx_to_position[out_node->func_idx_]].push_back(out_node);
                  // std::cout << "   Added to agenda " << out_node->id_ << std::endl;
                  agenda_size++;
                } else {
                  --node_to_inputs[out_node->id_];
                }
              }
            }
          }
        }
      }
    }
  }
  executor->nodes_.clear();
#ifdef DB_PROFILING
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStopCall();
  }
#endif
}

template <>
void EagerAllocationLazyExecutor::BatchedExecute(bool coarsened_execution,
                                                 bool all_nodes_same_depth) {
  BatchedExecuteImpl<NDArray, const Object*>(this, coarsened_execution, all_nodes_same_depth);
  if (accelerator_device_ == GPU_INDEX) {
    auto& gpu_device = vm_shared_state_->devices_[GPU_INDEX];
    DeviceAPI::Get(gpu_device)->StreamSync(gpu_device, nullptr);
  }
}

template <>
void LazyAllocationLazyExecutor::BatchedExecute(bool coarsened_execution,
                                                bool all_nodes_same_depth) {
  BatchedExecuteImpl<DLTensor*, DLTensor*>(this, coarsened_execution, all_nodes_same_depth);
  if (accelerator_device_ == GPU_INDEX) {
    auto& gpu_device = vm_shared_state_->devices_[GPU_INDEX];
    DeviceAPI::Get(gpu_device)->StreamSync(gpu_device, nullptr);
  }
}

void DepthTrackingExecutor::AddPackedCallUnrolledWithDepth(const Index func_idx, const int depth,
                                                           DLTensor** args, int num_args) {
  auto size = nodes_.size();
  auto depthp1 = depth + 1;
  if (size < depthp1) {
    nodes_.resize(depthp1);
  }
  nodes_[depth].emplace_back(size, func_idx, args, num_args);
}

void DepthTrackingExecutor::ExecuteOpNodeBatch(const Index func_idx,
                                               const std::vector<LazyOpNode*>& nodes) {
  LazyAllocationExecuteOpNodeBatch<DepthTrackingExecutor>(*this, func_idx, nodes);
}

void DepthTrackingExecutor::Execute() {
  ICHECK(false)
      << "Not implemented. Please use LazyExecutor<DLTensor*> class for this functionality.";
}

void DepthTrackingExecutor::BatchedExecute(bool coarsened_execution, bool all_nodes_same_depth) {
#ifdef DB_PROFILING
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStartCall("scheduling");
  }
#endif
  for (size_t j = 0; j < nodes_.size(); ++j) {
    auto& depth_nodes = nodes_[j];
    // std::cout << "[LZ] Depth " << j << " " << depth_nodes.size() << std::endl;
    std::unordered_map<int, std::vector<LazyOpNode*>> func_to_node;

    for (auto& node : depth_nodes) {
      func_to_node[node.func_idx_].push_back(&node);
    }

    for (auto kv : func_to_node) {
      if (kv.first == REDUCE_SUM_FUNC_INDEX) {
        ExecuteReduceSum<DepthTrackingExecutor>(*this, kv.second);
      } else {
        ExecuteOpNodeBatch(kv.first, kv.second);
      }
    }
  }
  nodes_.clear();
  if (accelerator_device_ == GPU_INDEX) {
    auto& gpu_device = vm_shared_state_->devices_[GPU_INDEX];
    DeviceAPI::Get(gpu_device)->StreamSync(gpu_device, nullptr);
  }
#ifdef DB_PROFILING
  if (VMDBProfiler::DoProfile()) {
    VMDBProfiler::ProfileHostStopCall();
  }
#endif
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
