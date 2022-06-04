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
 * \file tvm/runtime/vm/vm.h
 * \brief The Relay virtual machine runtime.
 */
#ifndef TVM_RUNTIME_VM_LAZY_EXECUTOR_H_
#define TVM_RUNTIME_VM_LAZY_EXECUTOR_H_

#include <tvm/runtime/container/closure.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/bytecode.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/memory_manager.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace vm {

class VirtualMachine;

template <typename ExecutorType>
struct VMSharedState;

template <typename ExecutorType, typename TensorType>
class DynBatchRuntime;

template <typename TensorType>
class OpNode {
 public:
  OpNode(const int id, const Index func_idx, const std::vector<TensorType> args)
      : id_(id), func_idx_(func_idx), args_(args) {}

  OpNode(const int id, const Index func_idx, TensorType* args, int num_args)
      : id_(id), func_idx_(func_idx) {
    args_.reserve(num_args);
    for (int i = 0; i < num_args; ++i) {
      args_.push_back(args[i]);
    }
  }

  const int id_;
  const Index func_idx_;
  std::vector<TensorType> args_;
};

typedef OpNode<NDArray> EagerOpNode;
typedef OpNode<DLTensor*> LazyOpNode;

template <typename ConcreteExecutorType, typename TensorType>
class AbstractExecutor {
 public:
  virtual void AddPackedCall(const Index func_idx, const Index arg_count, const Index output_size,
                             const ObjectRef* args, int num_args) = 0;

  virtual void AddPackedCallUnrolled(const Index func_idx, TensorType* args, int num_args) = 0;

  virtual void AddPackedCallUnrolledWithDepth(const Index func_idx, const int depth,
                                              TensorType* args, int num_args) = 0;

  virtual void Execute() = 0;

  virtual void BatchedExecute(bool coarsened_execution, bool all_nodes_same_depth = false) = 0;

  virtual void ExecuteOpNodeBatch(const Index func_idx,
                                  const std::vector<OpNode<TensorType>*>& nodes) = 0;

  inline size_t InputStart(const Index idx) const { return 0; }
  inline size_t InputEnd(const Index idx) const { return vm_shared_state_->outputs_start[idx]; }

  inline size_t OutputStart(const Index idx) const { return vm_shared_state_->outputs_start[idx]; }
  inline size_t OutputEnd(const Index idx) const { return vm_shared_state_->inouts_start[idx]; }

  inline size_t InoutStart(const Index idx) const { return vm_shared_state_->inouts_start[idx]; }
  inline size_t InoutEnd(const Index idx) const { return vm_shared_state_->args_end[idx]; }

  inline size_t GetArity(const Index idx) const { return vm_shared_state_->args_end[idx]; }

  inline void NextProgramPhase() {
    std::cout << "[LAZY] NP" << std::endl;
    phase_++;
  }

  inline void ResetProgramPhase() { phase_ = 0; }

  /*! \brief Pointer to the shared state of the VM this executor is
      associated with */
  VMSharedState<ConcreteExecutorType>* vm_shared_state_;
  /*! \brief The index of the primary execution device. We support the host CPU or a GPU for this
   * field. The host CPU is always 0, while a GPU is 1 */
  int accelerator_device_{0};
  /*! \brief Current program phase */
  int phase_{0};
  /*! \brief Profiling */
  runtime::profiling::Profiler* profiler_{nullptr};
};

struct KernelPGOStats {
  int execution_counts_{0};
  int average_dynamic_batch_size_{0};
};

/*!
 * \brief A lazy tensor executor for the virtual machine.
 *
 */
template <typename TensorType>
class LazyExecutor final : public AbstractExecutor<LazyExecutor<TensorType>, TensorType> {
 public:
  void AddPackedCall(const Index func_idx, const Index arg_count, const Index output_size,
                     const ObjectRef* args, int num_args);

  void AddPackedCallUnrolled(const Index func_idx, TensorType* args, int num_args);

  void AddPackedCallUnrolledWithDepth(const Index func_idx, const int depth, TensorType* args,
                                      int num_args) {
    ICHECK(false) << "Not implemented. Use a depth tracking executor.";
  }

  void Execute();

  void BatchedExecute(bool coarsened_execution, bool all_nodes_same_depth = false);

  void ExecuteOpNodeBatch(const Index func_idx, const std::vector<OpNode<TensorType>*>& nodes);

  void SetPGO(bool value) { this->pgo_ = value; }

  /*! \brief list of nodes to execute */
  std::vector<std::vector<OpNode<TensorType>>> nodes_;
  /*! \brief node counter to assign node ids */
  int node_ctr_{0};
  /*! \brief Whether to execute or to gather PGO stats */
  bool pgo_{false};
  /*! \brief Execution counts for PackedFuncs, for when pgo is turned on */
  std::unordered_map<std::string, KernelPGOStats> pgo_stats_;
};

typedef LazyExecutor<NDArray> EagerAllocationLazyExecutor;
typedef LazyExecutor<DLTensor*> LazyAllocationLazyExecutor;

template <>
void EagerAllocationLazyExecutor::AddPackedCallUnrolled(const Index func_idx, NDArray* args,
                                                        int num_args);

template <>
void LazyAllocationLazyExecutor::AddPackedCallUnrolled(const Index func_idx, DLTensor** args,
                                                       int num_args);

template <>
void EagerAllocationLazyExecutor::Execute();

template <>
void LazyAllocationLazyExecutor::Execute();

template <>
void EagerAllocationLazyExecutor::BatchedExecute(bool coarsened_execution,
                                                 bool all_nodes_same_depth);

template <>
void LazyAllocationLazyExecutor::BatchedExecute(bool coarsened_execution,
                                                bool all_nodes_same_depth);

template <>
void EagerAllocationLazyExecutor::ExecuteOpNodeBatch(const Index func_idx,
                                                     const std::vector<EagerOpNode*>& func_nodes);

template <>
void LazyAllocationLazyExecutor::ExecuteOpNodeBatch(const Index func_idx,
                                                    const std::vector<LazyOpNode*>& func_nodes);

class DepthTrackingExecutor final : public AbstractExecutor<DepthTrackingExecutor, DLTensor*> {
 public:
  void AddPackedCall(const Index func_idx, const Index arg_count, const Index output_size,
                     const ObjectRef* args, int num_args) {
    ICHECK(false) << "Not implemented. Use a depth tracking executor.";
  }

  void AddPackedCallUnrolled(const Index func_idx, DLTensor** args, int num_args) {
    ICHECK(false) << "Not implemented. Use a depth tracking executor.";
  }

  void AddPackedCallUnrolledWithDepth(const Index func_idx, const int depth, DLTensor** args,
                                      int num_args);

  void Execute();

  void BatchedExecute(bool coarsened_execution, bool all_nodes_same_depth = false);

  void ExecuteOpNodeBatch(const Index func_idx, const std::vector<LazyOpNode*>& nodes);

  /*! \brief list of nodes to execute sorted by depth */
  std::vector<std::vector<std::vector<OpNode<DLTensor*>>>> nodes_{MAX_PROGRAM_PHASES};
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_LAZY_EXECUTOR_H_
