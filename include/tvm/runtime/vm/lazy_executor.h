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
class VMSharedState;

class OpNode {
 public:
  OpNode(const int id, const Index func_idx, const Index arg_count, const Index output_size,
         const std::vector<NDArray> args)
      : id_(id),
        func_idx_(func_idx),
        arg_count_(arg_count),
        output_size_(output_size),
        args_(args) {}

  inline Index InputStart() { return 0; }

  inline Index InputEnd() { return arg_count_ - output_size_; }

  inline Index OutputStart() { return arg_count_ - output_size_; }

  inline Index OutputEnd() { return arg_count_; }

  const int id_;
  const Index func_idx_;
  const Index arg_count_;
  const Index output_size_;
  const std::vector<NDArray> args_;
};

/*!
 * \brief A lazy tensor executor for the virtual machine.
 *
 */
class LazyExecutor {
 public:
  void AddPackedCall(const Index func_idx, const Index arg_count, const Index output_size,
                     const std::vector<ObjectRef> args);

  void Execute();

  void BatchedExecute(bool coarsened_execution, bool all_nodes_same_depth = false);

 private:
  void ExecuteOpNodeBatch(const std::unordered_map<int, std::vector<OpNode*>>& func_to_node);

  friend class VirtualMachine;
  friend class ConcurrentVirtualMachine;
  friend class DynBatchRuntime;

  /*! \brief Pointer to the shared state of the VM this executor is
      associated with */
  const VMSharedState* vm_shared_state_;
  /*! \brief list of nodes to execute */
  std::vector<OpNode> nodes_;
  //   /*! \brief Profiling data storage */
  // #ifdef PROFILE_VM
  //   std::unordered_map<Index, float> profiling_data_;
  // #endif
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_LAZY_EXECUTOR_H_
