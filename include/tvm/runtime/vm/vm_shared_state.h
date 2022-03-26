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
 * \brief The shared state for the Relay virtual machine runtime.
 */
#ifndef TVM_RUNTIME_VM_VM_SHARED_STATE_H_
#define TVM_RUNTIME_VM_VM_SHARED_STATE_H_

#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/bytecode.h>
#include <tvm/runtime/vm/dynamic_batching.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/lazy_executor.h>
#include <tvm/runtime/vm/memory_manager.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {
namespace vm {

/*! \brief range over one dimension */
class VMExecutionOptionsNode : public Object {
 public:
  /*! \brief whether the prim funcs are coarsened */
  bool coarsened_execution;
  /*! \brief whether to execute tensor operations lazily */
  bool lazy_execution;
  /*! \brief whether to execute tensor operations in a batched manner */
  bool batched_execution;
  /*! \brief whether the batched kernels operate on scattered tensors */
  bool scattered_kernels;
  /*! \brief whether to launch multiple concurrent VMs, each
   * corresponding to one batch element instance */
  bool concurrent_execution;
  /*! \brief the batch size to be used for execution */
  size_t batch_size;

  VMExecutionOptionsNode() {}
  VMExecutionOptionsNode(bool coarsened_execution_, bool lazy_execution_, bool batched_execution_,
                         bool scattered_kernels_, bool concurrent_execution_, size_t batch_size_)
      : coarsened_execution(coarsened_execution_),
        lazy_execution(lazy_execution_),
        batched_execution(batched_execution_),
        scattered_kernels(scattered_kernels_),
        concurrent_execution(concurrent_execution_),
        batch_size(batch_size_) {}

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "VMExecutionOptions";
  TVM_DECLARE_FINAL_OBJECT_INFO(VMExecutionOptionsNode, Object);
};

/*! \brief VMExecutionOptions constainer  */
class VMExecutionOptions : public ObjectRef {
 public:
  /*!
   * \brief constructor
   * \param lazy_execution whether to execute tensor operations lazily.
   */
  TVM_DLL VMExecutionOptions(bool coarsened_execution, bool lazy_execution, bool batched_execution,
                             bool scattered_kernels, bool concurrent_execution, size_t batch_size);
  // declare VMExecutionOptions.
  TVM_DEFINE_OBJECT_REF_METHODS(VMExecutionOptions, ObjectRef, VMExecutionOptionsNode);
};

/*!
 * \brief A srtuct that aggregates all global state of the vm
 * i.e. everything except the current runtime state (the pc, stacks,
 * etc).
 */
template <typename ExecutorType>
struct VMSharedState {
  /*! \brief The executable the VM will operate on. */
  Executable* exec_ = nullptr;
  /*! \brief The virtual machine's packed function table. */
  std::vector<PackedFunc> packed_funcs_;
  /*!
   * \brief The "physical" devices the VM can execute primitives on. All "device indexes"
   * are w.r.t. this vector. Each entry in this vector must match the corresponding entry
   * in the executable's "virtual" devices vector.
   */
  std::vector<Device> devices_;
  /*! \brief The cached memory allocators, one per device. */
  std::vector<Allocator*> allocators_;
  /*!
   * \brief The constant pool for runtime. It caches the device dependent
   * object to avoid rellocation of constants during inference.
   */
  std::vector<ObjectRef> const_pool_;
  /*!
   * \brief A lazy executor which maintains a computational graph of
   * all the packed funcs executed.
   */
  ExecutorType lazy_executor_;
  /*!
   * \brief A mapping from packed_funcs to their batched counterparts.
   */
  std::vector<Index> batched_funcs_;
  /*!
   * \brief A mapping from packed_funcs to their batched arg modes.
   */
  std::vector<std::vector<DBBatchedArgMode>> batched_func_arg_mode_;
  /*!
   * \brief A mapping from packed_funcs to their arg access modes
   * counterparts.
   */
  std::vector<std::vector<DBArgAccessMode>> prim_func_arg_access_mode_;
  /*!
   * \brief A mapping from packed_funcs to the index in their parameters when their output tensors
   * start.
   */
  std::vector<int> outputs_start;
  /*!
   * \brief A mapping from packed_funcs to the index in their
   * parameters when their input/output tensors start.
   */
  std::vector<int> inouts_start;
  /*!
   * \brief A mapping from packed_funcs to the number of their arguments.
   */
  std::vector<int> args_end;
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_VM_SHARED_STATE_H_
