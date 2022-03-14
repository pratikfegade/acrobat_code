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
#ifndef TVM_RUNTIME_VM_DYN_BATCH_RUNTIME_H_
#define TVM_RUNTIME_VM_DYN_BATCH_RUNTIME_H_

#include <tvm/runtime/container/closure.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/bytecode.h>
#include <tvm/runtime/vm/dynamic_batching.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/lazy_executor.h>
#include <tvm/runtime/vm/memory_manager.h>
#include <tvm/runtime/vm/vm.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace vm {

/*!
 * \brief The virtual machine.
 *
 * The virtual machine contains all the current execution state,
 * as well as the executable.
 *
 * The goal is to have a single self-contained object,
 * enabling one to easily pass around VMs, execute them on
 * multiple threads, or serialize them to disk or over the
 * wire.
 */
class DynBatchRuntime : public runtime::ModuleNode {
 public:
  /*!
   * \brief Get a PackedFunc from module.
   *
   *  The PackedFunc may not be fully initialized,
   *  there might still be first time running overhead when
   *  executing the function on certain devices.
   *  For benchmarking, use prepare to eliminate
   *
   * \param name the name of the function.
   * \param sptr_to_self The shared_ptr that points to this module node.
   *
   * \return PackedFunc(nullptr) when it is not available.
   *
   * \note The function will always remain valid.
   *   If the function needs resource from the module(e.g. late linking),
   *   it should capture sptr_to_self.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  ~DynBatchRuntime() {}

  const char* type_key() const final { return "DynBatchRuntime"; }

  DynBatchRuntime() {}

  /*!
   * \brief load the executable for the virtual machine.
   * \param exec The executable.
   */
  void LoadExecutable(Executable* exec);

  /*!
   * \brief set runtime options for the VM.
   * \param The options.
   */
  void SetExecutionOptions(VMExecutionOptions options);

  /*!
   * \brief Initialize the shared state if needed.
   */
  void InitSharedState();

  /*!
   * \brief Load constant from the executable.
   * \param const_index The constant index in the executable.
   *
   * \return The result constant array
   */
  NDArray LoadConstant(int64_t const_index);

  /*!
   * \brief Invoke a packed function.
   * \param packed_index The index of the callee in the executable.
   * \param arity The arity of the function
   * \param output_size The number of outputs of the function
   * \param args Pointer to args
   * \param num_args number of arguments
   */
  void InvokePacked(int64_t packed_index, int64_t arg_count, int64_t output_size,
                    const tvm::runtime::NDArray* args, int64_t num_args);

  /*!
   * \brief Allocate a memory storage object.
   * \param allocation_size The size of the storage to be allocated.
   * \param alignment The alignment of the storage to be allocated.
   * \param dtype The datatype of the storage to be allocated.
   * \param device_index The device the memory is to be allocated on
   *
   * \return The allocated storage object
   */
  Storage AllocateStorage(int64_t size, int64_t alignment, DLDataType dtype, int64_t device_index);

  /*!
   * \brief Allocate a tensor.
   * \param storage The storage to allocate from.
   * \param offset The offset into the storage to allocate from
   * \param ndim The number of dimensions.
   * \param shape The shape of tensor.
   * \param dtype The datatype of tensor to be allocated
   *
   * \return The allocated tensor object
   */
  NDArray AllocTensor(const Storage& storage, int64_t offset, uint32_t ndim, int64_t* shape,
                      DLDataType dtype);

  /*!
   * \brief Allocate a tensor.
   * \param storage The storage to allocate from.
   * \param offset The offset into the storage to allocate from
   * \param shape_tensor The tensor storing the shape of the tensor to be allocated.
   * \param dtype The datatype of tensor to be allocated
   *
   * \return The allocated tensor object
   */
  NDArray AllocTensorReg(const Storage& storage, int64_t offset, const NDArray shape_tensor,
                         DLDataType dtype);

  /*!
   * \brief Copy a tensor between devices.
   * \param src_data The tensor to copy from.
   * \param src_device_index The index of the source device
   * \param dst_device_index The index of the destination device
   *
   * \return The copied tensor object
   */
  NDArray DeviceCopy(const NDArray& src_data, const int64_t src_device_index,
                     const int64_t dst_device_index);

  /*!
   * \brief Reshape a tensor.
   * \param tensor_arr The tensor to be reshaped.
   * \param shape_tensor The new shape
   *
   * \return The reshaped tensor object
   */
  NDArray ReshapeTensor(NDArray& tensor_arr, const NDArray& shape_tensor);

  /*!
   * \brief Obtain the shape of a tensor.
   * \param input_array The input tensor.
   *
   * \return The shape of the input tensor
   */
  NDArray ShapeOf(const NDArray& input_array);

  /*!
   * \brief Initialize the virtual machine for a set of (physical) devices.
   * \param physical_devices The set of TVM devices.
   * \param alloc_types The allocator types for each device.
   */
  void Init(const std::vector<Device>& physical_devices,
            const std::vector<AllocatorType>& alloc_types);

  /*!
   * \brief Get the current instance of the runtime.
   */
  static inline ObjectPtr<DynBatchRuntime> Current() { return current_; }

  static inline ObjectPtr<DynBatchRuntime> CreateRuntime() {
    ICHECK(current_ == nullptr);
    current_ = make_object<DynBatchRuntime>();
    current_ref_ = ObjectRef(current_);
    return current_;
  }

 protected:
  /*! \brief Get device from the device list based on a given device index. */
  Device GetDevice(Index device_index) const;
  Allocator* GetAllocator(Index device_index) const;

  /*!
   * \brief Invoke a global setting up the VM state to execute.
   *
   * This does not begin execution of the VM.
   */
  void InvokeGlobal(const VMFunction& func, const std::vector<ObjectRef>& args, const int offset);

  /*!
   * \brief Set inputs to a function.
   * \param name The function name
   * \param args args[offset:] are arguments to the
   * function. If the arguments are not of the correct device for the function,
   * they will be copied to the device.
   * \param offset Starting offset of the arguments in `args`.
   * \param batch_size Batch size.
   * \param num_args Number of args to consume from args.
   */
  void SetInput(std::string name, TVMArgs args, int offset, int batch_size, int num_args);

 protected:
  friend class LazyExecutor;

  /*! \brief The global state excluding all runtime state. Aggregated
      in a struct for easier shared across multiple vm instances when
      executing multiple concurrent batch elements */
  VMSharedState* shared_state_;
  /*! \brief The function name to inputs mapping. */
  std::unordered_map<std::string, std::vector<ObjectRef>> inputs_;
  /*!
   * \brief Whether the generated prim funcs are coarsened.
   */
  bool coarsened_execution_ = false;
  /*!
   * \brief Whether to execute tensor ops lazily.
   */
  bool lazy_execution_ = false;
  /*!
   * \brief Whether to execute tensor ops in a batched manner.
   */
  bool batched_execution_ = false;
  /*!
   * \brief Whether the batched kernels operate on scattered tensors
   */
  bool scattered_kernels_ = false;
  /*!
   * \brief whether to launch multiple concurrent VMs, each
   * corresponding to one batch element instance
   */
  bool concurrent_execution_;
  /*!
   * \brief Whether the current VM object is a concurrent VM object
   */
  bool concurrent_vm_ = false;
  /*!
   * \brief Batch size the VM is operating on
   */
  int batch_size_;

  /*!
   * \brief Current instance of the runtime
   */
  static ObjectPtr<DynBatchRuntime> current_;
  /*!
   * \brief Current instance of the runtime
   */
  static ObjectRef current_ref_;
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_DYN_BATCH_RUNTIME_H_