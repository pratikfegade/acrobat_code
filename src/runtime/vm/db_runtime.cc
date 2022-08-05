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

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <dmlc/memory_io.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/vm/arena.h>
#include <tvm/runtime/vm/db_execution_utils.h>
#include <tvm/runtime/vm/db_runtime.h>

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

template class DynBatchRuntime<LazyExecutor<NDArray>, NDArray>;
template class DynBatchRuntime<LazyExecutor<DLTensor*>, DLTensor*>;
template class DynBatchRuntime<DepthTrackingExecutor, DLTensor*>;

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::CacheConstants() {
  for (size_t const_index = 0; const_index < shared_state_.exec_->constants.size(); ++const_index) {
    auto constant_obj = shared_state_.exec_->constants[const_index];
    // We cache the allocated object in the constant pool. To measure, the
    // first iteration will set the pool up. The other iterations will
    // directly reuse the allocated objects.
    if (shared_state_.const_pool_.size() <= static_cast<size_t>(const_index)) {
      shared_state_.const_pool_.resize(const_index + 1);
    }

    if (!shared_state_.const_pool_[const_index].defined()) {
      Device dev = GetDevice(shared_state_.exec_->const_device_indexes[const_index]);
      shared_state_.const_pool_[const_index] = CopyTo(constant_obj, dev);
    }
  }
}

template <typename ExecutorType, typename TensorType>
NDArray DynBatchRuntime<ExecutorType, TensorType>::GetConstant(int64_t const_index) {
  return Downcast<NDArray>(shared_state_.const_pool_[const_index]);
}

template <typename ExecutorType, typename TensorType>
Storage DynBatchRuntime<ExecutorType, TensorType>::AllocateStorage(int64_t size, int64_t alignment,
                                                                   DLDataType dtype,
                                                                   int64_t device_index) {
  auto storage_obj = SimpleObjAllocator().make_object<StorageObj>();
  Allocator* allocator = GetAllocator(device_index);
  ICHECK(allocator) << "Did you forget to init the VirtualMachine with devices?";
  VLOG(2) << "AllocStorage: allocation_size=" << size << ", alignment=" << alignment
          << ", dtype_hint=" << DLDataType2String(dtype) << ", device_index=" << device_index;

  storage_obj->buffer = allocator->Alloc(size, alignment, dtype);
  Storage storage(storage_obj);
  return storage;
}
template <typename ExecutorType, typename TensorType>

NDArray DynBatchRuntime<ExecutorType, TensorType>::AllocTensor(const Storage& storage,
                                                               int64_t offset, uint32_t ndim,
                                                               int64_t* shape_array,
                                                               DLDataType dtype) {
  auto shape = std::vector<int64_t>(ndim);
  for (uint32_t i = 0; i < ndim; ++i) {
    shape[i] = shape_array[i];
  }

#if TVM_LOG_DEBUG
  std::ostringstream os;
  os << "AllocTensor: ";
  os << "offset=" << offset;
  os << ", shape=[";
  for (auto i : shape) {
    os << i << ",";
  }
  os << "]";
  os << ", dtype=" << DLDataType2String(dtype);
  VLOG(2) << os.str();
#endif
  auto obj = storage->AllocNDArray(offset, shape, dtype);
  return obj;
}

template <typename ExecutorType, typename TensorType>
NDArray DynBatchRuntime<ExecutorType, TensorType>::AllocTensorReg(const Storage& storage,
                                                                  int64_t offset,
                                                                  const NDArray shape_tensor_dev,
                                                                  DLDataType dtype) {
  Device cpu_dev = GetDevice(shared_state_.exec_->host_device_index);
  auto shape_tensor = Downcast<NDArray>(CopyTo(shape_tensor_dev, cpu_dev));
  auto shape = ToShape(shape_tensor);
  return storage->AllocNDArray(offset, shape, dtype);
}

template <typename ExecutorType, typename TensorType>
DLTensor* DynBatchRuntime<ExecutorType, TensorType>::AllocArrayWrapper(int64_t* shape_data,
                                                                       int64_t ndim,
                                                                       DLDataType dtype,
                                                                       int64_t device_index) {
  DLTensor* wrapper = Arena::Current()->allocate_<DLTensor>();
  wrapper->device = shared_state_.devices_[device_index];
  wrapper->data = nullptr;
  wrapper->strides = nullptr;
  wrapper->ndim = ndim;
  wrapper->dtype = std::move(dtype);
  wrapper->shape = shape_data;
  wrapper->byte_offset = 0;
  return wrapper;
}

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::InvokePacked(int64_t packed_index, TensorType* args,
                                                             int64_t num_args) {
  if (concurrent_execution_ || lazy_execution_) {
    shared_state_.lazy_executor_.AddPackedCallUnrolled(packed_index, args, num_args);
  } else {
    InvokePackedFnUnrolled(packed_index, shared_state_.packed_funcs_[packed_index], args, num_args);
  }
}

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::InvokePackedWithDepth(int64_t packed_index,
                                                                      int64_t depth,
                                                                      TensorType* args,
                                                                      int64_t num_args) {
  if (concurrent_execution_ || lazy_execution_) {
    shared_state_.lazy_executor_.AddPackedCallUnrolledWithDepth(packed_index, depth, args,
                                                                num_args);
  } else {
    ICHECK(false) << "Depth tracking not supported with eager execution";
  }
}

template <typename ExecutorType, typename TensorType>
NDArray DynBatchRuntime<ExecutorType, TensorType>::DeviceCopy(const NDArray& src_data,
                                                              const int64_t src_device_index,
                                                              const int64_t dst_device_index) {
  Device actual_src_dev = src_data->device;
  Device inst_src_dev = GetDevice(src_device_index);
  ICHECK_EQ(actual_src_dev.device_type, inst_src_dev.device_type);
  ICHECK_EQ(actual_src_dev.device_id, inst_src_dev.device_id);
  Device dst_dev = GetDevice(dst_device_index);
  return src_data.CopyTo(dst_dev);
}

template <typename ExecutorType, typename TensorType>
NDArray DynBatchRuntime<ExecutorType, TensorType>::ReshapeTensor(NDArray& tensor_arr,
                                                                 const NDArray& shape_tensor) {
  // Read the shape from shape tensor
  const DLTensor* dl_tensor = shape_tensor.operator->();
  ICHECK_EQ(dl_tensor->dtype.code, 0u);
  ICHECK_EQ(dl_tensor->dtype.bits, 64u);
  int64_t* dims = reinterpret_cast<int64_t*>(dl_tensor->data);
  int64_t ndim = shape_tensor->shape[0];
  std::vector<int64_t> shape(dims, dims + ndim);
  // Reshape the input tensor
#if TVM_LOG_DEBUG
  std::ostringstream os;
  os << "ReshapeTensor: ";
  os << "shape=[";
  for (auto i : shape) {
    os << i << ",";
  }
  os << "]";
  os << ", dtype=" << DLDataType2String(tensor_arr->dtype);
  VLOG(2) << os.str();
#endif
  return tensor_arr.CreateView(shape, tensor_arr->dtype);
}

template <typename ExecutorType, typename TensorType>
DLTensor* DynBatchRuntime<ExecutorType, TensorType>::ReshapeTensor(DLTensor* tensor_arr,
                                                                   const DLTensor* shape_tensor) {
  // Read the shape from shape tensor
  ICHECK_EQ(shape_tensor->dtype.code, 0u);
  ICHECK_EQ(shape_tensor->dtype.bits, 64u);
  int64_t* dims = reinterpret_cast<int64_t*>(shape_tensor->data);
  int64_t ndim = shape_tensor->shape[0];

  DLTensor* result = Arena::Current()->allocate_<DLTensor>();
  {
    result->device = tensor_arr->device;
    result->strides = nullptr;
    result->ndim = ndim;
    result->data = tensor_arr->data;
    result->dtype = tensor_arr->dtype;
    result->shape = dims;
    result->byte_offset = tensor_arr->byte_offset;
  }
  return result;
}

template <typename ExecutorType, typename TensorType>
NDArray DynBatchRuntime<ExecutorType, TensorType>::ShapeOf(const NDArray& input_array) {
  int ndim = input_array->ndim;
  auto out_tensor =
      NDArray::Empty({ndim}, {kDLInt, 64, 1}, GetDevice(shared_state_.exec_->host_device_index));
  for (int i = 0; i < ndim; ++i) {
    reinterpret_cast<int64_t*>(out_tensor->data)[i] = input_array->shape[i];
  }
  return out_tensor;
}

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::NextProgramPhase() {
  shared_state_.lazy_executor_.NextProgramPhase();
}

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::SetProgramPhase(int phase) {
  shared_state_.lazy_executor_.SetProgramPhase(phase);
}

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::ResetProgramPhase() {
  shared_state_.lazy_executor_.ResetProgramPhase();
}

template <typename ExecutorType, typename TensorType>
bool DynBatchRuntime<ExecutorType, TensorType>::MightOOMSoon() {
  size_t total = 0;
  size_t free = 0;
  cudaMemGetInfo(&free, &total);
  return static_cast<double>(free) < 0.3 * static_cast<double>(total);
}

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::StartCUDAProfiler() {
  cudaProfilerStart();
}

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::StopCUDAProfiler() {
  cudaProfilerStop();
}

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::ResetExecutionState(int seed, bool release_memory) {
  shared_state_.lazy_executor_.ResetExecutor();
  RecycleAllArenaMemory();
  if (release_memory) {
    ReleaseAllArenaMemory();
  }
  if (seed >= 0) {
    RandomGenerator::Current().ResetWithSeed(seed);
  } else {
    RandomGenerator::Current().ResetWithRandomDevice();
  }
}

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::ReleaseAllArenaMemory() {
  for (auto& allocator : shared_state_.allocators_) {
    allocator->ReleaseAll();
  }
}

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::RecycleAllArenaMemory() {
  Arena::Current()->RecycleAll();
  for (auto& allocator : shared_state_.allocators_) {
    allocator->ArenaFree();
  }
}

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::LazyExecute() {
  if (batched_execution_) {
    shared_state_.lazy_executor_.BatchedExecute(coarsened_execution_);
  } else {
    shared_state_.lazy_executor_.Execute();
  }
}

template <typename ExecutorType, typename TensorType>
PackedFunc DynBatchRuntime<ExecutorType, TensorType>::GetFunction(
    const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  return {};
}

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::InitSharedState() {
  this->shared_state_.lazy_executor_.vm_shared_state_ = &this->shared_state_;
}

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::SetExecutionOptions(VMExecutionOptions options) {
  this->coarsened_execution_ = options->coarsened_execution;
  this->lazy_execution_ = options->lazy_execution;
  this->scattered_kernels_ = options->scattered_kernels;
  this->batched_execution_ = options->batched_execution;
  this->concurrent_execution_ = options->concurrent_execution;
  this->batch_size_ = options->batch_size;

  if (false) {
    if (options->coarsened_execution) {
      std::cout << "[VM] Executing coarsened" << std::endl;
    } else {
      std::cout << "[VM] Executing uncoarsened" << std::endl;
    }

    if (options->batched_execution) {
      std::cout << "[VM] Executing batched" << std::endl;
    } else {
      std::cout << "[VM] Executing unbatched" << std::endl;
    }

    if (options->lazy_execution) {
      std::cout << "[VM] Executing lazily" << std::endl;
    } else {
      std::cout << "[VM] Executing eagerly" << std::endl;
    }

    if (options->scattered_kernels) {
      std::cout << "[VM] Executing scattered kernels" << std::endl;
    } else {
      std::cout << "[VM] Executing unscattered kernels" << std::endl;
    }

    if (options->concurrent_execution) {
      std::cout << "[VM] Executing concurrent" << std::endl;
    } else {
      std::cout << "[VM] Executing unconcurrent" << std::endl;
    }
#ifdef DEBUG_CHECKS
    std::cout << "[VM] Enabling debug checks" << std::endl;
#endif
  }
}

template <typename ExecutorType, typename TensorType>
inline Device DynBatchRuntime<ExecutorType, TensorType>::GetDevice(Index device_index) const {
  ICHECK_GE(shared_state_.devices_.size(), device_index)
      << "invalid device index: " << device_index;
  return shared_state_.devices_[device_index];
}

template <typename ExecutorType, typename TensorType>
inline Allocator* DynBatchRuntime<ExecutorType, TensorType>::GetAllocator(
    Index device_index) const {
  ICHECK_GE(shared_state_.allocators_.size(), device_index)
      << "invalid device index: " << device_index;
  return shared_state_.allocators_[device_index];
}

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::LoadExecutable(Executable* exec) {
  ICHECK(exec) << "The executable is not created yet.";
  ICHECK(exec->late_bound_constant_names.empty())
      << "Need to load late-bound-constants before creating VM";

  shared_state_.exec_ = exec;

  runtime::Module lib = shared_state_.exec_->GetLib();

  ICHECK(exec->primitive_map.empty() || lib.operator->())
      << "If the executable has declared primitive functions, the "
      << "generated kernel library must non-be null.";

  size_t num_packed_funs = shared_state_.exec_->primitive_map.size();
  ICHECK_EQ(num_packed_funs, shared_state_.exec_->batched_func_arg_mode.size());
  ICHECK_EQ(num_packed_funs, shared_state_.exec_->prim_func_arg_access_mode.size());
  for (size_t i = 0; i < num_packed_funs; ++i) {
    shared_state_.batched_func_arg_mode_.push_back(shared_state_.exec_->batched_func_arg_mode[i]);
    shared_state_.prim_func_arg_access_mode_.push_back(
        shared_state_.exec_->prim_func_arg_access_mode[i]);
  }

  std::vector<std::string> ordered_packed_fn_names(num_packed_funs);
  for (const auto& it : shared_state_.exec_->primitive_map) {
    const auto& packed_name = it.first;
    auto packed_index = static_cast<size_t>(it.second);
    ordered_packed_fn_names[packed_index] = packed_name;
  }

  for (size_t packed_index = 0; packed_index < num_packed_funs; ++packed_index) {
    const auto& packed_name = ordered_packed_fn_names[packed_index];
    if (shared_state_.packed_funcs_.size() <= packed_index) {
      shared_state_.packed_funcs_.resize(packed_index + 1);
      shared_state_.outputs_start.resize(packed_index + 1);
      shared_state_.inouts_start.resize(packed_index + 1);
      shared_state_.args_end.resize(packed_index + 1);
    }
    tvm::runtime::PackedFunc pf = lib.GetFunction(packed_name, /*query_imports=*/true);

    ICHECK(pf != nullptr) << "Cannot find function in module: " << packed_name;
    shared_state_.packed_funcs_[packed_index] = pf;

    auto& arg_access_modes = shared_state_.prim_func_arg_access_mode_[packed_index];

    int num_inputs = 0;
    int num_outputs = 0;
    int num_inouts = 0;
    for (size_t i = 0; i < arg_access_modes.size(); ++i) {
      switch (arg_access_modes[i]) {
        case kInput:
          num_inputs++;
          break;
        case kOutput:
          num_outputs++;
          break;
        case kInputOutput:
          num_inouts++;
          break;
        case kUnused:
          ICHECK(false);
          break;
      }
    }

    shared_state_.outputs_start[packed_index] = num_inputs;
    shared_state_.inouts_start[packed_index] = num_inputs + num_outputs;
    shared_state_.args_end[packed_index] = arg_access_modes.size();

    ICHECK(pf != nullptr) << packed_name;
    auto& registry = ::tvm::runtime::Registry::Register(packed_name);
    registry.set_body(pf);

    if (batched_execution_) {
      auto bit = shared_state_.exec_->primitive_map.find(GetBatchedName(packed_name));
      if (bit != shared_state_.exec_->primitive_map.end()) {
        if (shared_state_.batched_funcs_.size() <= packed_index) {
          shared_state_.batched_funcs_.resize(packed_index + 1, -1);
        }
        shared_state_.batched_funcs_[packed_index] = bit->second;
      }
    }

    bool print = false;
    if (print) {
      if (batched_execution_) {
        std::cout << "[VM] Fun " << packed_index << " " << packed_name;
        if (!IsBatchedName(packed_name)) {
          std::cout << " " << shared_state_.batched_funcs_[packed_index];
        }
      } else {
        std::cout << "[VM] Fun " << packed_index << " " << packed_name;
      }

      // if (coarsened_execution_) {
      std::cout << "  ScMode: [";
      for (size_t i = 0; i < shared_state_.batched_func_arg_mode_[packed_index].size(); ++i) {
        std::cout << shared_state_.batched_func_arg_mode_[packed_index][i] << " ";
      }
      std::cout << "]";
      // }
      std::cout << "   AccMode: [";
      for (size_t i = 0; i < shared_state_.prim_func_arg_access_mode_[packed_index].size(); ++i) {
        std::cout << shared_state_.prim_func_arg_access_mode_[packed_index][i] << " ";
      }
      std::cout << "] " << shared_state_.outputs_start[packed_index] << " "
                << shared_state_.inouts_start[packed_index] << std::endl;
    }
  }

  for (size_t i = 0; i < shared_state_.packed_funcs_.size(); ++i) {
    ICHECK(shared_state_.packed_funcs_[i] != nullptr)
        << "Packed function " << i << " is not initialized";
  }
}

template <typename ExecutorType, typename TensorType>
void DynBatchRuntime<ExecutorType, TensorType>::Init(
    const std::vector<Device>& physical_devices, const std::vector<AllocatorType>& alloc_types) {
  ICHECK_EQ(physical_devices.size(), alloc_types.size());

  // Find a physical device to represent each virtual device the VM code requires.
  // (Recall the VM instructions refer to devices by "device index" into this vector of
  // virtual devices.)
  const size_t num_virtual_devices = shared_state_.exec_->virtual_devices.size();
  if (num_virtual_devices == 2) {
    // GPU execution
    shared_state_.lazy_executor_.accelerator_device_ = 1;
  } else {
    shared_state_.lazy_executor_.accelerator_device_ = 0;
  }
  shared_state_.devices_.reserve(num_virtual_devices);
  shared_state_.allocators_.reserve(num_virtual_devices);

  for (size_t device_index = 0; device_index < num_virtual_devices; ++device_index) {
    // We'll retain the legacy behaviour and just match by device type.
    // TODO(mbs): Generalize.
    DLDeviceType virtual_device_type =
        shared_state_.exec_->virtual_devices[device_index].device_type;
    auto itr = std::find_if(physical_devices.begin(), physical_devices.end(),
                            [virtual_device_type](const Device& physical_device) {
                              return physical_device.device_type == virtual_device_type;
                            });
    CHECK(itr != physical_devices.end())
        << "Unable to find a physical device (from among the " << physical_devices.size()
        << " given) to match the virtual device with device type " << virtual_device_type;
    const size_t i = std::distance(physical_devices.begin(), itr);
    shared_state_.devices_.push_back(*itr);
    shared_state_.allocators_.push_back(MemoryManager::GetOrCreateAllocator(*itr, alloc_types[i]));
  }

  // if (mlockall(MCL_CURRENT | MCL_FUTURE) < 0) {
  //   perror("mlockall");
  //   exit(-1);
  // }
}

runtime::Module CreateEagerAllocationDynBatchRuntime(Executable* exec) {
  auto vm = make_object<DynBatchRuntime<EagerAllocationLazyExecutor, NDArray>>();
  vm->LoadExecutable(exec);
  return runtime::Module(vm);
}

runtime::Module CreateLazyAllocationDynBatchRuntime(Executable* exec) {
  auto vm = make_object<DynBatchRuntime<LazyAllocationLazyExecutor, DLTensor*>>();
  vm->LoadExecutable(exec);
  return runtime::Module(vm);
}

TVM_REGISTER_GLOBAL("runtime._DynBatchRuntime").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  auto* exec = dynamic_cast<Executable*>(mod.operator->());
  bool eager = args[1];
  ICHECK(exec) << "The virtual machine executable has not been defined yet.";
  if (eager) {
    *rv = CreateEagerAllocationDynBatchRuntime(exec);
  } else {
    *rv = CreateLazyAllocationDynBatchRuntime(exec);
  }
});

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
