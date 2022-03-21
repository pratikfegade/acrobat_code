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

ObjectPtr<DynBatchRuntime> DynBatchRuntime::current_;
ObjectRef DynBatchRuntime::current_ref_;

void DynBatchRuntime::CacheConstants() {
  for (int64_t const_index = 0; const_index < shared_state_->exec_->constants.size();
       ++const_index) {
    auto constant_obj = shared_state_->exec_->constants[const_index];
    // We cache the allocated object in the constant pool. To measure, the
    // first iteration will set the pool up. The other iterations will
    // directly reuse the allocated objects.
    if (shared_state_->const_pool_.size() <= static_cast<size_t>(const_index)) {
      shared_state_->const_pool_.resize(const_index + 1);
    }

    if (!shared_state_->const_pool_[const_index].defined()) {
      Device dev = GetDevice(shared_state_->exec_->const_device_indexes[const_index]);
      shared_state_->const_pool_[const_index] = CopyTo(constant_obj, dev);
    }
  }
}

NDArray DynBatchRuntime::GetConstant(int64_t const_index) {
  return Downcast<NDArray>(shared_state_->const_pool_[const_index]);
}

void DynBatchRuntime::InvokePacked(int64_t packed_index, int64_t arg_count, int64_t output_size,
                                   const tvm::runtime::NDArray* args, int64_t num_args) {
  if (concurrent_execution_ || lazy_execution_) {
    shared_state_->lazy_executor_.AddPackedCallUnrolled(packed_index, arg_count, output_size, args,
                                                        num_args);
  } else {
    // std::cout << "Invoking " << packed_index << std::endl;
    InvokePackedFnUnrolled(packed_index, shared_state_->packed_funcs_[packed_index], output_size,
                           args, num_args);
  }
}

Storage DynBatchRuntime::AllocateStorage(int64_t size, int64_t alignment, DLDataType dtype,
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

NDArray DynBatchRuntime::AllocTensor(const Storage& storage, int64_t offset, uint32_t ndim,
                                     int64_t* shape_array, DLDataType dtype) {
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

NDArray DynBatchRuntime::AllocTensorReg(const Storage& storage, int64_t offset,
                                        const NDArray shape_tensor_dev, DLDataType dtype) {
  Device cpu_dev = GetDevice(shared_state_->exec_->host_device_index);
  auto shape_tensor = Downcast<NDArray>(CopyTo(shape_tensor_dev, cpu_dev));
  auto shape = ToShape(shape_tensor);
  return storage->AllocNDArray(offset, shape, dtype);
}

NDArray DynBatchRuntime::DeviceCopy(const NDArray& src_data, const int64_t src_device_index,
                                    const int64_t dst_device_index) {
  Device actual_src_dev = src_data->device;
  Device inst_src_dev = GetDevice(src_device_index);
  ICHECK_EQ(actual_src_dev.device_type, inst_src_dev.device_type);
  ICHECK_EQ(actual_src_dev.device_id, inst_src_dev.device_id);
  Device dst_dev = GetDevice(dst_device_index);
  return src_data.CopyTo(dst_dev);
}

NDArray DynBatchRuntime::ReshapeTensor(NDArray& tensor_arr, const NDArray& shape_tensor) {
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

NDArray DynBatchRuntime::ShapeOf(const NDArray& input_array) {
  int ndim = input_array->ndim;
  auto out_tensor =
      NDArray::Empty({ndim}, {kDLInt, 64, 1}, GetDevice(shared_state_->exec_->host_device_index));
  for (int i = 0; i < ndim; ++i) {
    reinterpret_cast<int64_t*>(out_tensor->data)[i] = input_array->shape[i];
  }
  return out_tensor;
}

void DynBatchRuntime::LazyExecute() {
  if (batched_execution_) {
    shared_state_->lazy_executor_.BatchedExecute(coarsened_execution_);
  } else {
    shared_state_->lazy_executor_.Execute();
  }
}

PackedFunc DynBatchRuntime::GetFunction(const std::string& name,
                                        const ObjectPtr<Object>& sptr_to_self) {
  return {};
}

void DynBatchRuntime::InitSharedState() {
  if (!shared_state_) {
    shared_state_ = new VMSharedState();
  }
  this->shared_state_->lazy_executor_.vm_shared_state_ = this->shared_state_;
}

void DynBatchRuntime::SetExecutionOptions(VMExecutionOptions options) {
  this->coarsened_execution_ = options->coarsened_execution;
  this->lazy_execution_ = options->lazy_execution;
  this->scattered_kernels_ = options->scattered_kernels;
  this->batched_execution_ = options->batched_execution;
  this->concurrent_execution_ = options->concurrent_execution;
  this->batch_size_ = options->batch_size;

  if (true) {
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
  }
}

inline Device DynBatchRuntime::GetDevice(Index device_index) const {
  ICHECK_GE(shared_state_->devices_.size(), device_index)
      << "invalid device index: " << device_index;
  return shared_state_->devices_[device_index];
}

inline Allocator* DynBatchRuntime::GetAllocator(Index device_index) const {
  ICHECK_GE(shared_state_->allocators_.size(), device_index)
      << "invalid device index: " << device_index;
  return shared_state_->allocators_[device_index];
}

void DynBatchRuntime::LoadExecutable(Executable* exec) {
  ICHECK(exec) << "The executable is not created yet.";
  ICHECK(exec->late_bound_constant_names.empty())
      << "Need to load late-bound-constants before creating VM";

  shared_state_->exec_ = exec;

  runtime::Module lib = shared_state_->exec_->GetLib();

  ICHECK(exec->primitive_map.empty() || lib.operator->())
      << "If the executable has declared primitive functions, the "
      << "generated kernel library must non-be null.";

  this->shared_state_->batched_func_arg_mode_ = shared_state_->exec_->batched_func_arg_mode;
  this->shared_state_->prim_func_arg_access_mode_ = shared_state_->exec_->prim_func_arg_access_mode;

  for (const auto& it : shared_state_->exec_->primitive_map) {
    const auto& packed_name = it.first;
    auto packed_index = static_cast<size_t>(it.second);
    if (shared_state_->packed_funcs_.size() <= packed_index) {
      shared_state_->packed_funcs_.resize(packed_index + 1);
    }
    tvm::runtime::PackedFunc pf = lib.GetFunction(packed_name, /*query_imports=*/true);

    ICHECK(pf != nullptr) << "Cannot find function in module: " << packed_name;
    shared_state_->packed_funcs_[packed_index] = pf;

    if (batched_execution_) {
      std::cout << "[VM] Fun " << packed_index << " " << packed_name << " "
                << shared_state_->exec_->batched_func_arg_mode[packed_index].size() << std::endl;
    } else {
      std::cout << "[VM] Fun " << packed_index << " " << packed_name << std::endl;
    }
    ICHECK(pf != nullptr) << packed_name;
    auto& registry = ::tvm::runtime::Registry::Register(packed_name);
    registry.set_body(pf);

    if (batched_execution_) {
      auto bit = shared_state_->exec_->primitive_map.find(GetBatchedName(packed_name));
      if (bit != shared_state_->exec_->primitive_map.end()) {
        if (shared_state_->batched_funcs_.size() <= packed_index) {
          shared_state_->batched_funcs_.resize(packed_index + 1, -1);
        }
        shared_state_->batched_funcs_[packed_index] = bit->second;
      }
    }

    // if (coarsened_execution_) {
    //   std::cout << "[VM]   ArgAccessModes: [";
    //   for (size_t i = 0; i <
    //   this->shared_state_->prim_func_arg_access_mode_[packed_index].size();
    //        ++i) {
    //     std::cout << this->shared_state_->prim_func_arg_access_mode_[packed_index][i] << " ";
    //   }
    //   std::cout << "]" << std::endl;
    // }
  }

  // for (const auto& it : shared_state_->exec_->primitive_map) {
  //   const auto& packed_name = it.first;
  //   ICHECK(Registry::Get(packed_name)->body());
  // }

  // for (auto name : Registry::ListNames()) {
  //   const PackedFunc* f = Registry::Get(name);
  //   if (!f->body()) {
  //     std::cout << "[VM] NoBody1 " << name << std::endl;
  //   }
  // }

  for (size_t i = 0; i < shared_state_->packed_funcs_.size(); ++i) {
    ICHECK(shared_state_->packed_funcs_[i] != nullptr)
        << "Packed function " << i << " is not initialized";
  }
}

void DynBatchRuntime::Init(const std::vector<Device>& physical_devices,
                           const std::vector<AllocatorType>& alloc_types) {
  ICHECK_EQ(physical_devices.size(), alloc_types.size());

  // Find a physical device to represent each virtual device the VM code requires.
  // (Recall the VM instructions refer to devices by "device index" into this vector of
  // virtual devices.)
  const size_t num_virtual_devices = shared_state_->exec_->virtual_devices.size();
  shared_state_->devices_.reserve(num_virtual_devices);
  shared_state_->allocators_.reserve(num_virtual_devices);

  for (size_t device_index = 0; device_index < num_virtual_devices; ++device_index) {
    // We'll retain the legacy behaviour and just match by device type.
    // TODO(mbs): Generalize.
    DLDeviceType virtual_device_type =
        shared_state_->exec_->virtual_devices[device_index].device_type;
    auto itr = std::find_if(physical_devices.begin(), physical_devices.end(),
                            [virtual_device_type](const Device& physical_device) {
                              return physical_device.device_type == virtual_device_type;
                            });
    CHECK(itr != physical_devices.end())
        << "Unable to find a physical device (from among the " << physical_devices.size()
        << " given) to match the virtual device with device type " << virtual_device_type;
    const size_t i = std::distance(physical_devices.begin(), itr);
    shared_state_->devices_.push_back(*itr);
    shared_state_->allocators_.push_back(MemoryManager::GetOrCreateAllocator(*itr, alloc_types[i]));
  }
}

runtime::Module CreateDynBatchRuntime(Executable* exec) {
  auto vm = make_object<DynBatchRuntime>();
  vm->LoadExecutable(exec);
  return runtime::Module(vm);
}

TVM_REGISTER_GLOBAL("runtime._DynBatchRuntime").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  auto* exec = dynamic_cast<Executable*>(mod.operator->());
  ICHECK(exec) << "The virtual machine executable has not been defined yet.";
  *rv = CreateDynBatchRuntime(exec);
});

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
