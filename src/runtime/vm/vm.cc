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
#include <tvm/node/reflection.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/vm/db_execution_utils.h>
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

TVM_REGISTER_OBJECT_TYPE(VMClosureObj);
TVM_REGISTER_OBJECT_TYPE(VMExecutionOptionsNode);

VMExecutionOptions::VMExecutionOptions(bool coarsened_execution, bool lazy_execution,
                                       bool batched_execution, bool scattered_kernels,
                                       bool concurrent_execution, bool pgo, size_t batch_size)
    : VMExecutionOptions(make_object<VMExecutionOptionsNode>(
          coarsened_execution, lazy_execution, batched_execution, scattered_kernels,
          concurrent_execution, pgo, batch_size)) {}

TVM_REGISTER_GLOBAL("runtime.CreateVMExecutionOptions")
    .set_body_typed([](bool coarsened_execution, bool lazy_execution, bool batched_execution,
                       bool scattered_kernels, bool concurrent_execution, bool pgo,
                       size_t batch_size) {
      return VMExecutionOptions(coarsened_execution, lazy_execution, batched_execution,
                                scattered_kernels, concurrent_execution, pgo, batch_size);
    });

TVM_REGISTER_GLOBAL("runtime.CreatePGOVMExecutionOptions")
    .set_body_typed([](VMExecutionOptions options) {
      return VMExecutionOptions(false, true, true, true, options->concurrent_execution, true,
                                options->batch_size);
    });

VMClosure::VMClosure(size_t func_index, std::vector<ObjectRef> free_vars) {
  auto ptr = make_object<VMClosureObj>();
  ptr->func_index = func_index;
  ptr->free_vars = std::move(free_vars);
  data_ = std::move(ptr);
}

void VMFunctionPrint(std::ostream& os, const VMFunction& vm_func) {
  os << vm_func.name << ": " << std::endl;
  for (size_t i = 0; i < vm_func.instructions.size(); ++i) {
    os << i << ": " << vm_func.instructions[i] << ";" << std::endl;
  }
}

std::ostream& operator<<(std::ostream& os, const VMFunction& vm_func) {
  VMFunctionPrint(os, vm_func);
  return os;
}

void VirtualMachine::OpStartHook(Instruction instr) {}
void VirtualMachine::OpStopHook() {}

void VirtualMachine::InvokeWrapper(std::string func_name, TVMRetValue* rv) {
  ICHECK(shared_state_->exec_) << "The executable is not created yet.";

  auto git = shared_state_->exec_->global_map.find(func_name);
  ICHECK(git != shared_state_->exec_->global_map.end())
      << "Cannot find function " << func_name << " in the executable";
  auto func = shared_state_->exec_->functions[git->second];
  if (func.params.empty()) {
    *rv = Invoke(func, {});
  } else {
    auto it = inputs_.find(func_name);
    ICHECK(it != inputs_.end()) << "Input has not been set for function " << func_name
                                << " in concurrent? " << concurrent_vm_;
    const std::vector<ObjectRef>& func_args = it->second;
    *rv = Invoke(func, func_args);
  }
}

PackedFunc VirtualMachine::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  if (name == "invoke") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->InvokeWrapper(args[0], rv); });
  } else if (name == "get_pgo_stats") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK(shared_state_->lazy_executor_.pgo_);
      Map<String, Map<String, String>> stats;
      for (auto kv : shared_state_->lazy_executor_.pgo_stats_) {
        auto exe_count = kv.second.execution_counts_;
        auto batch_size = (kv.second.average_dynamic_batch_size_ * 1.0) / std::max(exe_count, 1);
        Map<String, String> this_stats;
        this_stats.Set("exe_count", String(std::to_string(exe_count)));
        this_stats.Set("batch_size", String(std::to_string(batch_size)));
        stats.Set(String(kv.first), this_stats);
      }
      *rv = stats;
    });
  } else if (name == "invoke_stateful") {
    // TODO(tkonolige, jroesch, tqchen): invoke_stateful and get_output are
    // stop-gap measure to allow using vm over a remote connection.
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      PackedFunc invoke = GetFunction("invoke", sptr_to_self);
      TVMRetValue rv_;
      invoke.CallPacked(args, &rv_);
    });
  } else if (name == "invoke_return_to_device") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      Device host{static_cast<DLDeviceType>(args[1].operator int()), args[2].operator int()};

      SetInput(args[0].operator std::string(), args, 3, 1, args.size() - 3);
      PackedFunc invoke = GetFunction("invoke", sptr_to_self);
      TVMRetValue rv_;
      invoke.CallPacked(args, &rv_);  // Invoke only uses the first arg, so the rest of the args
                                      // should not cause an issue
      if (rv_.type_code() == kTVMObjectHandle) {
        ADT adt = Downcast<ADT>(rv_.operator ObjectRef());
        std::vector<ObjectRef> transfered;
        for (size_t i = 0; i < adt.size(); i++) {
          transfered.push_back(CopyTo(adt[i], host));
        }
        *rv = ADT(adt.tag(), transfered);
      } else {
        *rv = CopyTo(rv_, host);
      }
    });
  } else if (name == "get_output") {
    return TypedPackedFunc<NDArray(int64_t)>([this](int64_t index) {
      if (this->return_register_.as<ADTObj>()) {
        return Downcast<NDArray>(Downcast<ADT>(this->return_register_)[index]);
      } else {
        CHECK_EQ(index, 0) << "VM output contains only one item, but you are trying to get the "
                           << index << "th.";
        return Downcast<NDArray>(this->return_register_);
      }
    });
  } else if (name == "get_num_outputs") {
    return TypedPackedFunc<int64_t(void)>([this]() -> int64_t {
      // single output is an NDArray not an ADT
      if (this->return_register_.as<ADTObj>()) {
        return Downcast<ADT>(this->return_register_).size();
      } else {
        return 1;
      }
    });
  } else if (name == "get_input_index") {
    return TypedPackedFunc<int64_t(std::string, std::string)>(
        [this](std::string input_name, std::string func_name) {
          auto gvit = shared_state_->exec_->global_map.find(func_name);
          ICHECK(gvit != shared_state_->exec_->global_map.end())
              << "Cannot find function " << func_name;
          auto func_index = gvit->second;
          const auto& vm_func = shared_state_->exec_->functions[func_index];
          const auto& param_names = vm_func.params;
          for (uint64_t i = 0; i < param_names.size(); i++) {
            if (input_name == param_names[i]) {
              return static_cast<int64_t>(i);
            }
          }
          return static_cast<int64_t>(-1);
        });
  } else if (name == "init") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.size() % 3, 0);
      std::vector<Device> devices;
      std::vector<AllocatorType> alloc_types;
      for (int i = 0; i < args.size() / 3; ++i) {
        Device dev;
        int device_type = args[i * 3];
        dev.device_type = DLDeviceType(device_type);
        dev.device_id = args[i * 3 + 1];
        int type = args[i * 3 + 2];
        devices.push_back(dev);
        alloc_types.push_back(AllocatorType(type));
      }
      this->Init(devices, alloc_types);
    });
  } else if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      SetInput(args[0], args, 2, args[1], args.size() - 2);
    });
  } else if (name == "load_late_bound_consts") {
    return PackedFunc([this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.size(), 1);
      std::string path = args[0];
      shared_state_->exec_->LoadLateBoundConstantsFromFile(path);
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
  }
}

void VirtualMachine::SetInput(std::string func_name, TVMArgs args, int offset, int batch_size,
                              int num_args) {
  if (concurrent_execution_) {
    ICHECK(!concurrent_vm_);
    ICHECK_EQ(batch_size, 1);
  }

  ICHECK_EQ(num_args % batch_size, 0);
  int num_args_per_instance = num_args / batch_size;
  ICHECK(shared_state_->exec_) << "The executable is not created yet.";
  auto gvit = shared_state_->exec_->global_map.find(func_name);
  ICHECK(gvit != shared_state_->exec_->global_map.end()) << "Cannot find function " << func_name;
  auto func_index = gvit->second;
  const auto& vm_func = shared_state_->exec_->functions[func_index];
  const auto& param_names = vm_func.params;
  ICHECK_EQ(num_args_per_instance, param_names.size())
      << "The number of provided parameters doesn't match the number of arguments "
      << num_args_per_instance << " " << offset << " " << param_names.size();
  ICHECK_EQ(param_names.size(), vm_func.param_device_indexes.size())
      << "The number of provided parameters doesn't match the number of assigned devices";

  std::vector<ObjectRef> batch_func_args(batch_size * num_args_per_instance);
  for (int j = 0; j < batch_size; ++j) {
    // std::cout << "[VM] Setting inputs for " << func_name << std::endl;
    for (int i = offset; i < num_args_per_instance + offset; ++i) {
      Device dev = GetDevice(vm_func.param_device_indexes[i - offset]);

      if (args[i].type_code() == kTVMDLTensorHandle) {
        // Automatically convert input DLTensors to NDArray
        DLTensor* tensor = args[i];
        std::vector<int64_t> shape;
        for (int64_t i = 0; i < tensor->ndim; i++) {
          shape.push_back(tensor->shape[i]);
        }
        NDArray ary = NDArray::Empty(shape, tensor->dtype, dev);
        ary.CopyFrom(tensor);
        batch_func_args[j * num_args_per_instance + i - offset] = ary;
      } else {
        ObjectRef obj = CopyTo(args[i], dev);
        batch_func_args[j * num_args_per_instance + i - offset] = obj;
      }
    }
    offset += num_args_per_instance;
  }
  inputs_.erase(func_name);
  inputs_.emplace(func_name, batch_func_args);
}

void VirtualMachine::InitSharedState(bool pgo) {
  if (!shared_state_) {
    shared_state_ = new VMSharedState<EagerAllocationLazyExecutor>();
    this->shared_state_->lazy_executor_.SetPGO(pgo);
  }
  this->shared_state_->lazy_executor_.vm_shared_state_ = this->shared_state_;
}

void VirtualMachine::SetExecutionOptions(VMExecutionOptions options) {
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
  }
}

inline Device VirtualMachine::GetDevice(Index device_index) const {
  ICHECK_GE(shared_state_->devices_.size(), device_index)
      << "invalid device index: " << device_index;
  return shared_state_->devices_[device_index];
}

inline Allocator* VirtualMachine::GetAllocator(Index device_index) const {
  ICHECK_GE(shared_state_->allocators_.size(), device_index)
      << "invalid device index: " << device_index;
  return shared_state_->allocators_[device_index];
}

void VirtualMachine::PushFrame(Index arg_count, Index ret_pc, const VMFunction& vm_func) {
  auto frame = VMFrame(ret_pc, func_index_, arg_count, code_, vm_func.register_file_size);
  frames_.push_back(frame);
}

Index VirtualMachine::PopFrame() {
  ICHECK_GT(frames_.size(), 0);
  const VMFrame& fr = frames_.back();
  func_index_ = fr.func_index;
  code_ = fr.code;
  pc_ = fr.pc;
  auto call_stack_size = frames_.size();
  frames_.pop_back();
  return call_stack_size;
}

void VirtualMachine::InvokeGlobal(const VMFunction& func, const std::vector<ObjectRef>& args,
                                  const int offset) {
  VLOG(2) << "Invoking global " << func.name << " " << args.size();

  PushFrame(func.params.size(), this->pc_ + 1, func);
  for (size_t i = 0; i < func.params.size(); ++i) {
    WriteRegister(i, args[i + offset]);
  }
  VLOG(2) << "func.params= " << func.params.size();

  code_ = func.instructions.data();
  pc_ = 0;
}

ObjectRef VirtualMachine::Invoke(const VMFunction& func, const std::vector<ObjectRef>& args) {
  DLOG(INFO) << "Executing Function: " << std::endl << func;
  for (int i = 0; i < static_cast<int>(shared_state_->devices_.size()); ++i) {
    DLOG(INFO) << "Device " << i << " has device type " << shared_state_->devices_[i].device_type
               << " and device id " << shared_state_->devices_[i].device_id
               << (i == shared_state_->exec_->host_device_index ? " (using as host device)" : "");
  }

  std::cout << "HELLO " << batch_size_ << std::endl;
  for (int i = 0; i < batch_size_; ++i) {
    InvokeGlobal(func, args, i * (args.size() / batch_size_));
    RunLoop();
  }

  if (lazy_execution_) {
    if (batched_execution_) {
      std::cout << "YOLO" << std::endl;
      shared_state_->lazy_executor_.BatchedExecute(true, coarsened_execution_);
    } else {
      shared_state_->lazy_executor_.Execute();
    }
  }

  return return_register_;
}

ObjectRef VirtualMachine::Invoke(const std::string& name, const std::vector<ObjectRef>& args) {
  ICHECK(shared_state_->exec_) << "The executable has not been created yet.";
  auto it = shared_state_->exec_->global_map.find(name);
  ICHECK(it != shared_state_->exec_->global_map.end())
      << "Cannot find function " << name << " in the executable";
  Index func_index = it->second;
  VLOG(2) << "Invoke Global " << name << " at index " << func_index;
  return Invoke(shared_state_->exec_->functions[func_index], args);
}

void VirtualMachine::InvokePacked(Index packed_index, Index arg_count, Index output_size,
                                  const std::vector<ObjectRef>& args, bool batched) {
  if (concurrent_execution_ || lazy_execution_) {
    shared_state_->lazy_executor_.AddPackedCall(packed_index, arg_count, output_size, args.data(),
                                                args.size());
  } else {
    if (batched) {
      InvokePackedFn(shared_state_->packed_funcs_[packed_index], arg_count, output_size,
                     args.data(), args.size(), shared_state_->batched_func_arg_mode_[packed_index],
                     batched, this->scattered_kernels_);
    } else {
      InvokePackedFn(shared_state_->packed_funcs_[packed_index], arg_count, output_size,
                     args.data(), args.size(), {}, batched, this->scattered_kernels_);
    }
  }
}

void VirtualMachine::LoadExecutable(Executable* exec) {
  ICHECK(exec) << "The executable is not created yet.";
  ICHECK(exec->late_bound_constant_names.empty())
      << "Need to load late-bound-constants before creating VM";

  shared_state_->exec_ = exec;

  runtime::Module lib = shared_state_->exec_->GetLib();

  ICHECK(exec->primitive_map.empty() || lib.operator->())
      << "If the executable has declared primitive functions, the "
      << "generated kernel library must non-be null.";

  this->shared_state_->batched_func_arg_mode_ = exec->batched_func_arg_mode;
  this->shared_state_->prim_func_arg_access_mode_ = exec->prim_func_arg_access_mode;

  for (const auto& it : shared_state_->exec_->primitive_map) {
    const auto& packed_name = it.first;
    auto packed_index = static_cast<size_t>(it.second);
    if (shared_state_->packed_funcs_.size() <= packed_index) {
      shared_state_->packed_funcs_.resize(packed_index + 1);
      shared_state_->outputs_start.resize(packed_index + 1);
      shared_state_->inouts_start.resize(packed_index + 1);
      shared_state_->args_end.resize(packed_index + 1);
    }
    tvm::runtime::PackedFunc pf = lib.GetFunction(packed_name, /*query_imports=*/true);

    ICHECK(pf != nullptr) << "Cannot find function in module: " << packed_name;
    shared_state_->packed_funcs_[packed_index] = pf;

    auto& arg_access_modes = shared_state_->prim_func_arg_access_mode_[packed_index];

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

    shared_state_->outputs_start[packed_index] = num_inputs;
    shared_state_->inouts_start[packed_index] = num_inputs + num_outputs;
    shared_state_->args_end[packed_index] = arg_access_modes.size();

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

    bool print = true;
    if (print) {
      if (batched_execution_) {
        std::cout << "[VM] Fun " << packed_index << " " << packed_name;
        if (!IsBatchedName(packed_name)) {
          std::cout << " " << shared_state_->batched_funcs_[packed_index];
        }
      } else {
        std::cout << "[VM] Fun " << packed_index << " " << packed_name;
      }

      if (coarsened_execution_) {
        std::cout << "  ScMode: [";
        for (size_t i = 0; i < shared_state_->batched_func_arg_mode_[packed_index].size(); ++i) {
          std::cout << shared_state_->batched_func_arg_mode_[packed_index][i] << " ";
        }
        std::cout << "]";
      }
      std::cout << "   AccMode: [";
      for (size_t i = 0; i < shared_state_->prim_func_arg_access_mode_[packed_index].size(); ++i) {
        std::cout << shared_state_->prim_func_arg_access_mode_[packed_index][i] << " ";
      }
      std::cout << "] " << shared_state_->outputs_start[packed_index] << " "
                << shared_state_->inouts_start[packed_index] << std::endl;
    }
  }

  for (size_t i = 0; i < shared_state_->packed_funcs_.size(); ++i) {
    ICHECK(shared_state_->packed_funcs_[i] != nullptr)
        << "Packed function " << i << " is not initialized";
  }
}

void VirtualMachine::Init(const std::vector<Device>& physical_devices,
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

inline void VirtualMachine::WriteRegister(Index r, const ObjectRef& val) {
  frames_.back().register_file[r] = val;
}

ObjectRef VirtualMachine::ReadRegister(Index r) const { return frames_.back().register_file[r]; }

int64_t VirtualMachine::LoadScalarInt(Index r) const {
  int64_t result = 0;
  const auto& obj = Downcast<NDArray>(ReadRegister(r));

  NDArray array =
      Downcast<NDArray>(CopyTo(obj, GetDevice(shared_state_->exec_->host_device_index)));

  switch (array->dtype.bits) {
    case 1: {
      result = reinterpret_cast<bool*>(array->data)[0];
      break;
    }
    case 8: {
      result = reinterpret_cast<int8_t*>(array->data)[0];
      break;
    }
    case 16: {
      result = reinterpret_cast<int16_t*>(array->data)[0];
      break;
    }
    case 32: {
      result = reinterpret_cast<int32_t*>(array->data)[0];
      break;
    }
    case 64: {
      result = reinterpret_cast<int64_t*>(array->data)[0];
      break;
    }
    default:
      LOG(FATAL) << "Unknown scalar int type: " << DLDataType2String(array->dtype);
  }
  return result;
}

void VirtualMachine::RunLoop() {
  ICHECK(this->shared_state_->exec_);
  ICHECK(this->code_);
  pc_ = 0;
  Index frame_start = frames_.size();
  while (true) {
    if (RunOneIteration(frame_start)) {
      break;
    }
  }
}

bool VirtualMachine::RunOneIteration(int frame_start) {
  auto const& instr = code_[this->pc_];
  VLOG(2) << "Executing(" << pc_ << "): " << instr;
  // std::cout << "Executing(" << pc_ << "): " << instr << std::endl;

  switch (instr.op) {
    case Opcode::Move: {
      ObjectRef from_obj;
      from_obj = ReadRegister(instr.from);
      WriteRegister(instr.dst, from_obj);
      pc_++;
      return false;
    }
    case Opcode::Fatal: {
      throw std::runtime_error("VM encountered fatal error");
    }
    case Opcode::LoadConst: {
      bool is_not_cached =
          shared_state_->const_pool_.size() <= static_cast<size_t>(instr.const_index) ||
          !shared_state_->const_pool_[instr.const_index].defined();
      if (is_not_cached) {
        OpStartHook(instr);
      }
      auto constant_obj = shared_state_->exec_->constants[instr.const_index];
      // We cache the allocated object in the constant pool. To measure, the
      // first iteration will set the pool up. The other iterations will
      // directly reuse the allocated objects.
      if (shared_state_->const_pool_.size() <= static_cast<size_t>(instr.const_index)) {
        shared_state_->const_pool_.resize(instr.const_index + 1);
      }

      if (!shared_state_->const_pool_[instr.const_index].defined()) {
        Device dev = GetDevice(shared_state_->exec_->const_device_indexes[instr.const_index]);
        shared_state_->const_pool_[instr.const_index] = CopyTo(constant_obj, dev);
      }
      WriteRegister(instr.dst, shared_state_->const_pool_[instr.const_index]);
      if (is_not_cached) {
        OpStopHook();
      }
      pc_++;
      return false;
    }
    case Opcode::LoadConsti: {
      // auto tensor =
      // NDArray::Empty({1}, {kDLInt, 64, 1}, GetDevice(shared_state_->exec_->host_device_index));
      // reinterpret_cast<int64_t*>(tensor->data)[0] = instr.load_consti.val;

      auto tensor =
          NDArray::Empty({}, {kDLInt, 32, 1}, GetDevice(shared_state_->exec_->host_device_index));
      reinterpret_cast<int32_t*>(tensor->data)[0] = instr.load_consti.val;

      WriteRegister(instr.dst, tensor);
      pc_++;
      return false;
    }
    case Opcode::Invoke: {
      if (instr.func_index == DB_RANDOM_UNIFORM_INDEX) {
        ICHECK_EQ(instr.num_args, 2);
        auto lo = LoadScalarInt(instr.invoke_args_registers[0]);
        auto hi = LoadScalarInt(instr.invoke_args_registers[1]);
        ICHECK_GE(hi, lo);
        auto output = GetRandom(lo, hi);

        auto tensor =
            NDArray::Empty({}, {kDLInt, 32, 1}, GetDevice(shared_state_->exec_->host_device_index));
        reinterpret_cast<int32_t*>(tensor->data)[0] = output;
        auto output_reg = instr.dst;
        WriteRegister(output_reg, tensor);
        pc_++;
      } else if (instr.func_index == DB_PHASE_CHANGE_INDEX) {
        pc_++;
      } else {
        std::vector<ObjectRef> args;
        for (Index i = 0; i < instr.num_args; ++i) {
          args.push_back(ReadRegister(instr.invoke_args_registers[i]));
        }
        InvokeGlobal(shared_state_->exec_->functions[instr.func_index], args, 0);
        frames_.back().caller_return_register = instr.dst;
      }
      return false;
    }
    case Opcode::InvokePacked: {
      VLOG(2) << "InvokedPacked " << instr.packed_index << " arity=" << instr.arity;

      // std::cout << vm_id_ << " InvokedPacked " << instr.packed_index << " arity=" <<
      // instr.arity << std::endl;
      ICHECK_LE(instr.packed_index, shared_state_->packed_funcs_.size());
      const auto& arity = instr.arity;
      std::vector<ObjectRef> args;
      for (Index i = 0; i < arity; ++i) {
        VLOG(2) << "arg" << i << " $" << instr.packed_args[i];
        // std::cout << "[VM] Reading register " << instr.packed_args[i] << std::endl;
        auto arg = ReadRegister(instr.packed_args[i]);
        ICHECK(arg.defined());
        args.push_back(arg);
      }

      // We no longer need to write the registers back, we write directly
      // through the registers mutably.
      InvokePacked(
          instr.packed_index, arity, instr.output_size, args,
          (batched_execution_ && (shared_state_->batched_funcs_[instr.packed_index] >= 0)));
      pc_++;
      return false;
    }
    case Opcode::InvokeClosure: {
      auto object = ReadRegister(instr.closure);
      const auto* closure = object.as<VMClosureObj>();
      ICHECK(closure);
      std::vector<ObjectRef> args;
      for (auto free_var : closure->free_vars) {
        args.push_back(free_var);
      }
      for (Index i = 0; i < instr.num_closure_args; ++i) {
        args.push_back(ReadRegister(instr.closure_args[i]));
      }
      InvokeGlobal(shared_state_->exec_->functions[closure->func_index], args, 0);
      frames_.back().caller_return_register = instr.dst;
      return false;
    }
    case Opcode::GetField: {
      auto object = ReadRegister(instr.object);
      const auto& tuple = Downcast<ADT>(object);
      auto field = tuple[instr.field_index];
      WriteRegister(instr.dst, field);
      pc_++;
      return false;
    }
    case Opcode::GetTag: {
      auto object = ReadRegister(instr.get_tag.object);
      const auto& adt = Downcast<ADT>(object);
      auto tag = adt.tag();
      auto tag_tensor =
          NDArray::Empty({1}, {kDLInt, 32, 1}, GetDevice(shared_state_->exec_->host_device_index));
      reinterpret_cast<int32_t*>(tag_tensor->data)[0] = tag;
      WriteRegister(instr.dst, tag_tensor);
      pc_++;
      return false;
    }
    case Opcode::Goto: {
      pc_ += instr.pc_offset;
      return false;
    }
    case Opcode::If: {
      int32_t test_val = LoadScalarInt(instr.if_op.test);
      int32_t target_val = LoadScalarInt(instr.if_op.target);

      // std::cout << "[VM] If " << test_val << " " << target_val << std::endl;

      if (test_val == target_val) {
        ICHECK_NE(instr.if_op.true_offset, 0);
        pc_ += instr.if_op.true_offset;
      } else {
        ICHECK_NE(instr.if_op.false_offset, 0);
        pc_ += instr.if_op.false_offset;
      }

      return false;
    }
    case Opcode::AllocTensor: {
      OpStartHook(instr);
      auto shape = std::vector<int64_t>(instr.alloc_tensor.ndim);

      // std::cout << "[VM] AllocTensor " << pc_ << std::endl;
      for (uint32_t i = 0; i < instr.alloc_tensor.ndim; ++i) {
        shape[i] = instr.alloc_tensor.shape[i];
        // std::cout << "[VM]   Shape1 " << shape[i] << std::endl;
      }

      auto storage_obj = ReadRegister(instr.alloc_tensor.storage);
      auto offset = LoadScalarInt(instr.alloc_tensor.offset);
      auto storage = Downcast<Storage>(storage_obj);
#if TVM_LOG_DEBUG
      std::ostringstream os;
      os << "AllocTensor: ";
      os << "offset=" << offset;
      os << ", shape=[";
      for (auto i : shape) {
        os << i << ",";
      }
      os << "]";
      os << ", dtype=" << DLDataType2String(instr.alloc_tensor.dtype);
      VLOG(2) << os.str();
#endif
      // for (uint32_t i = 0; i < shape.size(); ++i) {
      // std::cout << "[VM]   Shape2 " << shape[i] << std::endl;
      // }
      auto obj = storage->AllocNDArray(offset, shape, instr.alloc_tensor.dtype);

      WriteRegister(instr.dst, obj);
      OpStopHook();
      pc_++;
      return false;
    }
    case Opcode::AllocTensorReg: {
      OpStartHook(instr);
      Device cpu_dev = GetDevice(shared_state_->exec_->host_device_index);
      auto shape_obj = ReadRegister(instr.alloc_tensor_reg.shape_register);
      NDArray shape_tensor = Downcast<NDArray>(CopyTo(shape_obj, cpu_dev));
      auto shape = ToShape(shape_tensor);
      auto storage_obj = ReadRegister(instr.alloc_tensor_reg.storage);
      auto storage = Downcast<Storage>(storage_obj);
      auto offset = LoadScalarInt(instr.alloc_tensor.offset);
      auto obj = storage->AllocNDArray(offset, shape, instr.alloc_tensor_reg.dtype);

      WriteRegister(instr.dst, obj);
      OpStopHook();
      pc_++;
      return false;
    }
    case Opcode::AllocADT: {
      std::vector<ObjectRef> fields;
      for (Index i = 0; i < instr.num_fields; ++i) {
        fields.push_back(ReadRegister(instr.datatype_fields[i]));
      }
      ObjectRef obj = ADT(instr.constructor_tag, fields);
      WriteRegister(instr.dst, obj);
      pc_++;
      return false;
    }
    case Opcode::AllocClosure: {
      std::vector<ObjectRef> free_vars;
      for (Index i = 0; i < instr.num_freevar; i++) {
        free_vars.push_back(ReadRegister(instr.free_vars[i]));
      }
      WriteRegister(instr.dst, VMClosure(instr.func_index, free_vars));
      pc_++;
      return false;
    }
    case Opcode::AllocStorage: {
      OpStartHook(instr);
      auto size = LoadScalarInt(instr.alloc_storage.allocation_size);
      auto alignment = instr.alloc_storage.alignment;

      auto storage_obj = SimpleObjAllocator().make_object<StorageObj>();
      Allocator* allocator = GetAllocator(instr.alloc_storage.device_index);
      ICHECK(allocator) << "Did you forget to init the VirtualMachine with devices?";
      VLOG(2) << "AllocStorage: allocation_size=" << size << ", alignment=" << alignment
              << ", dtype_hint=" << DLDataType2String(instr.alloc_storage.dtype_hint)
              << ", device_index=" << instr.alloc_storage.device_index;

      storage_obj->buffer = allocator->Alloc(size, alignment, instr.alloc_storage.dtype_hint);
      Storage storage(storage_obj);
      WriteRegister(instr.dst, storage);
      OpStopHook();
      pc_++;
      return false;
    }
    case Opcode::ShapeOf: {
      auto input = ReadRegister(instr.shape_of.tensor);
      NDArray input_array = Downcast<NDArray>(input);
      int ndim = input_array->ndim;
      auto out_tensor = NDArray::Empty({ndim}, {kDLInt, 64, 1},
                                       GetDevice(shared_state_->exec_->host_device_index));
      for (int i = 0; i < ndim; ++i) {
        reinterpret_cast<int64_t*>(out_tensor->data)[i] = input_array->shape[i];
      }
      WriteRegister(instr.dst, out_tensor);
      pc_++;
      return false;
    }
    case Opcode::Ret: {
      // If we have hit the point from which we started
      // running, we should return to the caller breaking
      // the dispatch loop.
      return_register_ = ReadRegister(instr.result);
      auto caller_return_register = frames_.back().caller_return_register;

      if (PopFrame() == frame_start) {
        return true;
        // Otherwise we are just returning from a local call.
      } else {
        WriteRegister(caller_return_register, return_register_);
        return false;
      }
    }
    case Opcode::ReshapeTensor: {
      OpStartHook(instr);
      Device cpu_dev = GetDevice(shared_state_->exec_->host_device_index);
      auto tensor_obj = ReadRegister(instr.reshape_tensor.tensor);
      NDArray tensor_arr = Downcast<NDArray>(tensor_obj);
      // Read the shape from shape tensor
      auto shape_obj = ReadRegister(instr.reshape_tensor.newshape);
      NDArray shape_tensor = Downcast<NDArray>(CopyTo(shape_obj, cpu_dev));
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
      auto out_tensor = tensor_arr.CreateView(shape, tensor_arr->dtype);
      WriteRegister(instr.dst, out_tensor);
      OpStopHook();
      pc_++;
      return false;
    }
    case Opcode::DeviceCopy: {
      OpStartHook(instr);
      auto tensor_src = ReadRegister(instr.device_copy.src);
      NDArray src_data = Downcast<NDArray>(tensor_src);
      Device actual_src_dev = src_data->device;
      Device inst_src_dev = GetDevice(instr.device_copy.src_device_index);
      ICHECK_EQ(actual_src_dev.device_type, inst_src_dev.device_type);
      ICHECK_EQ(actual_src_dev.device_id, inst_src_dev.device_id);
      Device dst_dev = GetDevice(instr.device_copy.dst_device_index);

      NDArray dst_data = src_data.CopyTo(dst_dev);
      WriteRegister(instr.dst, dst_data);
      OpStopHook();
      pc_++;
      return false;
    }
    default:
      LOG(FATAL) << "Unknown instruction opcode: " << int(instr.op);
      return false;
  }
}

// ConcurrentVM ///////////////////////////////////////////////
inline VirtualMachine* GetVMFromModule(const Module& mod) {
  return static_cast<VirtualMachine*>(const_cast<ModuleNode*>(mod.operator->()));
}

void ConcurrentVirtualMachine::InvokeWrapper(std::string func_name, TVMRetValue* rv) {
  ICHECK(shared_state_->exec_) << "The executable is not created yet.";

  auto git = shared_state_->exec_->global_map.find(func_name);
  ICHECK(git != shared_state_->exec_->global_map.end())
      << "Cannot find function " << func_name << " in the executable";
  auto func = shared_state_->exec_->functions[git->second];

  for (auto& vm_mod : vms_) {
    auto* vm = GetVMFromModule(vm_mod);

    if (func.params.empty()) {
      vm->InvokeGlobal(func, {}, 0);
    } else {
      auto it = vm->inputs_.find(func_name);
      ICHECK(it != vm->inputs_.end()) << "Input has not been set for function " << func_name;
      const std::vector<ObjectRef>& func_args = it->second;
      vm->InvokeGlobal(func, func_args, 0);
    }
  }

  RunLoop();

  Array<ObjectRef> res;
  for (auto& vm_mod : vms_) {
    auto* vm = GetVMFromModule(vm_mod);
    res.push_back(vm->return_register_);
  }

  *rv = res;
}

void ConcurrentVirtualMachine::LoadExecutable(Executable* exec) {
  VirtualMachine::LoadExecutable(exec);
}

void ConcurrentVirtualMachine::Init(const std::vector<Device>& physical_devices,
                                    const std::vector<AllocatorType>& alloc_types) {
  VirtualMachine::Init(physical_devices, alloc_types);
}

void ConcurrentVirtualMachine::SetExecutionOptions(VMExecutionOptions options) {
  this->num_vms_ = options->batch_size;
  VirtualMachine::SetExecutionOptions(options);

  for (size_t i = 0; i < num_vms_; ++i) {
    auto vm_ptr = make_object<VirtualMachine>();
    vm_ptr->vm_id_ = i;
    auto module = Module(vm_ptr);
    vms_.push_back(module);
  }

  for (auto& vm : vms_) {
    GetVMFromModule(vm)->SetExecutionOptions(options);
    GetVMFromModule(vm)->batch_size_ = 1;
  }
}

void ConcurrentVirtualMachine::InitSharedState(bool pgo) {
  this->concurrent_vm_ = true;
  VirtualMachine::InitSharedState(pgo);
  for (auto& vm : vms_) {
    VirtualMachine* vm_ptr = GetVMFromModule(vm);
    vm_ptr->shared_state_ = this->shared_state_;
  }
}

DBVMExecutionState ConcurrentVirtualMachine::RunOneStage(size_t vm_id, VirtualMachine* vm,
                                                         int frame_start) {
  // std::cout << "[CVM] Running a stage" << std::endl;
  while (true) {
    // std::cout << "[CVM]  Next Instr " << static_cast<int>(instr.op) << " " << vm->pc_ <<
    // std::endl;
    if (vm->RunOneIteration(frame_start)) {
      return kExecutionEnd;
      // } else if (lazy_execution_) {
      //   if (false /* Some condition that checks if the value of a tensor
      // 		  is to be read */) {
      //     return kStageEnd;
      //   } else {
      //     continue;
      //   }
      // } else if (vm->code_[vm->pc_ - 1].op == Opcode::InvokePacked) {
    } else if ((vm->code_[vm->pc_].op == Opcode::Invoke &&
                vm->code_[vm->pc_].func_index == DB_RANDOM_UNIFORM_INDEX) ||
               vm->code_[vm->pc_].op == Opcode::If) {
      return kStageEnd;
    }
  }
}

void ConcurrentVirtualMachine::RunLoop() {
  std::vector<bool> mask(vms_.size(), true);
  int alive = vms_.size();
  int frame_start = GetVMFromModule(vms_[0])->frames_.size();

  while (alive > 0) {
    // Run one control flow stage for all VMs
    for (size_t i = 0; i < vms_.size(); ++i) {
      if (mask[i]) {
        auto& vm = vms_[i];
        auto stage_state = RunOneStage(i, GetVMFromModule(vm), frame_start);
        if (stage_state == kExecutionEnd) {
          alive--;
          mask[i] = false;
        }
      }
    }

    // std::cout << "[VM]  One stage done" << std::endl;

    // Run all tensor ops
    if (batched_execution_) {
      shared_state_->lazy_executor_.BatchedExecute(coarsened_execution_, !lazy_execution_);
    } else {
      shared_state_->lazy_executor_.Execute();
    }
  }
}

bool ConcurrentVirtualMachine::RunOneIteration(int frame_start) {
  ICHECK(false) << "This should not be called!";
  return true;
}

void ConcurrentVirtualMachine::InvokeGlobal(const VMFunction& func,
                                            const std::vector<ObjectRef>& args, const int offset) {
  for (auto& vm : vms_) {
    GetVMFromModule(vm)->InvokeGlobal(func, args, 0);
  }
}

ObjectRef ConcurrentVirtualMachine::Invoke(const VMFunction& func,
                                           const std::vector<ObjectRef>& args) {
  DLOG(INFO) << "Executing Function: " << std::endl << func;
  for (int i = 0; i < static_cast<int>(shared_state_->devices_.size()); ++i) {
    DLOG(INFO) << "Device " << i << " has device type " << shared_state_->devices_[i].device_type
               << " and device id " << shared_state_->devices_[i].device_id
               << (i == shared_state_->exec_->host_device_index ? " (using as host device)" : "");
  }

  // std::cout << "[VM] Executing Function: " << func.name << std::endl;

  InvokeGlobal(func, args, 0);
  RunLoop();

  if (lazy_execution_) {
    if (batched_execution_) {
      shared_state_->lazy_executor_.BatchedExecute(true, coarsened_execution_);
    } else {
      shared_state_->lazy_executor_.Execute();
    }
  }

  return return_register_;
}

void ConcurrentVirtualMachine::SetInput(std::string name, TVMArgs args, int offset, int batch_size,
                                        int num_args) {
  ICHECK_EQ(num_args % batch_size, 0);
  int per_vm = num_args / batch_size;
  for (auto& vm : vms_) {
    GetVMFromModule(vm)->SetInput(name, args, offset, 1, per_vm);
    offset += per_vm;
  }
}

runtime::Module CreateVirtualMachine(Executable* exec) {
  auto vm = make_object<VirtualMachine>();
  vm->LoadExecutable(exec);
  return runtime::Module(vm);
}

TVM_REGISTER_GLOBAL("runtime._VirtualMachine").set_body([](TVMArgs args, TVMRetValue* rv) {
  runtime::Module mod = args[0];
  auto* exec = dynamic_cast<Executable*>(mod.operator->());
  ICHECK(exec) << "The virtual machine executable has not been defined yet.";
  *rv = CreateVirtualMachine(exec);
});

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
