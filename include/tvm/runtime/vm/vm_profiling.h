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
#ifndef TVM_RUNTIME_VM_VM_PROFILING_H_
#define TVM_RUNTIME_VM_VM_PROFILING_H_

#include <tvm/runtime/profiling.h>

namespace tvm {
namespace runtime {
namespace vm {

class VMDBProfiler {
 public:
  VMDBProfiler(std::vector<Device> devs) : profiler_(devs, {}), devices_(devs) {}
  profiling::Profiler profiler_;
  std::vector<Device> devices_;
  static VMDBProfiler* instance_;

  static inline VMDBProfiler& Current() { return *instance_; }

  static inline void Init(std::vector<Device> devices) { instance_ = new VMDBProfiler(devices); }

  static inline bool DoProfile() { return instance_ != nullptr; }

  static inline void ProfileStart() { instance_->profiler_.Start(); }

  static inline void ProfileStop() { instance_->profiler_.Stop(); }

  static inline void ProfileHostStartCall(std::string call) {
    instance_->profiler_.StartCall(call, VMDBProfiler::GetHost());
  }

  static inline void ProfileHostStopCall() { instance_->profiler_.StopCall(); }

  static inline void ProfileDeviceStartCall(std::string call) {
    instance_->profiler_.StartCall(call, VMDBProfiler::GetDevice());
  }

  static inline void ProfileDeviceStopCall() { instance_->profiler_.StopCall(); }

  static inline std::string GetReport() {
    std::stringstream ss;
    ss << instance_->profiler_.Report(true, true)->AsTable();
    return ss.str();
  }

  static inline Device& GetHost() { return instance_->devices_[0]; }

  static inline Device& GetDevice() {
    return instance_->devices_.size() > 1 ? instance_->devices_[1] : instance_->devices_[0];
  }
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_VM_PROFILING_H_
