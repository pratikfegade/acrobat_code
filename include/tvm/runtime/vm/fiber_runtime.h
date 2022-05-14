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

#ifndef TVM_RUNTIME_VM_FIBER_RUNTIME_H_
#define TVM_RUNTIME_VM_FIBER_RUNTIME_H_

#include <boost/fiber/all.hpp>

namespace tvm {
namespace runtime {
namespace vm {

enum FiberState { kYield = 0, kEnd = 1, kResume = 2 };

typedef boost::fibers::unbuffered_channel<FiberState> channel_t;
typedef boost::fibers::fiber fiber_t;
typedef boost::fibers::fiber::id fiber_id_t;

class FiberRuntime {
 public:
  FiberRuntime(int num_fibers)
      : num_fibers_(num_fibers),
        fibers_(num_fibers),
        start_channels_(num_fibers),
        stop_channels_(num_fibers),
        alive_(num_fibers, true),
        alive_num_(num_fibers),
        sync_next_(false) {}

  void WorkerYield(int idx, bool sync_next = true) {
    /* std::cout << "[" << idx << "] Worker yielding" << std::endl; */
    stop_channels_[idx].push(kYield);
    sync_next_ = sync_next_ || sync_next;
    FiberState state;
    start_channels_[idx].pop(state);
  }

  void WorkerEnd(int idx) {
    /* std::cout << "[" << idx << "] Worker ending" << std::endl; */
    stop_channels_[idx].push(kEnd);
  }

  void MainWaitForWorkers() {
    // std::cout << "[M] Main waiting" << std::endl;
    for (int i = 0; i < num_fibers_; ++i) {
      if (alive_[i]) {
        FiberState state;
        stop_channels_[i].pop(state);
        if (state == kEnd) {
          alive_[i] = false;
          alive_num_--;
        }
      }
    }
  }

  void MainResumeWorkers() {
    // std::cout << "[M] Main resuming workers" << std::endl;
    for (int i = 0; i < num_fibers_; ++i) {
      if (alive_[i]) {
        start_channels_[i].push(kResume);
      }
    }
    sync_next_ = false;
  }

  void AddFiber(int idx, fiber_t* fiber) { fibers_[idx] = fiber; }

  bool ContinueExecution() { return alive_num_ > 0; }

  bool IsAlive(int idx) { return alive_[idx]; }

  bool PerformSync() { return sync_next_; }

  void MainEndFiberExecution() {
    for (auto fiber : fibers_) {
      fiber->join();
    }
  }

  inline static void Init(int num_fibers) {
    if (instance_) {
      delete instance_;
    }
    instance_ = new FiberRuntime(num_fibers);
  }

  inline static FiberRuntime& Current() { return *instance_; }

  int num_fibers_;
  std::vector<fiber_t*> fibers_;
  std::vector<channel_t> start_channels_;
  std::vector<channel_t> stop_channels_;
  std::vector<bool> alive_;
  int alive_num_;
  bool sync_next_;

  static FiberRuntime* instance_;
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
#endif
