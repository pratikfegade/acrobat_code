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
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace runtime {
namespace vm {

enum FiberState { kYield = 0, kEnd = 1, kResume = 2, kPhaseWaiting = 3, kChildrenWaiting = 4 };

typedef boost::fibers::buffered_channel<FiberState> channel_t;
typedef boost::fibers::fiber fiber_t;
typedef boost::fibers::barrier barrier_t;
typedef boost::fibers::fiber::id fiber_id_t;

#define MAX_FIBER_COUNT 128

class FiberRuntime {
 public:
  FiberRuntime(int num_fibers)
      : num_fibers_(num_fibers),
        orig_num_fibers_(num_fibers),
        fibers_(num_fibers),
        alive_(num_fibers, true),
        phase_waiting_(num_fibers, false),
        children_waiting_(num_fibers, false),
        parents_(num_fibers, -1),
        alive_num_(num_fibers),
        phase_waiting_num_(0),
        sync_next_(false) {
    for (int i = 0; i < num_fibers; ++i) {
      start_channels_.push_back(new channel_t(2));
      stop_channels_.push_back(new channel_t(2));
    }
  }

  template <typename Fn_t>
  fiber_t* CreateFiber(int parent_id, Fn_t fn) {
    int child_id = num_fibers_;
    auto child_fiber = new fiber_t(std::move(fn));
    start_channels_.push_back(new channel_t(2));
    stop_channels_.push_back(new channel_t(2));
    fibers_.push_back(child_fiber);
    alive_.push_back(true);
    phase_waiting_.push_back(false);
    children_waiting_.push_back(false);
    alive_num_++;
    num_fibers_++;
    // std::cout << "[" << parent_id << "] New fiber created " << child_id << std::endl;
    children_[parent_id].insert(child_id);
    parents_.push_back(parent_id);
    return child_fiber;
  }

  bool CanCreateNewFiber() { return num_fibers_ < MAX_FIBER_COUNT; }

  int NewFiberID() { return num_fibers_; }

  inline void WorkerWaitInternal(int idx, FiberState state) {
    stop_channels_[idx]->push(state);
    FiberState dummy;
    start_channels_[idx]->pop(dummy);
  }

  void WorkerYield(int idx, bool sync_next = true) {
    sync_next_ = sync_next_ || sync_next;
    // std::cout << "[" << idx << "] Worker yielding" << std::endl;
    WorkerWaitInternal(idx, kYield);
  }

  void WorkerEnd(int idx) {
    // std::cout << "[" << idx << "] Worker ending" << std::endl;
    stop_channels_[idx]->push(kEnd);
    // std::cout << "[" << idx << "] Worker end sent" << std::endl;
  }

  void WorkerPhaseBarrierWait(int idx) {
    // std::cout << "[" << idx << "] Worker phase waiting" << std::endl;
    WorkerWaitInternal(idx, kPhaseWaiting);
  }

  void WorkerChildrenWait(int idx) {
    // std::cout << "[" << idx << "] Worker children waiting" << std::endl;
    WorkerWaitInternal(idx, kChildrenWaiting);
  }

  void MainWaitForWorkers() {
    // std::cout << "[M] Main waiting" << std::endl;
    for (int i = 0; i < num_fibers_; ++i) {
      if (alive_[i] && !phase_waiting_[i] && !children_waiting_[i]) {
        FiberState state;
        stop_channels_[i]->pop(state);
        if (state == kEnd) {
          alive_[i] = false;
          alive_num_--;
          fibers_[i]->join();
          fibers_[i] = nullptr;
          int parent = parents_[i];
          if (parent >= 0) {
            children_[parent].erase(i);
            parents_[i] = -1;
            // std::cout << "[M] Worker ended " << i << ". Parent children " << parent << " "
            // << children_.at(parent).size() << std::endl;
          }
        } else if (state == kPhaseWaiting) {
          phase_waiting_[i] = true;
          phase_waiting_num_++;
        } else if (state == kChildrenWaiting) {
          children_waiting_[i] = true;
        }
      }
    }
  }

  void MainResumeWorkers() {
    // std::cout << "[M] Main resuming workers" << std::endl;
    if (phase_waiting_num_ == alive_num_) {
      for (int i = 0; i < num_fibers_; ++i) {
        if (alive_[i]) {
          phase_waiting_[i] = false;
          // std::cout << "[M]  Resume1 " << i << std::endl;
          start_channels_[i]->push(kResume);
        }
      }
      phase_waiting_num_ = alive_num_;
    } else {
      for (int i = 0; i < num_fibers_; ++i) {
        if (alive_[i] && !phase_waiting_[i]) {
          if (children_waiting_[i] && children_.at(i).empty()) {
            // std::cout << "[M]  Resume2 " << i << std::endl;
            start_channels_[i]->push(kResume);
            children_waiting_[i] = false;
          } else if (!children_waiting_[i]) {
            // std::cout << "[M]  Resume3 " << i << std::endl;
            start_channels_[i]->push(kResume);
          }
        }
      }
    }
    sync_next_ = false;
  }

  void AddFiber(int idx, fiber_t* fiber) { fibers_[idx] = fiber; }

  bool ContinueExecution() {
    // std::cout << "Alive " << alive_num_ << std::endl;
    return alive_num_ > 0;
  }

  bool IsAlive(int idx) { return alive_[idx]; }

  bool PerformSync() { return sync_next_; }

  void MainEndFiberExecution() {
    // for (int i = 0; i < orig_num_fibers_; ++i) {
    // fibers_[i]->join();
    // }
  }

  inline static void Init(int num_fibers) {
    if (instance_) {
      delete instance_;
    }
    instance_ = new FiberRuntime(num_fibers);
  }

  inline static FiberRuntime& Current() { return *instance_; }

  int orig_num_fibers_;
  int num_fibers_;
  std::vector<fiber_t*> fibers_;
  std::vector<channel_t*> start_channels_;
  std::vector<channel_t*> stop_channels_;
  std::vector<bool> alive_;
  std::vector<bool> phase_waiting_;
  std::vector<bool> children_waiting_;
  std::unordered_map<int, std::unordered_set<int>> children_;
  std::vector<int> parents_;
  int alive_num_;
  int phase_waiting_num_;
  bool sync_next_;

  static FiberRuntime* instance_;
};

// template <typename RetType>
// class JoinableTaskWrapper {
//  public:
//   JoinableTaskWrapper(const std::function<std::pair<RetType, int>()>& task,
//                       const int parent_fiber_id) {
//     boost::fibers::packaged_task<std::pair<RetType, int>()> packaged_task(std::move(task));
//     future_ = packaged_task.get_future();
//     fiber_ = FiberRuntime::Current().CreateFiber(parent_fiber_id, std::move(packaged_task));
//   }

//   inline void Wait() { fiber_->join(); }

//   inline std::pair<RetType, int> GetResultAndDepth() { return future_.get(); }

//  private:
//   boost::fibers::future<std::pair<RetType, int>> future_;
//   fiber_t* fiber_;
// };

template <typename RetType>
class JoinableTaskWrapper {
 public:
  JoinableTaskWrapper(const std::function<std::pair<RetType, int>()>& task,
                      const int parent_fiber_id) {
    boost::fibers::packaged_task<std::pair<RetType, int>()> packaged_task(std::move(task));
    future_ = packaged_task.get_future();
    fiber_ = FiberRuntime::Current().CreateFiber(parent_fiber_id, std::move(packaged_task));
  }

  inline void Wait() { fiber_->join(); }

  inline std::pair<RetType, int> GetResultAndDepth() { return future_.get(); }

 private:
  boost::fibers::future<std::pair<RetType, int>> future_;
  fiber_t* fiber_;
};

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
#endif
