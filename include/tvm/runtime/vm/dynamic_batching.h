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
 * \file tvm/runtime/vm/executable.h
 * \brief The Relay virtual machine executable.
 */
#ifndef TVM_RUNTIME_VM_DYNAMIC_BATCHING_H_
#define TVM_RUNTIME_VM_DYNAMIC_BATCHING_H_

namespace tvm {
namespace runtime {
namespace vm {

/*!
 * \brief Dynamic batching argument mode.
 *
 * For a batched version of a kernel, this specifies which arguments
 * are to be (logically or physically) concatenated, which ones can be
 * reused and which ones are to be ignored.
 */
enum DBBatchedArgMode {
  kIgnore = 0,
  kReuse = 1,
  kScatter = 2,
  kConcat = 3,
  kContiguous = 4,
};

/*!
 * \brief Argument access mode.
 *
 * This specifies whether an argument tensor is read/written/read and
 * written, or not used, in that order. This is used to determine
 * dependences between operators during dynamic batching.
 */
enum DBArgAccessMode {
  kInput = 0,
  kOutput = 1,
  kInputOutput = 2,
  kUnused = 3,
};

#define DB_BATCHED_SUFFIX "_batched"

inline std::string GetBatchedName(std::string name) { return name + DB_BATCHED_SUFFIX; }

inline bool IsBatchedName(std::string name) {
  std::string suffix = DB_BATCHED_SUFFIX;
  if (suffix.size() <= name.size()) {
    return std::equal(suffix.rbegin(), suffix.rend(), name.rbegin());
  }
  return false;
}

// Device index during execution
#define CPU_INDEX 0
#define GPU_INDEX 1
#define MAX_PROGRAM_PHASES 8
#define REDUCE_SUM_FUNC_INDEX (1 << 16)
#define DB_RANDOM_UNIFORM_INDEX ((1 << 16) + 1)
#define DB_SET_PHASE_INDEX ((1 << 16) + 2)

// #define DEBUG_CHECKS
#define DB_PROFILING

}  // namespace vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_VM_DYNAMIC_BATCHING_H_
