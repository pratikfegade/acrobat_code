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
#include <tvm/runtime/vm/lazy_executor.h>
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
void OpNode::Execute() {
  // std::cout << "Vec " << args_.data() << " " << args_[0].defined() << std::endl;
  // InvokePackedFn(func_, arg_count_, output_size_, args_);
}

void LazyExecutor::AddPackedCall(const PackedFunc& func, const Index arg_count,
                                 const Index output_size, const std::vector<ObjectRef> args) {
  std::vector<ObjectRef> args_copy;
  for (auto& arg : args) {
    ICHECK(arg.use_count() > 0);
    args_copy.push_back(arg);
  }
  OpNode node(func, arg_count, output_size, args_copy);
  nodes_.push_back(node);
}

void LazyExecutor::Execute() {
  std::cout << "Executing nodes" << std::endl;
  for (OpNode& node : nodes_) {
    node.Execute();
  }
  nodes_.clear();
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
