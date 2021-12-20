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
 * \file constant_folding.cc
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

#include "../op/memory/on_device.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {
namespace transform {

Pass PrintCurrentIR(String previous_pass, bool clean_up) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext pc) {
    std::cout << "[PRINT] IR after " << previous_pass << std::endl;
    if (clean_up) {
      for (auto kv : m->functions) {
        if (kv.second->HasNonzeroAttr(attr::kPrimitive) || !kv.second.as<FunctionNode>()) {
          // std::cout << kv.first << ": " << kv.second << std::endl;
        } else {
          // std::cout << " " << kv.second << std::endl;
          std::cout << kv.first << ": " << RemoveOnDeviceCalls(kv.second) << std::endl;
        }
      }
    } else {
      for (auto kv : m->functions) {
        if (kv.second->HasNonzeroAttr(attr::kPrimitive) || !kv.second.as<FunctionNode>()) {
          // std::cout << kv.first << ": " << kv.second << std::endl;
        } else {
          // std::cout << " " << kv.second << std::endl;
          std::cout << kv.first << ": " << kv.second << std::endl;
        }
      }
    }
    return m;
  };
  return CreateModulePass(pass_func, 1, "PrintCurrentIR", {});
}

TVM_REGISTER_GLOBAL("relay._transform.PrintCurrentIR").set_body_typed(PrintCurrentIR);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
