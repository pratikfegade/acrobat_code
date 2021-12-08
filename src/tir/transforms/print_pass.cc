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
 * \file storage_flatten.cc
 * \brief Flattens storage from multi-dimensional array to 1D buffer access
 */
// The pass definition originates from Halide pipeline.

#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

PrimFunc PrintCurrentIRFn(PrimFunc func, String previous_pass) {
  std::cout << "[PRINT] IR after " << previous_pass << std::endl;
  std::cout << func << std::endl;
  return func;
}

namespace transform {

Pass PrintCurrentIR(String previous_pass) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return PrintCurrentIRFn(std::move(f), previous_pass);
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.PrintCurrentIR", {});
}

TVM_REGISTER_GLOBAL("tir.transform.PrintCurrentIR").set_body_typed(PrintCurrentIR);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
