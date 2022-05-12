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

#ifndef TVM_RELAY_BACKEND_MODEL_PARAMETER_TAINT_ANALYSIS_H_
#define TVM_RELAY_BACKEND_MODEL_PARAMETER_TAINT_ANALYSIS_H_

#include <tvm/ir/attrs.h>
#include <tvm/ir/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>

namespace tvm {
namespace relay {

namespace tec {
IRModule ModelParameterTaintAnalysis(IRModule& mod, bool repeat = true);

inline tvm::transform::Pass ModelParameterTaintAnalysisPass(bool repeat) {
  runtime::TypedPackedFunc<IRModule(IRModule, tvm::transform::PassContext)> pass_func =
      [=](IRModule module, tvm::transform::PassContext ctx) {
        return ModelParameterTaintAnalysis(module, repeat);
      };

  return tvm::transform::CreateModulePass(pass_func, 0, "ModelParameterTaintAnalysisPass",
                                          {"InferType"});
}

}  // namespace tec
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_MODEL_PARAMETER_TAINT_ANALYSIS_H_
