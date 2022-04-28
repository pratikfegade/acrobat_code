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
 *
 * \file eliminate_common_subexpr.cc
 * \brief Combine common subexpressions.
 *
 * This is an optimization pass that eliminates common subexpressions. During the pass, it tries
 * to replace an expression with a previously appeared expression with the same input and
 * attributes. The fskip callback argument allows us to skip specific expressions.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/function.h>

#include <unordered_map>

#include "pattern_utils.h"

namespace tvm {
namespace relay {

class FunctionNamer : public ExprMutator {
 public:
  Expr VisitExpr_(const FunctionNode* op) override {
    auto fn = Downcast<Function>(ExprMutator::VisitExpr_(op));
    auto name = fn->GetAttr<String>(tir::attr::kDBFunctionName);
    if (!name) {
      fn = WithAttr(fn, tir::attr::kDBFunctionName, String("function" + std::to_string(ctr_++)));
    }
    return fn;
  }

 private:
  int ctr_{0};
};

IRModule NameFunctions(IRModule& mod) {
  tvm::Map<GlobalVar, BaseFunc> new_funcs;
  auto funcs = mod->functions;
  FunctionNamer function_namer;
  for (const auto& it : funcs) {
    auto base_func = it.second;
    if (base_func.as<FunctionNode>()) {
      auto func = Downcast<Function>(base_func);
      func = WithAttr(func, tir::attr::kDBFunctionName, String(it.first->name_hint));
      func = Downcast<Function>(function_namer(func));
      new_funcs.Set(it.first, func);
    }
  }

  for (auto pair : new_funcs) {
    mod->Add(pair.first, pair.second, true);
  }
  return mod;
}

namespace transform {

Pass NameAllFunctions() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return NameFunctions(m); };
  return CreateModulePass(pass_func, 0, "NameAllFunctions", {});
}

TVM_REGISTER_GLOBAL("relay._transform.NameAllFunctions").set_body_typed(NameAllFunctions);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
