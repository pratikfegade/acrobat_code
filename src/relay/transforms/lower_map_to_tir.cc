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
 * \file src/relay/transforms/fuse_ops.cc
 *
 * \brief This is a backend-aware optimization pass.
 *   Fuse necessary ops into a single one.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/vm/dynamic_batching.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include "../../support/arena.h"
#include "../op/annotation/annotation.h"
#include "../op/vm/vm.h"
#include "./expr_subst.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {

class MapLowerer : public ExprMutator {
  Expr VisitExpr_(const CallNode* op) override {
    if (auto callee_ptr = op->op.as<FunctionNode>()) {
      auto callee_name = callee_ptr->GetAttr<String>(tvm::attr::kGlobalSymbol);
      std::cout << "[LMT] Call " << GetRef<Expr>(op) << std::endl;
      std::cout << "[LMT]   Callee " << callee_name << std::endl;
    }
    return ExprMutator::VisitExpr_(op);
  }
};

IRModule LowerMap(IRModule& mod, bool batched_execution, bool scattered_kernels) {
  tvm::Map<GlobalVar, Function> updates;
  tvm::Map<GlobalVar, tir::PrimFunc> new_prim_funcs;

  auto funcs = mod->functions;
  for (const auto& it : funcs) {
    if (const auto* n = it.second.as<FunctionNode>()) {
      ICHECK_EQ(FreeVars(it.second).size(), 0);
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
      Function func = GetRef<Function>(n);

      MapLowerer()(func);
    }
  }

  for (auto pair : updates) {
    mod->Add(pair.first, pair.second, true);
  }

  for (auto pair : new_prim_funcs) {
    mod->Add(pair.first, pair.second, true);
  }

  return mod;
}

namespace transform {
Pass LowerMapToTIR(bool batched_execution, bool scattered_kernels) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return LowerMap(m, batched_execution, scattered_kernels); };
  return CreateModulePass(pass_func, 0, "LowerMapToTIR", {});
}

TVM_REGISTER_GLOBAL("relay._transform.LowerMapToTIR").set_body_typed(FuseOps);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
