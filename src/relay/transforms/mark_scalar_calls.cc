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
 * \file src/relay/transforms/annotate_target.cc
 * \brief Wraps an expr with compiler_begin and compiler_end to indicate that
 * this expr should be handled by the external compiler.
 */

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include "../op/random/db_random.h"
#include "pass_utils.h"

namespace tvm {
namespace relay {

Function MarkScalarCallsInFunc(const Function& func) {
  class Marker : public ExprMutator {
   public:
    Expr VisitExpr_(const CallNode* op) {
      auto on_device_props = GetOnDeviceProps(op);
      if (on_device_props.body.defined()) {
        return VisitExpr(on_device_props.body);
      } else if (op->op == GetDBRandomUniformOp()) {
        return ExprMutator::VisitExpr_(op);
      }
      auto mutated = ExprMutator::VisitExpr_(op);
      auto mutated_node = mutated.as<CallNode>();
      auto callee_op_node = op->op.as<OpNode>();
      if (callee_op_node && IsOpOnScalars(mutated_node)) {
        // std::cout << "[SCA] Marking scalar op " << mutated << std::endl;
        auto new_attrs =
            DictAttrs::WithAttr(mutated_node->attrs, tir::attr::kDBScalarCall, mutated);
        auto ret = Call(mutated_node->op, mutated_node->args, new_attrs, mutated_node->type_args,
                        mutated_node->span);
        ret->checked_type_ = op->checked_type_;
        return ret;
      }
      return mutated;
    }
  };

  Marker marker;
  auto mutated = Function(func->params, marker(func->body), func->ret_type, func->type_params,
                          func->attrs, func->span);
  mutated->checked_type_ = func->checked_type_;
  return mutated;
}

namespace transform {

Pass MarkScalarCalls() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return MarkScalarCallsInFunc(f); };
  auto func_pass = CreateFunctionPass(pass_func, 0, "MarkScalarCallsFunc", {"InferType"});
  return transform::Sequential({func_pass, InferType()}, "MarkScalarCalls");
}

TVM_REGISTER_GLOBAL("relay._transform.MarkScalarCalls").set_body_typed(MarkScalarCalls);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
