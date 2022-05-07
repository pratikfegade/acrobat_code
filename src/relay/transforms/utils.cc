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
#include <tvm/relay/transform.h>
#include <tvm/runtime/vm/dynamic_batching.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include "./pass_utils.h"

namespace tvm {
namespace relay {

class LetLifter : public ExprMutator {
 public:
  Expr VisitExpr_(const LetNode* outer_let) {
    auto outer_var = outer_let->var;
    auto outer_value = VisitExpr(outer_let->value);
    auto outer_body = VisitExpr(outer_let->body);

    auto on_device_props = GetOnDeviceProps(outer_value);
    if (on_device_props.body.defined()) {
      outer_value = on_device_props.body;
    }

    if (auto inner_let = outer_value.as<LetNode>()) {
      auto inner_var = inner_let->var;
      auto inner_value = inner_let->value;
      auto inner_body = inner_let->body;

      Expr ret = NullValue<Expr>();
      if (StructuralEqual()(outer_var, outer_body)) {
        ret = Let(inner_var, WrapOnDevice(on_device_props, inner_value),
                  WrapOnDevice(on_device_props, inner_body));
        std::cout << "[LL] Visiting\n " << DebugPrint(GetRef<Expr>(outer_let)) << std::endl;
        std::cout << "[LL]   Returning1\n " << DebugPrint(ret) << "\n\n" << std::endl;
      } else {
        ret = Let(inner_var, WrapOnDevice(on_device_props, inner_value),
                  Let(outer_var, WrapOnDevice(on_device_props, inner_body), outer_body));
        std::cout << "[LL] Visiting\n " << DebugPrint(GetRef<Expr>(outer_let)) << std::endl;
        std::cout << "[LL]   Returning2\n " << DebugPrint(ret) << "\n\n" << std::endl;
      }
      return ret;
    } else {
      return Let(outer_var, WrapOnDevice(on_device_props, outer_value), outer_body);
    }
  }

  Expr WrapOnDevice(const OnDeviceProps& props, const Expr& e) {
    if (props.body.defined()) {
      return OnDevice(e, props.se_scope, props.constrain_result, props.constrain_body);
    } else {
      return e;
    }
  }
};

Expr LiftLetsOutOfValues(const Expr& expr) { return LetLifter()(expr); }

Expr RemoveOnDeviceCalls(const Expr& expr) {
  class OnDeviceRemover : public ExprMutator {
   public:
    Expr VisitExpr_(const CallNode* call) override {
      return IgnoreOnDevice(ExprMutator::VisitExpr_(call));
    }
  };
  return OnDeviceRemover()(expr);
}

Type GetVarType(relay::Var var) {
  if (var->checked_type_.defined()) {
    return var->checked_type();
  } else {
    ICHECK(var->type_annotation.defined());
    return var->type_annotation;
  }
}

}  // namespace relay
}  // namespace tvm
