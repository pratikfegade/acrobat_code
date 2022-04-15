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

#include "../../printer/text_printer.h"
#include "./function_pointer_analysis.h"
#include "pass_utils.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

constexpr int MAX_DEPTH_VALUE = 1 << 4;
class FoldReduceSumsIdentifier : public ExprMutator {
 public:
  // FoldReduceSumsIdentifier(const IRModule& mod, const FPAVarStateMap& var_points_to_with_context)
  //     : mod_(mod) {
  //   for (auto kv : var_points_to_with_context) {
  //     auto ctx = kv.first.first;
  //     auto var = kv.first.second;
  //     auto points_to = kv.second;
  //     var_points_to_map_[var].insert(points_to.begin(), points_to.end());
  //   }
  // }

  FoldReduceSumsIdentifier(const IRModule& mod) : mod_(mod) {}

  Function Visit(const Function& f) {
    auto body = this->VisitExpr(f->body);
    if (body.same_as(f->body)) {
      return f;
    } else {
      return Function(f->params, body, f->ret_type, f->type_params, f->attrs, f->span);
    }
  }

 private:
  Type GetListType() { return mod_->GetGlobalTypeVar("List"); }

  bool IsListType(const Type& type) { return type == GetListType(); }

  bool IsTensorType(const Type& type) { return (type.as<TensorTypeNode>() != nullptr); }

  GlobalVar GetGlobalVar(const std::string& name) {
    if (mod_->ContainGlobalVar(name)) {
      return mod_->GetGlobalVar(name);
    }
    return NullValue<GlobalVar>();
  }

  GlobalVar GetFoldlGlobalVar() { return GetGlobalVar("foldl"); }

  GlobalVar GetFoldrGlobalVar() { return GetGlobalVar("foldr"); }

  bool IsTensorList(const Expr& expr) {
    auto type = expr->checked_type();
    if (auto tcn = type.as<TypeCallNode>()) {
      return IsListType(tcn->func) && tcn->args.size() == 1 && IsTensorType(tcn->args[0]);
    }
    return false;
  }

  bool DoesPointToReductionLambda(const Expr& e) {
    if (auto fn = e.as<FunctionNode>()) {
      ICHECK_EQ(fn->params.size(), 2);
      auto p1 = fn->params[0];
      auto p2 = fn->params[1];
      bool found_param_add = 0;
      int num_calls = 0;
      PostOrderVisit(fn->body, [&](const Expr& e) {
        if (auto cn = e.as<CallNode>()) {
          num_calls++;
          if (cn->op == GetAddOp() && cn->args.size() == 2 &&
              ((cn->args[0] == p1 && cn->args[1] == p2) ||
               (cn->args[0] == p2 && cn->args[1] == p1))) {
            found_param_add = true;
          }
        }
      });
      if (num_calls == 1 && found_param_add) {
        return true;
      }
    }
    return false;
  }

  Expr VisitExpr(const Expr& expr) {
    auto mutated = ExprMutator::VisitExpr(expr);
    mutated->checked_type_ = expr->checked_type_;
    return mutated;
  }

  Expr VisitExpr_(const CallNode* old_op) {
    auto mutated = ExprMutator::VisitExpr_(old_op);
    auto op = mutated.as<CallNode>();
    ICHECK(op);
    if (op->op == GetFoldlGlobalVar() || op->op == GetFoldrGlobalVar()) {
      // std::cout << "[FRS] Calling fold" << std::endl;
      ICHECK_EQ(op->args.size(), 3);
      auto lambda_arg = op->args[0];
      auto init_arg = op->args[1];
      auto list_arg = op->args[2];
      if (IsTensorList(list_arg) && DoesPointToReductionLambda(lambda_arg)) {
        // std::cout << "[FRS]  Is reduce sum!" << std::endl;
        auto new_attrs = DictAttrs::WithAttr(op->attrs, tir::attr::kDBFoldReduction, GetAddOp());
        mutated = Call(op->op, op->args, new_attrs, op->type_args, op->span);
        mutated->checked_type_ = old_op->checked_type_;
      }
    }
    return mutated;
  }

  const IRModule& mod_;
  // std::unordered_map<const VarNode*, FunctionSet> var_points_to_map_;
};

Function IdentifyFoldReduceSums(const IRModule& mod, Function f) {
  // auto var_points_to_map = GetVarPointsToMap(mod);
  // FoldReduceSumsIdentifier analysis(mod, var_points_to_map);
  FoldReduceSumsIdentifier analysis(mod);
  return analysis.Visit(f);
}

namespace transform {
Pass FoldReduceSumsIdentifierPass() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(IdentifyFoldReduceSums(m, f));
      };
  return CreateFunctionPass(pass_func, 3, "FoldReduceSumsIdentifierPass", {});
}

TVM_REGISTER_GLOBAL("relay._transform.IdentifyFoldReduceSums")
    .set_body_typed(FoldReduceSumsIdentifierPass);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
