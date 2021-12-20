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
#include <tvm/tir/op.h>

#include "../../support/arena.h"
#include "../op/annotation/annotation.h"
#include "./expr_subst.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {

Expr RemoveOnDeviceCalls(const Expr& expr) {
  class OnDeviceRemover : public ExprMutator {
   public:
    Expr VisitExpr_(const CallNode* call) override {
      return IgnoreOnDevice(ExprMutator::VisitExpr_(call));
    }
  };
  return OnDeviceRemover()(expr);
}

struct GroupFinderResult {
  bool structurally_allowed;
  bool has_op_calls;

  bool is_allowed() { return structurally_allowed && has_op_calls; }
};

typedef std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> Groups;
class GroupFinder : public ExprFunctor<GroupFinderResult(const Expr& n)> {
 public:
  Groups FindGroups(const Expr expr) {
    VisitExpr(expr);
    return found_groups;
  }

 private:
  void AddToGroup(Expr expr) {
    if (!expr.as<VarNode>()) {
      found_groups.insert(expr);
    }
  }

  GroupFinderResult VisitExpr_(const ConstantNode* op) { return {true, false}; }

  GroupFinderResult VisitExpr_(const TupleNode* op) {
    std::vector<bool> allowed_fields;
    bool allowed_tuple = true;
    bool tuple_has_call_ops = false;
    allowed_fields.reserve(op->fields.size());
    for (auto field : op->fields) {
      auto res = VisitExpr(field);
      allowed_tuple &= res.structurally_allowed;
      tuple_has_call_ops |= res.has_op_calls;
      allowed_fields.push_back(res.structurally_allowed && res.has_op_calls);
    }

    if (allowed_tuple && tuple_has_call_ops) {
      return {true, true};
    } else {
      for (size_t i = 0; i < op->fields.size(); ++i) {
        if (allowed_fields[i]) {
          AddToGroup(op->fields[i]);
        }
      }
      return {false, tuple_has_call_ops};
    }
  }

  GroupFinderResult VisitExpr_(const VarNode* op) { return {true, false}; }

  GroupFinderResult VisitExpr_(const GlobalVarNode* op) { return {false, false}; }

  GroupFinderResult VisitExpr_(const FunctionNode* op) {
    if (VisitExpr(op->body).is_allowed()) {
      AddToGroup(op->body);
    }
    return {false, false};
  }

  GroupFinderResult VisitExpr_(const CallNode* op) {
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      return VisitExpr(on_device_props.body);
    }
    std::vector<bool> allowed_args;
    bool allowed_all_args = true;
    bool args_have_call_ops = false;
    for (auto arg : op->args) {
      auto res = VisitExpr(arg);
      allowed_all_args &= res.structurally_allowed;
      args_have_call_ops |= res.has_op_calls;
      allowed_args.push_back(res.structurally_allowed && res.has_op_calls);
    }

    // std::cout << "Visiting call " << GetRef<Expr>(op) << std::endl;
    if (op->op.as<OpNode>()) {
      // std::cout << "  Is op "
      // << " " << allowed_all_args << std::endl;
      if (allowed_all_args) {
        return {true, true};
      }
    } else {
      auto op_res = VisitExpr(op->op);
      if (op_res.structurally_allowed && op_res.has_op_calls) {
        AddToGroup(op->op);
      }
    }

    for (size_t i = 0; i < op->args.size(); ++i) {
      if (allowed_args[i]) {
        AddToGroup(op->args[i]);
      }
    }
    return {false, args_have_call_ops};
  }

  GroupFinderResult VisitExpr_(const LetNode* op) {
    auto value_res = VisitExpr(op->value);
    auto body_res = VisitExpr(op->body);
    if (value_res.is_allowed() && body_res.is_allowed()) {
      return {true, true};
    } else if (value_res.is_allowed()) {
      AddToGroup(op->value);
    } else if (body_res.is_allowed()) {
      AddToGroup(op->body);
    }
    return {false, value_res.has_op_calls || body_res.has_op_calls};
  }

  GroupFinderResult VisitExpr_(const IfNode* op) {
    if (VisitExpr(op->cond).is_allowed()) {
      AddToGroup(op->cond);
    }
    if (VisitExpr(op->true_branch).is_allowed()) {
      AddToGroup(op->true_branch);
    }
    if (VisitExpr(op->false_branch).is_allowed()) {
      AddToGroup(op->false_branch);
    }
    return {false, false};
  }

  GroupFinderResult VisitExpr_(const OpNode* op) { return {true, true}; }

  GroupFinderResult VisitExpr_(const TupleGetItemNode* op) {
    auto ret = VisitExpr(op->tuple);
    // if (op->tuple.as<VarNode>()) {
    // std::cout << "Visiting tuple get " << GetRef<Expr>(op) << " " << ret.structurally_allowed
    // << " " << ret.has_op_calls << std::endl;
    // }
    return ret;
  }

  GroupFinderResult VisitExpr_(const RefCreateNode* op) {
    if (VisitExpr(op->value).is_allowed()) {
      AddToGroup(op->value);
    }
    return {false, false};
  }

  GroupFinderResult VisitExpr_(const RefReadNode* op) {
    if (VisitExpr(op->ref).is_allowed()) {
      AddToGroup(op->ref);
    }
    return {false, false};
  }

  GroupFinderResult VisitExpr_(const RefWriteNode* op) {
    if (VisitExpr(op->ref).is_allowed()) {
      AddToGroup(op->ref);
    }
    if (VisitExpr(op->value).is_allowed()) {
      AddToGroup(op->value);
    }
    return {false, false};
  }

  GroupFinderResult VisitExpr_(const ConstructorNode* op) { return {false, false}; }

  GroupFinderResult VisitExpr_(const MatchNode* op) {
    if (VisitExpr(op->data).is_allowed()) {
      AddToGroup(op->data);
    }
    for (auto clause : op->clauses) {
      if (VisitExpr(clause->rhs).is_allowed()) {
        AddToGroup(clause->rhs);
      }
    }
    return {false, false};
  }

  Groups found_groups;
};

class CoarsenRewriter : public ExprMutator {
 public:
  CoarsenRewriter(const Groups& groups, const IRModule& mod) : groups_(groups), mod_(mod) {}

 private:
  template <typename T>
  Expr Rewrite(const Expr& expr, const T* op) {
    if (groups_.count(expr)) {
      // std::cout << "Rewriting " << expr << std::endl;
      auto on_device_removed_expr = RemoveOnDeviceCalls(expr);
      auto free_vars = FreeVars(on_device_removed_expr);
      auto free_type_vars = FreeTypeVars(on_device_removed_expr, mod_);
      if (free_type_vars.size() > 0) {
        return ExprMutator::VisitExpr_(op);
      }

      std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> subst_map;
      Array<Var> params;
      for (auto var : free_vars) {
        auto new_var = Var(var->vid, var->type_annotation, var->span);
        subst_map.insert({var, new_var});
        params.push_back(new_var);
      }

      Function func = Function(params, ExprSubst(on_device_removed_expr, subst_map),
                               expr->checked_type_, free_type_vars);
      func = WithAttr(std::move(func), attr::kPrimitive, tvm::Integer(1));
      Var fun_var = Var("fun_coarsened", NullValue<Type>());
      Array<Expr> args;
      for (auto v : free_vars) {
        args.push_back(v);
      }
      Expr call = Call(fun_var, args);
      auto ret = Let(fun_var, func, call);
      // std::cout << "New primitive function " << func << std::endl;
      return ret;
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const ConstantNode* op) { return Rewrite(GetRef<Expr>(op), op); }

  Expr VisitExpr_(const TupleNode* op) { return Rewrite(GetRef<Expr>(op), op); }

  Expr VisitExpr_(const VarNode* op) { return Rewrite(GetRef<Expr>(op), op); }

  Expr VisitExpr_(const FunctionNode* op) { return Rewrite(GetRef<Expr>(op), op); }

  Expr VisitExpr_(const CallNode* op) { return Rewrite(GetRef<Expr>(op), op); }

  Expr VisitExpr_(const LetNode* op) { return Rewrite(GetRef<Expr>(op), op); }

  Expr VisitExpr_(const OpNode* op) { return Rewrite(GetRef<Expr>(op), op); }

  Expr VisitExpr_(const TupleGetItemNode* op) { return Rewrite(GetRef<Expr>(op), op); }

 private:
  const Groups& groups_;
  const IRModule& mod_;
};

Expr CoarsenPrimitiveFuncGranularity(const Expr& expr, const IRModule& module) {
  std::cout << "Before coarsening\n" << expr << "\n\n\n" << std::endl;
  // return CoarsenMutator().Mutate(expr);
  auto groups = GroupFinder().FindGroups(expr);
  std::cout << "Groups: " << groups.size() << std::endl;
  for (auto group : groups) {
    std::cout << "  Group " << group << std::endl;
  }
  exit(0);
  auto ret = CoarsenRewriter(groups, module)(expr);
  // std::cout << "After coarsening\n" << ret << "\n\n\n" << std::endl;
  return ret;
}

namespace transform {

Pass CoarsenPrimitiveFuncGranularity() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(CoarsenPrimitiveFuncGranularity(f, m));
      };
  return CreateFunctionPass(pass_func, 0, "CoarsenPrimitiveFuncGranularity", {});
}

TVM_REGISTER_GLOBAL("relay._transform.CoarsenPrimitiveFuncGranularity").set_body_typed(FuseOps);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
