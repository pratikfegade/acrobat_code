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
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include "../../printer/text_printer.h"
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
  const bool structurally_allowed;
  const bool has_op_calls;

  bool is_allowed() const { return structurally_allowed && has_op_calls; }
};

Op GetInvokeTVMOp() {
  static const Op& op = Op::Get("vm.invoke_tvm_op");
  return op;
}

std::string DebugPrint(const ObjectRef& obj) {
  return tvm::TextPrinter(false, nullptr, true).PrintFinal(obj).str();
}

typedef std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> Groups;
class GroupFinder : public ExprFunctor<GroupFinderResult(const Expr& n)> {
 public:
  Groups FindGroups(const Expr expr) {
    VisitExpr(expr);
    return found_groups;
  }

 private:
  void AddToGroup(Expr expr) { found_groups.insert(expr); }

  GroupFinderResult VisitExpr_(const ConstantNode* op) override { return {true, false}; }

  GroupFinderResult VisitExpr_(const TupleNode* op) override {
    // std::cout << "Visiting tuple " << GetRef<Expr>(op) << std::endl;
    std::vector<bool> allowed_fields;
    bool structurally_allowed_tuple = true;
    bool tuple_has_call_ops = false;
    allowed_fields.reserve(op->fields.size());
    for (auto field : op->fields) {
      auto res = VisitExpr(field);
      structurally_allowed_tuple &= res.structurally_allowed;
      tuple_has_call_ops |= res.has_op_calls;
      allowed_fields.push_back(res.structurally_allowed && res.has_op_calls);
    }

    if (structurally_allowed_tuple) {
      return {true, tuple_has_call_ops};
    } else {
      for (size_t i = 0; i < op->fields.size(); ++i) {
        if (allowed_fields[i]) {
          AddToGroup(op->fields[i]);
        }
      }
      return {false, tuple_has_call_ops};
    }
  }

  GroupFinderResult VisitExpr_(const VarNode* op) override { return {true, false}; }

  GroupFinderResult VisitExpr_(const GlobalVarNode* op) override { return {false, false}; }

  GroupFinderResult VisitExpr_(const FunctionNode* op) override {
    if (VisitExpr(op->body).is_allowed()) {
      AddToGroup(op->body);
    }
    return {false, false};
  }

  GroupFinderResult VisitExpr_(const CallNode* op) override {
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      return VisitExpr(on_device_props.body);
    } else if (op->op == GetInvokeTVMOp()) {
      if (VisitExpr(op->args[1]).structurally_allowed &&
          VisitExpr(op->args[2]).structurally_allowed) {
        return {true, true};
      }
    }
    std::vector<bool> allowed_args;
    for (auto arg : op->args) {
      if (VisitExpr(arg).is_allowed()) {
        AddToGroup(arg);
      }
    }

    if (VisitExpr(op->op).is_allowed()) {
      AddToGroup(op->op);
    }

    return {false, false};
  }

  GroupFinderResult VisitExpr_(const LetNode* op) override {
    auto value_res = VisitExpr(op->value);
    auto body_res = VisitExpr(op->body);
    if (value_res.structurally_allowed && body_res.structurally_allowed) {
      return {true, value_res.has_op_calls || body_res.has_op_calls};
    } else if (value_res.is_allowed()) {
      AddToGroup(op->value);
    } else if (body_res.is_allowed()) {
      AddToGroup(op->body);
    }
    return {false, false};
  }

  GroupFinderResult VisitExpr_(const IfNode* op) override {
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

  GroupFinderResult VisitExpr_(const OpNode* op) override { return {false, false}; }

  GroupFinderResult VisitExpr_(const TupleGetItemNode* op) override { return VisitExpr(op->tuple); }

  GroupFinderResult VisitExpr_(const RefCreateNode* op) override {
    if (VisitExpr(op->value).is_allowed()) {
      AddToGroup(op->value);
    }
    return {false, false};
  }

  GroupFinderResult VisitExpr_(const RefReadNode* op) override {
    if (VisitExpr(op->ref).is_allowed()) {
      AddToGroup(op->ref);
    }
    return {false, false};
  }

  GroupFinderResult VisitExpr_(const RefWriteNode* op) override {
    if (VisitExpr(op->ref).is_allowed()) {
      AddToGroup(op->ref);
    }
    if (VisitExpr(op->value).is_allowed()) {
      AddToGroup(op->value);
    }
    return {false, false};
  }

  GroupFinderResult VisitExpr_(const ConstructorNode* op) override { return {false, false}; }

  GroupFinderResult VisitExpr_(const MatchNode* op) override {
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

class TupleInliner : public ExprMutator {
 public:
  Expr VisitExpr_(const LetNode* let) override {
    Var var = let->var;
    Expr value = let->value;
    Expr body = let->body;

    if (value.as<TupleNode>()) {
      var_values_.Set(var, VisitExpr(value));
      return VisitExpr(body);
      // return Let(var, VisitExpr(value), VisitExpr(body));
    } else {
      return Let(var, VisitExpr(value), VisitExpr(body));
    }
  }

  Expr VisitExpr_(const VarNode* var) override {
    auto it = var_values_.find(GetRef<Var>(var));
    if (it != var_values_.end()) {
      return (*it).second;
    } else {
      return ExprMutator::VisitExpr_(var);
    }
  }

  Expr VisitExpr_(const TupleGetItemNode* op) override {
    Expr tuple = VisitExpr(op->tuple);
    if (auto tn = tuple.as<TupleNode>()) {
      return tn->fields[op->index];
    } else {
      return TupleGetItem(tuple, op->index);
    }
  }

  Expr VisitExpr_(const CallNode* op) override {
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      return VisitExpr(on_device_props.body);
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

 private:
  Map<Var, Expr> var_values_;
};

// Construct a schedule for a given Relay primitive function and target.
class TIRLowerer : public ExprFunctor<ObjectRef(const Expr& n)> {
 public:
  tir::Stmt LowerToTIR(const Expr& rexpr) {
    Array<Var> free_vars = FreeVars(rexpr);
    for (auto var : free_vars) {
      if (auto ttn = var->checked_type().as<TupleTypeNode>()) {
        Array<PrimExpr> tir_tuple_vars;
        int ctr = 0;
        for (auto field_type : ttn->fields) {
          tir_tuple_vars.push_back(tir::Var(var->vid->name_hint + std::to_string(ctr++),
                                            RelayTypeToTIRType(field_type)));
        }
        tuple_var_values_.Set(var, tir_tuple_vars);
      } else {
        var_values_.Set(var,
                        tir::Var(var->vid->name_hint, RelayTypeToTIRType(var->checked_type())));
      }
    }
    auto tir = ConvertToTIRStmt(VisitExpr(rexpr));
    return tir;
  }

  // let var == tvm_invoke_op
  // let var == tuple

  ObjectRef VisitExpr_(const LetNode* op) override {
    auto lhs = op->var;
    auto rhs = op->value;
    auto body = op->body;

    auto rhs_call = rhs.as<CallNode>();
    auto rhs_tuple = rhs.as<TupleNode>();
    if (rhs_call && rhs_call->op == GetInvokeTVMOp()) {
      return tir::SeqStmt(
          {tir::Evaluate(ConvertTVMOpInvoke(rhs_call)), ConvertToTIRStmt(VisitExpr(body))});
    } else if (rhs_tuple) {
      tuple_var_values_.Set(lhs, FlattenTuple(rhs));
      return VisitExpr(body);
    } else {
      var_values_.Set(lhs, ConvertToTIRExpr(VisitExpr(rhs)));
      return VisitExpr(body);
    }
  }

  ObjectRef VisitExpr_(const relay::TupleGetItemNode* op) override {
    if (op->tuple.as<relay::VarNode>()) {
      auto it = tuple_var_values_.find(Downcast<relay::Var>(op->tuple));
      ICHECK(it != tuple_var_values_.end());
      return ((*it).second)[op->index];
    } else if (auto tuple = op->tuple.as<relay::TupleNode>()) {
      return VisitExpr(tuple->fields[op->index]);
    } else {
      ICHECK(false);
      return NullValue<PrimExpr>();
    }
  }

  ObjectRef VisitExpr_(const VarNode* op) override {
    std::cout << "Visiting var " << GetRef<Expr>(op) << std::endl;
    auto it = var_values_.find(GetRef<relay::Var>(op));
    auto iit = tuple_var_values_.find(GetRef<relay::Var>(op));
    if (it != var_values_.end()) {
      return (*it).second;
    } else if (iit != tuple_var_values_.end()) {
      return (*iit).second;
    } else {
      return GetOrCreateVar(GetRef<Var>(op));
    }
  }

  Array<PrimExpr> FlattenTuple(const Expr& rexpr) {
    if (auto tuple = rexpr.as<TupleNode>()) {
      Array<PrimExpr> tir_fields;
      for (auto field : tuple->fields) {
        for (auto field_field : FlattenTuple(field)) {
          tir_fields.push_back(field_field);
        }
      }
      return tir_fields;
    } else if (rexpr.as<relay::VarNode>()) {
      auto var = Downcast<relay::Var>(rexpr);
      if (tuple_var_values_.count(var)) {
        return tuple_var_values_.at(var);
      }
    }

    return {ConvertToTIRExpr(VisitExpr(rexpr))};
  }

  PrimExpr ConvertTVMOpInvoke(const relay::CallNode* call) {
    Array<PrimExpr> args;
    // args.push_back(tir::StringImm(call->args[0].as<tir::PrimFuncNode>()->name));
    args.push_back(tir::StringImm("yumma yumma"));
    std::cout << "Call " << GetRef<Expr>(call) << std::endl;
    ICHECK_GE(call->args.size(), 3) << GetRef<Expr>(call);

    Expr inputs_tuple = call->args[1];
    Expr outputs_tuple = call->args[2];
    for (auto arg : FlattenTuple(inputs_tuple)) {
      args.push_back(arg);
    }
    for (auto arg : FlattenTuple(outputs_tuple)) {
      args.push_back(arg);
    }
    return tir::Call(DataType::Int(32), tir::builtin::tvm_call_packed(), args, call->span);
  }

  ObjectRef VisitExpr_(const IfNode* op) override {
    PrimExpr cond = ConvertToTIRExpr(VisitExpr(op->cond));
    tir::Stmt true_branch = ConvertToTIRStmt(VisitExpr(op->true_branch));
    tir::Stmt false_branch = ConvertToTIRStmt(VisitExpr(op->false_branch));

    return tir::IfThenElse(cond, true_branch, false_branch, op->span);
  }

  Type RelayTypeToTIRType(const Type& type) {
    if (type.as<PrimTypeNode>()) {
      return type;
    } else if (auto ttn = type.as<TensorTypeNode>()) {
      return PointerType(PrimType(ttn->dtype));
    } else {
      ICHECK(false) << "Do not know how to convert type " << type;
      return type;
    }
  }

  PrimExpr GetOrCreateVar(relay::Var rvar) {
    auto it = var_values_.find(rvar);
    if (it != var_values_.end()) {
      PrimExpr tvar = (*it).second;
      return tvar;
    } else {
      tir::Var tvar = tir::Var(rvar->vid->name_hint, RelayTypeToTIRType(rvar->checked_type()));
      var_values_.Set(rvar, tvar);
      return tvar;
    }
  }

  tir::Stmt ConvertToTIRStmt(const ObjectRef& obj) {
    if (obj.as<tir::StmtNode>()) {
      return Downcast<tir::Stmt>(obj);
    } else {
      ICHECK(obj.as<PrimExprNode>());
      return tir::Evaluate(Downcast<PrimExpr>(obj));
    }
  }

  PrimExpr ConvertToTIRExpr(const ObjectRef& obj) {
    ICHECK(obj.as<PrimExprNode>());
    return Downcast<PrimExpr>(obj);
  }

  ObjectRef VisitExpr_(const ConstantNode* op) override { return ThrowError(GetRef<Expr>(op)); }

  ObjectRef VisitExpr_(const TupleNode* op) override { return FlattenTuple(GetRef<Expr>(op)); }

  ObjectRef VisitExpr_(const relay::CallNode* op) override {
    if (op->op == GetInvokeTVMOp()) {
      return ConvertTVMOpInvoke(op);
    } else {
      return ThrowError(GetRef<Expr>(op));
    }
  }

  ObjectRef VisitExpr_(const GlobalVarNode* op) override { return ThrowError(GetRef<Expr>(op)); }
  ObjectRef VisitExpr_(const FunctionNode* op) override { return ThrowError(GetRef<Expr>(op)); }
  ObjectRef VisitExpr_(const OpNode* op) override { return ThrowError(GetRef<Expr>(op)); }
  ObjectRef VisitExpr_(const RefCreateNode* op) override { return ThrowError(GetRef<Expr>(op)); }
  ObjectRef VisitExpr_(const RefReadNode* op) override { return ThrowError(GetRef<Expr>(op)); }
  ObjectRef VisitExpr_(const RefWriteNode* op) override { return ThrowError(GetRef<Expr>(op)); }
  ObjectRef VisitExpr_(const ConstructorNode* op) override { return ThrowError(GetRef<Expr>(op)); }
  ObjectRef VisitExpr_(const MatchNode* op) override { return ThrowError(GetRef<Expr>(op)); }
  ObjectRef ThrowError(const Expr& expr) {
    ICHECK(false) << "Wrong expr type " << expr;
    return NullValue<PrimExpr>();
  }

 private:
  Map<relay::Var, PrimExpr> var_values_;
  Map<relay::Var, Array<PrimExpr>> tuple_var_values_;
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
  std::cout << "Coarsening now!" << std::endl;
  // return CoarsenMutator().Mutate(expr);
  auto groups = GroupFinder().FindGroups(expr);
  std::cout << "Groups: " << groups.size() << std::endl;
  for (auto group : groups) {
    group = TupleInliner()(group);
    std::cout << "  Group " << DebugPrint(group) << std::endl;
    auto tir = TIRLowerer().LowerToTIR(group);
    std::cout << "  TIR " << DebugPrint(group) << std::endl;
  }
  // auto ret = CoarsenRewriter(groups, module)(expr);
  // std::cout << "After coarsening\n" << ret << "\n\n\n" << std::endl;
  return expr;
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
