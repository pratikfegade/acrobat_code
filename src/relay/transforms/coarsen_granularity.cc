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
#include "../op/vm/vm.h"
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

class LetLifter : public ExprMutator {
 public:
  Expr VisitExpr_(const LetNode* outer_let) {
    auto outer_var = outer_let->var;
    auto outer_value = VisitExpr(outer_let->value);
    auto outer_body = VisitExpr(outer_let->body);

    if (auto inner_let = outer_value.as<LetNode>()) {
      auto inner_var = inner_let->var;
      auto inner_value = inner_let->value;
      auto inner_body = inner_let->body;

      if (StructuralEqual()(outer_var, outer_body)) {
        auto ret = Let(inner_var, inner_value, inner_body);
        // std::cout << "[LL] Visiting\n " << DebugPrint(GetRef<Expr>(outer_let)) << std::endl;
        // std::cout << "[LL]   Returning\n " << DebugPrint(ret) << "\n\n" << std::endl;
        return ret;
      } else {
        auto ret = Let(inner_var, inner_value, Let(outer_var, inner_body, outer_body));
        std::cout << "[LL] Visiting\n " << DebugPrint(GetRef<Expr>(outer_let)) << std::endl;
        std::cout << "[LL]   Returning\n " << DebugPrint(ret) << "\n\n" << std::endl;
        return ret;
      }
    } else {
      return Let(outer_var, outer_value, outer_body);
    }
  }
};

class TupleInliner : public ExprMutator {
 public:
  // Expr VisitExpr_(const LetNode* let) override {
  //   Var var = let->var;
  //   Expr value = let->value;
  //   Expr body = let->body;

  //   if (value.as<TupleNode>()) {
  //     var_values_.Set(var, VisitExpr(value));
  //     return VisitExpr(body);
  //     // return Let(var, VisitExpr(value), VisitExpr(body));
  //   } else {
  //     return Let(var, VisitExpr(value), VisitExpr(body));
  //   }
  // }

  // Expr VisitExpr_(const VarNode* var) override {
  //   auto it = var_values_.find(GetRef<Var>(var));
  //   if (it != var_values_.end()) {
  //     return (*it).second;
  //   } else {
  //     return ExprMutator::VisitExpr_(var);
  //   }
  // }

  // Expr VisitExpr_(const TupleGetItemNode* op) override {
  //   Expr tuple = VisitExpr(op->tuple);
  //   if (auto tn = tuple.as<TupleNode>()) {
  //     return tn->fields[op->index];
  //   } else {
  //     return TupleGetItem(tuple, op->index);
  //   }
  // }

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

class Serializer : public ExprVisitor {
 public:
  void Serialize(const Expr& expr) {
    VisitExpr(expr);
    if (!body_.as<VarNode>()) {
      auto var = relay::Var("dummy_body", body_->checked_type());
      bindings_.push_back(std::make_pair(var, body_));
      body_ = var;
    }
  }

  void VisitExpr_(const LetNode* op) override {
    Expr let = GetRef<Expr>(op);
    while (auto let_node = let.as<LetNode>()) {
      bindings_.push_back(std::make_pair(let_node->var, let_node->value));
      let = let_node->body;
    }
    body_ = let;
  }

  void VisitExpr_(const relay::TupleGetItemNode* op) override { HandleNotLet(GetRef<Expr>(op)); }
  void VisitExpr_(const VarNode* op) override { HandleNotLet(GetRef<Expr>(op)); }
  void VisitExpr_(const IfNode* op) override { HandleNotLet(GetRef<Expr>(op)); }
  void VisitExpr_(const ConstantNode* op) override { HandleNotLet(GetRef<Expr>(op)); }
  void VisitExpr_(const TupleNode* op) override { HandleNotLet(GetRef<Expr>(op)); }
  void VisitExpr_(const relay::CallNode* op) override { HandleNotLet(GetRef<Expr>(op)); }
  void VisitExpr_(const GlobalVarNode* op) override { HandleNotLet(GetRef<Expr>(op)); }
  void VisitExpr_(const FunctionNode* op) override { HandleNotLet(GetRef<Expr>(op)); }
  void VisitExpr_(const OpNode* op) override { HandleNotLet(GetRef<Expr>(op)); }
  void VisitExpr_(const RefCreateNode* op) override { HandleNotLet(GetRef<Expr>(op)); }
  void VisitExpr_(const RefReadNode* op) override { HandleNotLet(GetRef<Expr>(op)); }
  void VisitExpr_(const RefWriteNode* op) override { HandleNotLet(GetRef<Expr>(op)); }
  void VisitExpr_(const ConstructorNode* op) override { HandleNotLet(GetRef<Expr>(op)); }
  void VisitExpr_(const MatchNode* op) override { HandleNotLet(GetRef<Expr>(op)); }
  void HandleNotLet(const Expr& expr) { body_ = expr; }

  std::vector<std::pair<Var, Expr>> bindings_;
  Expr body_;
};

Type GetVarType(relay::Var var) {
  if (var->checked_type_.defined()) {
    return var->checked_type();
  } else {
    ICHECK(var->type_annotation.defined());
    return var->type_annotation;
  }
}

typedef std::unordered_set<relay::Var, ObjectPtrHash, ObjectPtrEqual> RelayVarSet;

void FlattenVar(const relay::Var& var, Map<relay::Var, Array<Expr>>* p_tuple_var_values,
                std::vector<relay::Var>* p_flattened_free_vars,
                std::vector<Expr>* p_flattened_call_args) {
  Map<relay::Var, Array<Expr>>& tuple_var_values = *p_tuple_var_values;
  std::vector<relay::Var>& flattened_free_vars = *p_flattened_free_vars;
  std::vector<Expr>& flattened_call_args = *p_flattened_call_args;

  if (auto ttn = var->checked_type().as<TupleTypeNode>()) {
    Array<Expr> tir_tuple_vars;
    int idx = 0;
    for (auto field_type : ttn->fields) {
      auto flattened_var =
          relay::Var("ft_" + var->vid->name_hint + std::to_string(idx), field_type);
      tir_tuple_vars.push_back(flattened_var);
      flattened_free_vars.push_back(flattened_var);
      flattened_call_args.push_back(TupleGetItem(var, idx++));
      if (field_type.as<TupleTypeNode>()) {
        FlattenVar(flattened_var, p_tuple_var_values, p_flattened_free_vars, p_flattened_call_args);
      }
      // std::cout << "[TL]  Flattened " << flattened_var << std::endl;
    }
    tuple_var_values.Set(var, tir_tuple_vars);
  } else {
    auto flattened_var = relay::Var("f_" + var->vid->name_hint, var->checked_type());
    tuple_var_values.Set(var, {flattened_var});
    flattened_free_vars.push_back(flattened_var);
    flattened_call_args.push_back(var);
    // std::cout << "[TL]  Flattened " << flattened_var << std::endl;
  }
}

struct TIRLowererResult {
  tir::PrimFunc func;
  std::vector<Expr> call_args;
  Expr replacement;
};

class TIRLowerer : public ExprFunctor<ObjectRef(const Expr& n)> {
 public:
  TIRLowererResult LowerToTIR(const Expr& rexpr, const Expr& body,
                              const std::vector<std::pair<relay::Var, Expr>> bindings) {
    RelayVarSet free_vars_set = FreeVarsDedup(rexpr);
    std::vector<Var> free_vars(free_vars_set.begin(), free_vars_set.end());
    std::vector<Var> flattened_free_vars;
    std::vector<Expr> flattened_call_args;

    for (auto var : free_vars) {
      // std::cout << "[TL]   FreeVar " << var->vid->name_hint << " " << var.get() << std::endl;
      FlattenVar(var, &tuple_var_values_, &flattened_free_vars, &flattened_call_args);
    }

    Array<tir::Stmt> tir_stmts;
    Map<relay::Var, Expr> bindings_map;
    for (auto pair : bindings) {
      auto rvar = pair.first;
      auto rvalue = pair.second;
      bindings_map.Set(rvar, rvalue);

      // std::cout << "[TL] Pair " << rvar->vid->name_hint << " " << rvalue << std::endl;
      if (auto call = rvalue.as<CallNode>()) {
        ICHECK(call->op == GetInvokeTVMOp());
        tir_stmts.push_back(tir::Evaluate(ConvertTVMOpInvoke(call)));
      } else if (auto tuple = rvalue.as<TupleNode>()) {
        tuple_var_values_.Set(rvar, tuple->fields);
      } else if (auto tuple_get = rvalue.as<TupleGetItemNode>()) {
        auto tuple = tuple_get->tuple;
        ICHECK(tuple.as<relay::VarNode>()) << tuple;
        auto tuple_var = Downcast<relay::Var>(tuple);
        ICHECK(tuple_var_values_.count(tuple_var));
        auto tuple_values = tuple_var_values_.at(tuple_var);
        ICHECK_GT(tuple_values.size(), tuple_get->index);
        auto tuple_get_value = tuple_values[tuple_get->index];
        ICHECK(tuple_get_value.as<relay::VarNode>());
        auto tuple_get_value_var = Downcast<relay::Var>(tuple_get_value);
        if (rvalue->checked_type().as<TupleTypeNode>()) {
          ICHECK(tuple_var_values_.count(tuple_get_value_var));
          tuple_var_values_.Set(rvar, tuple_var_values_.at(tuple_get_value_var));
        } else {
          tuple_var_values_.Set(rvar, {tuple_get_value});
        }
      } else if (rvalue.as<relay::VarNode>()) {
        auto value_var = Downcast<relay::Var>(rvalue);
        ICHECK(tuple_var_values_.count(value_var));
        tuple_var_values_.Set(rvar, tuple_var_values_.at(value_var));
      } else {
        ICHECK(false) << "Unsupported value type " << rvalue;
      }
    }

    // std::cout << "[TL] Body " << body << std::endl;
    Expr body_in_free_vars;
    if (body.as<relay::VarNode>() && tuple_var_values_.count(Downcast<relay::Var>(body))) {
      body_in_free_vars = ExpressBodyWithFreeVars(body, free_vars_set, bindings_map);
    } else {
      body_in_free_vars = Tuple(Array<Expr>());
    }
    // std::cout << "[TL] Rewritten body " << body_in_free_vars << std::endl;
    tir::Stmt prim_func_body = tir::SeqStmt(tir_stmts);
    Array<tir::Var> prim_func_params;
    for (auto rvar : flattened_free_vars) {
      prim_func_params.push_back(GetOrCreateVar(rvar));
    }

    return TIRLowererResult({tir::PrimFunc(prim_func_params, prim_func_body, VoidType()),
                             flattened_call_args, body_in_free_vars});
  }

  Expr ExpressBodyWithFreeVars(const Expr& body, const RelayVarSet& free_vars,
                               const Map<relay::Var, Expr>& bindings_map) {
    class Visitor : public ExprMutator {
     public:
      Visitor(const RelayVarSet& free_vars, const Map<relay::Var, Expr>& bindings_map)
          : free_vars_(free_vars), bindings_map_(bindings_map) {}
      Expr VisitExpr_(const VarNode* op) {
        // std::cout << "[TL]  Var " << op->vid->name_hint << std::endl;
        auto var = GetRef<relay::Var>(op);
        if (free_vars_.count(var)) {
          // std::cout << "[TL]   FreeVar" << std::endl;
          return var;
        } else {
          auto it = bindings_map_.find(var);
          // if (it == bindings_map_.end()) {
          //   for (auto var : free_vars_) {
          //     std::cout << "[FREEVAR]  " << var->vid->name_hint << " " << var.get() << std::endl;
          //   }
          // }
          ICHECK(it != bindings_map_.end()) << var;
          // std::cout << "[TL]   Value " << (*it).second << std::endl;
          return VisitExpr((*it).second);
        }
      }

      const RelayVarSet& free_vars_;
      const Map<relay::Var, Expr>& bindings_map_;
    };
    // std::cout << "[TL] Expressing " << body << std::endl;
    return Visitor(free_vars, bindings_map)(body);
  }

  PrimExpr ConvertTVMOpInvoke(const relay::CallNode* call) {
    Array<PrimExpr> args;
    args.push_back(tir::StringImm(call->args[0].as<GlobalVarNode>()->name_hint));
    ICHECK_GE(call->args.size(), 3) << GetRef<Expr>(call);

    Expr inputs_tuple = call->args[1];
    Expr outputs_tuple = call->args[2];
    // std::cout << "[TL] Call inputs " << inputs_tuple << std::endl;
    for (auto arg : FlattenTuple(inputs_tuple)) {
      ICHECK(arg.as<relay::VarNode>());
      // std::cout << "[TL]   Field " << arg << std::endl;
      args.push_back(GetOrCreateVar(Downcast<relay::Var>(arg)));
    }
    // std::cout << "[TL] Call outputs " << outputs_tuple << std::endl;
    for (auto arg : FlattenTuple(outputs_tuple)) {
      ICHECK(arg.as<relay::VarNode>());
      // std::cout << "[TL]   Field " << arg << std::endl;
      args.push_back(GetOrCreateVar(Downcast<relay::Var>(arg)));
    }
    return tir::Call(DataType::Int(32), tir::builtin::tvm_call_packed(), args, call->span);

    // ICHECK_GE(call->args.size(), 3) << GetRef<Expr>(call);

    // Expr inputs_tuple = call->args[1];
    // Expr outputs_tuple = call->args[2];
    // Array<PrimExpr> args;
    // for (auto arg : FlattenTuple(inputs_tuple)) {
    //   ICHECK(arg.as<relay::VarNode>());
    //   args.push_back(GetOrCreateVar(Downcast<relay::Var>(arg)));
    // }
    // for (auto arg : FlattenTuple(outputs_tuple)) {
    //   ICHECK(arg.as<relay::VarNode>());
    //   args.push_back(GetOrCreateVar(Downcast<relay::Var>(arg)));
    // }
    // return tir::Call(DataType::Void(), call->args[0], args);
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

  Array<Expr> FlattenTuple(const Expr& expr) {
    Array<Expr> tir_fields;
    if (auto tuple = expr.as<TupleNode>()) {
      for (auto field : tuple->fields) {
        for (auto field_field : FlattenTuple(field)) {
          tir_fields.push_back(field_field);
        }
      }
      return tir_fields;
    } else if (expr.as<relay::VarNode>()) {
      auto var = Downcast<relay::Var>(expr);
      if (tuple_var_values_.count(var)) {
        Array<Expr> tuple_value = tuple_var_values_.at(var);
        for (auto field : tuple_value) {
          for (auto field_field : FlattenTuple(field)) {
            tir_fields.push_back(field_field);
          }
        }
        return tir_fields;
      } else {
        tir_fields.push_back(expr);
      }
    }
    return tir_fields;
  }

  tir::Var GetOrCreateVar(relay::Var rvar) {
    // std::cout << "[TL] Var " << rvar->type_annotation << " " << rvar->vid->name_hint <<
    // std::endl;
    auto it = var_to_var_mapping_.find(rvar);
    if (it != var_to_var_mapping_.end()) {
      tir::Var tvar = (*it).second;
      return tvar;
    } else {
      Type var_type = GetVarType(rvar);
      tir::Var tvar = tir::Var(rvar->vid->name_hint, RelayTypeToTIRType(var_type));
      var_to_var_mapping_.Set(rvar, tvar);
      return tvar;
    }
  }

 private:
  Map<relay::Var, tir::Var> var_to_var_mapping_;
  Map<relay::Var, Array<Expr>> tuple_var_values_;
};

class Coarsener : public ExprMutator {
 public:
  Coarsener(const Groups& groups) : groups_(groups) {}

 private:
  Expr VisitExpr(const Expr& expr) override {
    auto it = this->memo_.find(expr);
    if (it != this->memo_.end()) {
      return it->second;
    } else {
      Expr new_expr = HandleExpr(expr);
      memo_[expr] = new_expr;
      return new_expr;
    }
  }

  Expr HandleExpr(const Expr& expr) {
    if (groups_.count(expr)) {
      auto tmp_expr = transform::ToANormalForm(expr);
      tmp_expr = TupleInliner()(expr);
      tmp_expr = LetLifter()(tmp_expr);
      Serializer serializer;
      serializer.Serialize(tmp_expr);
      auto res = TIRLowerer().LowerToTIR(tmp_expr, serializer.body_, serializer.bindings_);
      auto prim_func = res.func;
      auto call_args = res.call_args;
      auto replacement = res.replacement;
      std::string name = "prim_func" + std::to_string(ctr++);
      GlobalVar prim_func_var(name, VoidType());
      auto call = InvokeTVMOp(prim_func_var, Tuple(call_args), Tuple(Array<Expr>()),
                              DictAttrs({{"Primitive", tvm::Integer(1)}}));
      Target target({{String("kind"), String("llvm")}, {String("mcpu"), String("core-avx2")}});

      size_t hash = StructuralHash()(prim_func);

      // format hash as fixed length hex string so it is easier to read
      std::stringstream s;
      s << std::setfill('0') << std::setw(sizeof(size_t) * 2) << std::hex << hash;

      prim_func = WithAttrs(std::move(prim_func), {{"global_symbol", runtime::String(name)},
                                                   {tvm::attr::kTarget, target},
                                                   {"hash", String(s.str())},
                                                   {"coarsened_prim_func", Bool(true)}});
      prim_funcs_.push_back(std::make_pair(prim_func_var, prim_func));
      return Let(Var("dummy", VoidType()), call, replacement);
    } else {
      return ExprMutator::VisitExpr(expr);
    }
  }

 public:
  std::vector<std::pair<GlobalVar, tir::PrimFunc>> prim_funcs_;

 private:
  const Groups& groups_;
  static int ctr;
};

int Coarsener::ctr = 0;

IRModule CoarsenGranularity(const IRModule& mod) {
  // std::cout << "Coarsening now!" << std::endl;
  tvm::Map<GlobalVar, Function> updates;
  tvm::Map<GlobalVar, tir::PrimFunc> new_prim_funcs;
  auto funcs = mod->functions;
  for (const auto& it : funcs) {
    if (const auto* n = it.second.as<FunctionNode>()) {
      ICHECK_EQ(FreeVars(it.second).size(), 0);
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
      Function func = GetRef<Function>(n);

      auto groups = GroupFinder().FindGroups(func);
      Coarsener coarsener(groups);
      Function ret = Downcast<Function>(coarsener(func));

      updates.Set(it.first, ret);

      for (auto it : coarsener.prim_funcs_) {
        new_prim_funcs.Set(it.first, it.second);
      }
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

//   322 GlobalVar(vm_mod_fused_nn_dense_nn_bias_add): PrimFunc([placeholder, placeholder,
//   placeholder, T_add]) attrs={"
// 322 from_legacy_te_schedule": (bool)1, "global_symbol": "vm_mod_fused_nn_dense_nn_bias_add",
// "tir.noalias": (bool)1 322 , "target": llvm -keys=cpu -link-params=0} {

namespace transform {
Pass CoarsenPrimitiveFuncGranularity() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return CoarsenGranularity(m); };
  return CreateModulePass(pass_func, 0, "CoarsenPrimitiveFuncGranularity", {});
}

TVM_REGISTER_GLOBAL("relay._transform.CoarsenPrimitiveFuncGranularity").set_body_typed(FuseOps);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
