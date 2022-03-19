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
#include "../../support/arena.h"
#include "../op/annotation/annotation.h"
#include "../op/vm/vm.h"
#include "./expr_subst.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {

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
        // std::cout << "[LL] Visiting\n " << DebugPrint(GetRef<Expr>(outer_let)) << std::endl;
        // std::cout << "[LL]   Returning\n " << DebugPrint(ret) << "\n\n" << std::endl;
        return ret;
      }
    } else {
      return Let(outer_var, outer_value, outer_body);
    }
  }
};

Expr RemoveOnDeviceCalls(const Expr& expr) {
  class OnDeviceRemover : public ExprMutator {
   public:
    Expr VisitExpr_(const CallNode* call) override {
      return IgnoreOnDevice(ExprMutator::VisitExpr_(call));
    }
  };
  return OnDeviceRemover()(expr);
}

Op GetInvokeTVMOp() {
  static const Op& op = Op::Get("vm.invoke_tvm_op");
  return op;
}

typedef std::unordered_set<relay::Var, ObjectPtrHash, ObjectPtrEqual> RelayVarSet;

class LeafTensorAccessModeCalculator : public tir::StmtExprVisitor {
 public:
  LeafTensorAccessModeCalculator(const tir::PrimFunc& func) : func_(func) {}

  std::unordered_map<tir::Var, runtime::vm::DBArgAccessMode, ObjectPtrHash, ObjectPtrEqual>
  Compute() {
    for (auto pair : func_->buffer_map) {
      param_mapping_[pair.second->data] = pair.first;
    }
    VisitStmt(func_->body);
    return access_modes_;
  }

 private:
  void MergeAndSet(const tir::Var& var, runtime::vm::DBArgAccessMode mode) {
    auto global_symbol = func_->GetAttr<String>(tvm::attr::kGlobalSymbol);

    tir::Var replaced_var = var;
    auto it = param_mapping_.find(var);
    if (it != param_mapping_.end()) {
      replaced_var = it->second;
    }

    auto iit = access_modes_.find(replaced_var);
    if (iit == access_modes_.end()) {
      access_modes_[replaced_var] = mode;
    } else if (iit->second != mode) {
      access_modes_[replaced_var] = runtime::vm::kInputOutput;
    }
  }

  void VisitExpr_(const tir::LoadNode* op) {
    MergeAndSet(op->buffer_var, runtime::vm::kInput);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tir::StoreNode* op) {
    MergeAndSet(op->buffer_var, runtime::vm::kOutput);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const tir::ScatterLoadNode* op) {
    MergeAndSet(op->buffer_var, runtime::vm::kInput);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tir::ScatterStoreNode* op) {
    MergeAndSet(op->buffer_var, runtime::vm::kOutput);
    StmtExprVisitor::VisitStmt_(op);
  }

  std::unordered_map<tir::Var, tir::Var, ObjectPtrHash, ObjectPtrEqual> param_mapping_;
  std::unordered_map<tir::Var, runtime::vm::DBArgAccessMode, ObjectPtrHash, ObjectPtrEqual>
      access_modes_;
  const tir::PrimFunc func_;
};

class CoarsenedTensorAccessModeCalculator : public tir::StmtExprVisitor {
 public:
  CoarsenedTensorAccessModeCalculator(const tir::PrimFunc& func, const IRModule& mod)
      : func_(func), mod_(mod) {}

  Array<Integer> Compute() {
    VisitStmt(func_->body);

    Array<Integer> access_modes;
    for (auto var : func_->params) {
      auto it = access_modes_map_.find(var);
      auto mode = Integer(
          static_cast<int>((it != access_modes_map_.end() ? (*it).second : runtime::vm::kUnused)));
      access_modes.push_back(mode);
    }

    return access_modes;
  }

 private:
  void MergeAndSet(const tir::Var& var, runtime::vm::DBArgAccessMode mode) {
    auto iit = access_modes_map_.find(var);
    if (iit == access_modes_map_.end()) {
      access_modes_map_[var] = mode;
    } else if (iit->second != mode) {
      access_modes_map_[var] = runtime::vm::kInputOutput;
    }
  }

  void VisitExpr_(const tir::CallNode* op) {
    // std::cout << "[CTAMC] Visiting Call " << GetRef<PrimExpr>(op) << std::endl;
    ICHECK_EQ(op->op, tir::builtin::tvm_call_unpacked_from_packed());
    auto base_leaf_callee = mod_->Lookup(Downcast<tir::StringImm>(op->args[0])->value);
    ICHECK(base_leaf_callee.as<tir::PrimFuncNode>());
    auto leaf_callee = Downcast<tir::PrimFunc>(base_leaf_callee);

    auto func_access_modes = LeafTensorAccessModeCalculator(leaf_callee).Compute();

    // std::cout << "[CTAMC]  Function " << leaf_callee << std::endl;
    for (size_t i = 1; i < op->args.size(); ++i) {
      auto arg = op->args[i];
      auto param = leaf_callee->params[i - 1];
      auto it = func_access_modes.find(param);
      if (it != func_access_modes.end()) {
        auto mode = it->second;
        // std::cout << "[CTAMC]   " << arg << " " << param << " " << mode << std::endl;
        MergeAndSet(Downcast<tir::Var>(arg), mode);
      } else {
        // std::cout << "[CTAMC]   " << arg << " " << param << " Unused " << std::endl;
      }
    }
    // std::cout << std::endl;
  }

  std::unordered_map<tir::Var, runtime::vm::DBArgAccessMode, ObjectPtrHash, ObjectPtrEqual>
      access_modes_map_;
  const tir::PrimFunc func_;
  const IRModule mod_;
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
    }
    tuple_var_values.Set(var, tir_tuple_vars);
  } else {
    auto flattened_var = relay::Var("f_" + var->vid->name_hint, var->checked_type());
    tuple_var_values.Set(var, {flattened_var});
    flattened_free_vars.push_back(flattened_var);
    flattened_call_args.push_back(var);
  }
}

struct TIRLowererResult {
  tir::PrimFunc func;
  std::vector<Expr> call_args;
  Expr replacement;
  Array<Integer> arg_modes;
};

class AbstractTIRLowerer {
 public:
  AbstractTIRLowerer(const IRModule& mod, bool scattered_kernels)
      : mod_(mod), scattered_kernels_(scattered_kernels) {}

  ~AbstractTIRLowerer() {}

  virtual TIRLowererResult LowerToTIR(const RelayVarSet& free_vars, const Expr& body,
                                      const std::vector<std::pair<relay::Var, Expr>> bindings) = 0;

 protected:
  virtual PrimExpr ConvertTVMOpInvoke(const relay::CallNode* call) = 0;

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

  Type RelayTypeToTIRType(const Type& type, bool one_more_pointer = false) {
    if (type.as<PrimTypeNode>()) {
      return type;
    } else if (auto ttn = type.as<TensorTypeNode>()) {
      Type type = PointerType(PrimType(ttn->dtype), "");
      if (one_more_pointer) {
        type = PointerType(type, "");
      }
      return type;
    } else {
      ICHECK(false) << "Do not know how to convert type " << type;
      return type;
    }
  }

  tir::Var CreateTIRVar(relay::Var rvar) {
    auto it = var_to_var_mapping_.find(rvar);
    ICHECK(it == var_to_var_mapping_.end())
        << "A TIR variable corresponding to the relay variable " << rvar << " already exists.";
    Type var_type = GetVarType(rvar);
    tir::Var tvar = tir::Var(rvar->vid->name_hint, RelayTypeToTIRType(var_type));
    var_to_var_mapping_.Set(rvar, tvar);
    return tvar;
  }

  tir::Var GetTIRVar(relay::Var rvar) {
    auto it = var_to_var_mapping_.find(rvar);
    ICHECK(it != var_to_var_mapping_.end())
        << "No TIR variable corresponding to the relay variable " << rvar << " exists.";
    tir::Var tvar = (*it).second;
    return tvar;
  }

  const IRModule& mod_;
  bool scattered_kernels_;
  Map<relay::Var, tir::Var> var_to_var_mapping_;
  Map<relay::Var, Array<Expr>> tuple_var_values_;
};

class TIRLowererUnbatched : public AbstractTIRLowerer {
 public:
  TIRLowererUnbatched(const IRModule& mod, bool scattered_kernels)
      : AbstractTIRLowerer(mod, scattered_kernels) {}

  TIRLowererResult LowerToTIR(const RelayVarSet& free_vars_set, const Expr& body,
                              const std::vector<std::pair<relay::Var, Expr>> bindings) final {
    // std::cout << "[CG] Lowering group to TIR" << std::endl;
    std::vector<Var> free_vars(free_vars_set.begin(), free_vars_set.end());
    std::vector<Var> flattened_free_vars;
    std::vector<Expr> flattened_call_args;

    for (auto var : free_vars) {
      FlattenVar(var, &tuple_var_values_, &flattened_free_vars, &flattened_call_args);
    }

    Array<tir::Var> prim_func_params;
    Array<Type> prim_func_param_types;
    for (auto rvar : flattened_free_vars) {
      prim_func_params.push_back(CreateTIRVar(rvar));
      prim_func_param_types.push_back(GetVarType(rvar));
    }

    Array<tir::Stmt> tir_stmts;
    Map<relay::Var, Expr> bindings_map;
    for (auto pair : bindings) {
      auto rvar = pair.first;
      auto rvalue = pair.second;
      bindings_map.Set(rvar, rvalue);

      // std::cout << "[TL] Pair " << rvar->vid->name_hint << " " << rvalue << std::endl;
      if (auto call = rvalue.as<CallNode>()) {
        ICHECK(call->op == GetInvokeTVMOp()) << rvalue;
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

    Expr body_in_free_vars;
    if (body.as<relay::VarNode>() && tuple_var_values_.count(Downcast<relay::Var>(body))) {
      body_in_free_vars = ExpressBodyWithFreeVars(body, free_vars_set, bindings_map);
    } else {
      body_in_free_vars = Tuple(Array<Expr>());
    }

    tir::Stmt prim_func_body = tir::SeqStmt(tir_stmts);

    auto func = tir::PrimFunc(prim_func_params, prim_func_body, VoidType());
    func->checked_type_ = FuncType(prim_func_param_types, VoidType(), {}, {});
    func = WithAttr(func, tir::attr::kDBArgAccessModes,
                    CoarsenedTensorAccessModeCalculator(func, mod_).Compute());
    // std::cout << "[CG]  Generated PrimFunc " << func << std::endl;
    return TIRLowererResult({func, flattened_call_args, body_in_free_vars, Array<Integer>()});
  }

 private:
  PrimExpr ConvertTVMOpInvoke(const relay::CallNode* call) final {
    auto callee_gv = Downcast<GlobalVar>(call->args[0]);

    std::string name = callee_gv->name_hint;
    Array<PrimExpr> args;
    args.push_back(tir::StringImm(name));

    ICHECK_GE(call->args.size(), 3) << GetRef<Expr>(call);
    auto push_args = [&](const Expr& tuple) {
      for (auto arg : FlattenTuple(tuple)) {
        ICHECK(arg.as<relay::VarNode>());
        args.push_back(GetTIRVar(Downcast<relay::Var>(arg)));
      }
    };
    push_args(call->args[1]);
    push_args(call->args[2]);
    return tir::Call(DataType::Int(32), tir::builtin::tvm_call_unpacked_from_packed(), args,
                     call->span);
  }
};

class TIRLowererBatched : public AbstractTIRLowerer {
 public:
  TIRLowererBatched(const IRModule& mod, bool scattered_kernels)
      : AbstractTIRLowerer(mod, scattered_kernels) {
    batch_size_var_ = tir::Var("batch_size", DataType::Int(32));
  }

  TIRLowererResult LowerToTIR(const RelayVarSet& free_vars_set, const Expr& body,
                              const std::vector<std::pair<relay::Var, Expr>> bindings) final {
    bool print = false;  //(bindings.size() < 5);

    if (print) {
      for (auto pair : bindings) {
        std::cout << "[CG] Binding: " << pair.first->vid->name_hint << " = " << pair.second
                  << std::endl;
      }
    }

    /* Prologue: Initialization *************************************************************/

    std::vector<Var> free_vars(free_vars_set.begin(), free_vars_set.end());
    std::vector<Var> flattened_free_vars;
    std::vector<Expr> flattened_call_args;

    for (auto var : free_vars) {
      FlattenVar(var, &tuple_var_values_, &flattened_free_vars, &flattened_call_args);
    }

    for (auto rvar : flattened_free_vars) {
      CreateTIRVar(rvar);
    }

    /* First forward pass *************************************************************/

    for (auto pair : bindings) {
      auto rvar = pair.first;
      auto rvalue = pair.second;

      if (auto call = rvalue.as<CallNode>()) {
        ICHECK(call->op == GetInvokeTVMOp());
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

    /* Backward pass *************************************************************/

    auto merge_and_set_arg_mode = [&](const relay::Var& var, Integer val) {
      if (print) {
        std::cout << "[CG]  MaS: " << var->vid->name_hint << " = " << val << std::endl;
      }
      auto it = arg_mode_states_.find(var);
      if (it != arg_mode_states_.end()) {
        auto old_val = (*it).second;
        val = Integer(std::max(val->value, old_val->value));
        if (print) {
          std::cout << "[CG]   Merging: " << old_val << std::endl;
        }
      }
      if (print) {
        std::cout << "[CG]   Setting: " << val << std::endl;
      }
      arg_mode_states_.Set(var, val);
    };

    for (auto it = bindings.rbegin(); it != bindings.rend(); ++it) {
      auto rvar = it->first;
      auto rvalue = it->second;

      if (auto call = rvalue.as<CallNode>()) {
        if (print) {
          std::cout << "[CG]  Visiting: " << rvalue << " " << FlattenTuple(call->args[1])
                    << std::endl;
        }
        auto callee_gv = Downcast<GlobalVar>(call->args[0]);
        auto iit = mod_->batched_prim_funcs.find(callee_gv);
        ICHECK(iit != mod_->batched_prim_funcs.end()) << callee_gv->name_hint;
        auto batched_callee_gv = (*iit).second;
        auto it = mod_->batched_arg_modes.find(batched_callee_gv);
        ICHECK(it != mod_->batched_arg_modes.end()) << batched_callee_gv->name_hint;
        auto& arg_modes = (*it).second;
        int idx = 0;
        for (auto arg : FlattenTuple(call->args[1])) {
          ICHECK(arg.as<relay::VarNode>());
          auto arg_var = Downcast<relay::Var>(arg);
          auto arg_mode = arg_modes[idx++]->value;
          merge_and_set_arg_mode(arg_var, arg_mode);
        }
        for (auto arg : FlattenTuple(call->args[2])) {
          ICHECK(arg.as<relay::VarNode>());
          auto arg_var = Downcast<relay::Var>(arg);
          auto arg_mode = arg_modes[idx++]->value;
          merge_and_set_arg_mode(arg_var, arg_mode);
        }
      }
    }

    /* Final forward pass *************************************************************/
    var_to_var_mapping_.clear();

    Array<tir::Var> prim_func_params;
    Array<Type> prim_func_param_types;
    Array<Integer> prim_func_arg_modes;
    prim_func_params.push_back(batch_size_var_);
    prim_func_param_types.push_back(batch_size_var_->type_annotation);
    for (auto rvar : flattened_free_vars) {
      auto param = CreateTIRVarWithUpdatedType(rvar);
      prim_func_param_types.push_back(GetVarType(rvar));

      auto iit = arg_mode_states_.find(rvar);
      auto arg_mode = (iit != arg_mode_states_.end()
                           ? (*iit).second
                           : (scattered_kernels_ ? runtime::vm::DBBatchedArgMode::kScatter
                                                 : runtime::vm::DBBatchedArgMode::kConcat));

      if (arg_mode != runtime::vm::DBBatchedArgMode::kIgnore) {
        prim_func_params.push_back(param);
      }
      prim_func_arg_modes.push_back(arg_mode);
      if (print) {
        std::cout << "[CG]  ArgMode: " << rvar->vid->name_hint << " " << param->name_hint << " "
                  << arg_mode << " " << (iit != arg_mode_states_.end()) << std::endl;
      }
    }

    Array<tir::Stmt> tir_stmts;
    Map<relay::Var, Expr> bindings_map;
    for (auto pair : bindings) {
      auto rvar = pair.first;
      auto rvalue = pair.second;
      bindings_map.Set(rvar, rvalue);

      if (auto call = rvalue.as<CallNode>()) {
        tir_stmts.push_back(tir::Evaluate(ConvertTVMOpInvoke(call)));
      } else if (auto tuple = rvalue.as<TupleNode>()) {
        tuple_var_values_.Set(rvar, tuple->fields);
      } else if (auto tuple_get = rvalue.as<TupleGetItemNode>()) {
        auto tuple = tuple_get->tuple;
        auto tuple_var = Downcast<relay::Var>(tuple);
        auto tuple_values = tuple_var_values_.at(tuple_var);
        auto tuple_get_value = tuple_values[tuple_get->index];
        auto tuple_get_value_var = Downcast<relay::Var>(tuple_get_value);
        if (rvalue->checked_type().as<TupleTypeNode>()) {
          tuple_var_values_.Set(rvar, tuple_var_values_.at(tuple_get_value_var));
        } else {
          tuple_var_values_.Set(rvar, {tuple_get_value});
        }
      } else if (rvalue.as<relay::VarNode>()) {
        auto value_var = Downcast<relay::Var>(rvalue);
        tuple_var_values_.Set(rvar, tuple_var_values_.at(value_var));
      } else {
        ICHECK(false) << "Unsupported value type " << rvalue;
      }
    }

    /* Epilogue: Create function *************************************************************/

    Expr body_in_free_vars;
    if (body.as<relay::VarNode>() && tuple_var_values_.count(Downcast<relay::Var>(body))) {
      body_in_free_vars = ExpressBodyWithFreeVars(body, free_vars_set, bindings_map);
    } else {
      body_in_free_vars = Tuple(Array<Expr>());
    }
    tir::Stmt prim_func_body = tir::SeqStmt(tir_stmts);

    auto func = tir::PrimFunc(prim_func_params, prim_func_body, VoidType());
    func->checked_type_ = FuncType(prim_func_param_types, VoidType(), {}, {});
    return TIRLowererResult({func, flattened_call_args, body_in_free_vars, prim_func_arg_modes});
  }

 private:
  PrimExpr ConvertTVMOpInvoke(const relay::CallNode* call) final {
    auto callee_gv = Downcast<GlobalVar>(call->args[0]);
    std::string batched_name = runtime::vm::GetBatchedName(callee_gv->name_hint);

    auto iit = mod_->batched_prim_funcs.find(callee_gv);
    ICHECK(iit != mod_->batched_prim_funcs.end()) << callee_gv->name_hint;
    auto batched_callee_gv = (*iit).second;
    auto it = mod_->batched_arg_modes.find(batched_callee_gv);
    ICHECK(it != mod_->batched_arg_modes.end()) << batched_callee_gv->name_hint;
    auto& arg_modes = (*it).second;
    // std::cout << "[CoG] Arg modes: " << batched_callee_gv->name_hint << " " << arg_modes
    // << std::endl;

    Array<PrimExpr> args;
    args.push_back(tir::StringImm(batched_name));
    // args.push_back(batched_callee_gv);
    args.push_back(batch_size_var_);

    int idx = 0;
    auto push_args = [&](const Expr& tuple) {
      for (auto arg : FlattenTuple(tuple)) {
        if (arg_modes[idx++]->value == static_cast<int>(runtime::vm::kIgnore)) {
          continue;
        }
        ICHECK(arg.as<relay::VarNode>());
        args.push_back(GetTIRVar(Downcast<relay::Var>(arg)));
      }
    };

    ICHECK_GE(call->args.size(), 3) << GetRef<Expr>(call);
    push_args(call->args[1]);
    push_args(call->args[2]);
    // std::cout << "[ARGS] " << args << std::endl;
    // return tir::Call(DataType::Int(32), tir::builtin::tvm_call_packed(), args, call->span);
    // return tir::Call(DataType::Void(), batched_callee_gv, args, call->span);
    return tir::Call(DataType::Int(32), tir::builtin::tvm_call_unpacked_from_packed(), args,
                     call->span);
  }

  tir::Var CreateTIRVarWithUpdatedType(relay::Var rvar) {
    auto it = var_to_var_mapping_.find(rvar);
    ICHECK(it == var_to_var_mapping_.end())
        << "A TIR variable corresponding to the relay variable " << rvar << " already exists.";
    Type var_type = GetVarType(rvar);

    auto iit = arg_mode_states_.find(rvar);
    bool scattered_tensor =
        (iit != arg_mode_states_.end() &&
         (*iit).second->value == static_cast<int>(runtime::vm::DBBatchedArgMode::kScatter));

    std::string name = rvar->vid->name_hint + (scattered_tensor ? "_ptr" : "");
    tir::Var tvar = tir::Var(name, RelayTypeToTIRType(var_type, scattered_tensor));
    var_to_var_mapping_.Set(rvar, tvar);
    return tvar;
  }

  tir::Var batch_size_var_;
  Map<relay::Var, Integer> arg_mode_states_;
};

class BetterCoarsener : public ExprMutator {
 public:
  BetterCoarsener(const IRModule& mod, bool batched_execution, bool scattered_kernels)
      : mod_(mod), batched_execution_(batched_execution), scattered_kernels_(scattered_kernels) {}

  Function Coarsen(const Function& func, bool print) {
    print_ = print;
    auto expr = func->body;
    // expr = transform::ToANormalForm(expr);
    expr = LetLifter()(expr);
    expr = this->VisitExpr(expr);
    return Function(func->params, expr, func->ret_type, func->type_params, func->attrs, func->span);
  }

 private:
  Expr FlattenLets(const Expr& expr, std::vector<std::pair<Var, Expr>>* p_flattened) {
    if (auto ln = expr.as<LetNode>()) {
      p_flattened->push_back(std::make_pair(ln->var, ln->value));
      return FlattenLets(ln->body, p_flattened);
    } else {
      return expr;
    }
  }

  RelayVarSet GetFreeVarsInGroup(std::vector<std::pair<Var, Expr>> bindings) {
    Expr body = MakeConstantScalar(DataType::Int(32), 0);
    for (int i = static_cast<int>(bindings.size()) - 1; i >= 0; --i) {
      auto& p = bindings[i];
      body = Let(p.first, p.second, body);
    }
    return FreeVarsDedup(body);
  }

  tir::PrimFunc AddAttrsToWrapperFunc(const tir::PrimFunc& func, const std::string& name,
                                      bool batched) {
    // format hash as fixed length hex string so it is easier to read
    size_t hash = StructuralHash()(func);
    std::stringstream s;
    s << std::setfill('0') << std::setw(sizeof(size_t) * 2) << std::hex << hash;
    Target target({{String("kind"), String("llvm")}, {String("mcpu"), String("core-avx2")}});
    Map<String, ObjectRef> attrs;
    attrs.Set("global_symbol", runtime::String(name));
    attrs.Set("hash", String(s.str()));
    attrs.Set(tvm::attr::kTarget, target);
    attrs.Set(tir::attr::kDBCoarseWrapperPrimFunc, Integer(1));
    if (batched) {
      attrs.Set(tir::attr::kDBBatchedPrimFunc, Integer(1));
    }
    return WithAttrs(std::move(func), attrs);
  }

  Expr VisitExpr_(const LetNode* op) final {
    std::vector<std::pair<Var, Expr>> flattened;
    auto body = FlattenLets(GetRef<Expr>(op), &flattened);

    size_t start = -1;
    bool in_group = false;
    int op_calls_in_group = 0;
    int last_call_id = -1;
    // end --> begin index mapping for groups
    std::unordered_map<size_t, size_t> groups;

    auto start_or_continue_group = [&](size_t i, bool op_call = false) {
      if (!in_group) {
        start = i;
        in_group = true;
        if (print_) {
          std::cout << "\n\n[CG] GROUPSTART" << std::endl;
        }
      }
      if (op_call) {
        last_call_id = i;
        op_calls_in_group++;
      }
      ICHECK_GE(start, 0);

      if (print_) {
        auto& p = flattened[i];
        std::cout << "[CG] - " << p.first->name_hint() << " = "
                  << PrettyPrint(RemoveOnDeviceCalls(p.second)) << std::endl;
      }
    };

    auto end_group = [&](size_t i) {
      if (in_group) {
        ICHECK_GE(start, 0);
        if (op_calls_in_group > 0) {
          ICHECK_GE(last_call_id, 0);
          groups[last_call_id] = start;
        }
        if (print_) {
          std::cout << "[CG] GROUPEND " << (op_calls_in_group > 0) << std::endl;
        }
        start = -1;
        op_calls_in_group = 0;
        in_group = false;
      }
      ICHECK_EQ(start, -1);
      if (print_) {
        auto& p = flattened[i];
        std::cout << "[CG] x " << p.first->name_hint() << " = "
                  << PrettyPrint(RemoveOnDeviceCalls(p.second)) << std::endl;
      }
    };

    for (size_t i = 0; i < flattened.size(); ++i) {
      auto& p = flattened[i];
      auto& var = p.first;
      auto& value = p.second;
      Expr cleaned_value = value;
      if (auto vn = value.as<CallNode>()) {
        auto on_device_props = GetOnDeviceProps(vn);
        if (on_device_props.body.defined()) {
          cleaned_value = on_device_props.body;
        }
      }

      if (cleaned_value.as<ConstantNode>()) {
        start_or_continue_group(i);
      } else if (cleaned_value.as<TupleNode>()) {
        start_or_continue_group(i);
      } else if (cleaned_value.as<VarNode>()) {
        start_or_continue_group(i);
      } else if (cleaned_value.as<GlobalVarNode>()) {
        end_group(i);
      } else if (cleaned_value.as<FunctionNode>()) {
        end_group(i);
      } else if (auto vn = cleaned_value.as<CallNode>()) {
        if (vn->op == GetInvokeTVMOp()) {
          start_or_continue_group(i, true);
        } else {
          end_group(i);
        }
      } else if (cleaned_value.as<LetNode>()) {
        end_group(i);
      } else if (cleaned_value.as<IfNode>()) {
        end_group(i);
      } else if (cleaned_value.as<OpNode>()) {
        end_group(i);
      } else if (cleaned_value.as<TupleGetItemNode>()) {
        start_or_continue_group(i);
      } else if (cleaned_value.as<RefCreateNode>()) {
        end_group(i);
      } else if (cleaned_value.as<RefReadNode>()) {
        end_group(i);
      } else if (cleaned_value.as<RefWriteNode>()) {
        end_group(i);
      } else if (cleaned_value.as<ConstructorNode>()) {
        end_group(i);
      } else if (cleaned_value.as<MatchNode>()) {
        end_group(i);
      } else {
        end_group(i);
      }
    }
    if (in_group) {
      end_group(flattened.size() - 1);
    }

    if (print_) {
      std::cout << "[CG] Found groups: " << groups.size() << std::endl;
    }

    body = this->VisitExpr(body);

    for (int i = static_cast<int>(flattened.size()) - 1; i >= 0;) {
      auto it = groups.find(i);
      if (it == groups.end()) {
        auto& p = flattened[i];
        body = Let(p.first, this->VisitExpr(p.second), body);
        --i;
      } else {
        auto start = it->second;
        std::vector<std::pair<Var, Expr>> bindings;
        for (size_t j = start; j <= i; ++j) {
          bindings.push_back(
              std::make_pair(flattened[j].first, RemoveOnDeviceCalls(flattened[j].second)));
        }
        auto free_vars = GetFreeVarsInGroup(bindings);

        // if (print_) {
        //   std::cout << "[CG]   GROUP " << std::endl;
        //   for (auto p : bindings) {
        //     std::cout << "[CG]   " << p.first->name_hint() << " = "
        //               << PrettyPrint(RemoveOnDeviceCalls(p.second)) << std::endl;
        //   }
        // }

        auto res = TIRLowererUnbatched(mod_, scattered_kernels_)
                       .LowerToTIR(free_vars, MakeConstantScalar(DataType::Int(32), 0), bindings);

        auto prim_func = res.func;
        auto call_args = res.call_args;
        auto replacement = res.replacement;
        std::string name = "prim_func" + std::to_string(ctr++);
        GlobalVar prim_func_var(name, prim_func->checked_type_);
        auto input_args_tuple = Tuple(call_args);
        auto output_args_tuple = Tuple(Array<Expr>());
        auto call = InvokeTVMOp(prim_func_var, input_args_tuple, output_args_tuple,
                                DictAttrs({{attr::kPrimitive, tvm::Integer(1)},
                                           {tir::attr::kDBCoarseWrapperPrimFunc, Integer(1)}}));
        prim_func = AddAttrsToWrapperFunc(prim_func, name, false);
        prim_funcs_.push_back(std::make_pair(prim_func_var, prim_func));

        // if (groups.size() > 0) {
        //   std::cout << "[CG] Body before adding bindings " <<
        //   PrettyPrint(RemoveOnDeviceCalls(body))
        //             << std::endl;
        // }

        for (int j = static_cast<int>(bindings.size()) - 1; j >= 0; --j) {
          auto& p = bindings[j];
          if (p.second.as<TupleNode>() || p.second.as<VarNode>() ||
              p.second.as<TupleGetItemNode>()) {
            body = Let(p.first, p.second, body);
          }
        }

        body = Let(Var("dummy", VoidType()), call, body);

        i = start - 1;
      }
    }

    // if (groups.size() > 0) {
    // std::cout << "[CG] Body " << PrettyPrint(RemoveOnDeviceCalls(body)) << std::endl;
    // }

    return body;
  }

  bool print_;
  const IRModule& mod_;
  const bool batched_execution_;
  const bool scattered_kernels_;
  static int ctr;

 public:
  std::vector<std::pair<GlobalVar, tir::PrimFunc>> prim_funcs_;
  std::vector<std::pair<GlobalVar, GlobalVar>> batched_func_pairs_;
  std::vector<std::pair<GlobalVar, Array<Integer>>> batched_arg_modes_;
};

int BetterCoarsener::ctr = 0;

IRModule CoarsenGranularity(IRModule& mod, bool batched_execution, bool scattered_kernels) {
  std::cout << "Coarsening now!" << std::endl;
  tvm::Map<GlobalVar, Function> updates;
  tvm::Map<GlobalVar, tir::PrimFunc> new_prim_funcs;

  auto funcs = mod->functions;
  for (const auto& it : funcs) {
    if (const auto* n = it.second.as<FunctionNode>()) {
      ICHECK_EQ(FreeVars(it.second).size(), 0);
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
      Function func = GetRef<Function>(n);

      bool print = (it.first->name_hint == "lstm_cell");
      BetterCoarsener coarsener(mod, batched_execution, scattered_kernels);
      Function ret = coarsener.Coarsen(func, print);

      updates.Set(it.first, ret);

      for (auto it : coarsener.prim_funcs_) {
        new_prim_funcs.Set(it.first, it.second);
      }

      if (batched_execution) {
        for (auto it : coarsener.batched_func_pairs_) {
          mod->UpdateBatchedPrimFunc(it.first, it.second);
        }
        for (auto it : coarsener.batched_arg_modes_) {
          mod->UpdateArgMode(it.first, it.second);
        }
      }

      // auto groups = GroupFinder().FindGroups(func, print);
      // if (print) {
      //   std::cout << "[CG] function " << it.first << std::endl;
      //   std::cout << "[CG]  Groups " << groups.size() << std::endl;
      // }
      // Coarsener coarsener(groups, mod, batched_execution, scattered_kernels);
      // Function ret = Downcast<Function>(coarsener(func));
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
Pass CoarsenPrimitiveFuncGranularity(bool batched_execution, bool scattered_kernels) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext pc) {
    return CoarsenGranularity(m, batched_execution, scattered_kernels);
  };
  return CreateModulePass(pass_func, 0, "CoarsenPrimitiveFuncGranularity", {});
}

TVM_REGISTER_GLOBAL("relay._transform.CoarsenPrimitiveFuncGranularity").set_body_typed(FuseOps);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
