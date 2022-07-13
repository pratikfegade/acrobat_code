
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
#include <tvm/ir/transform.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/vm/dynamic_batching.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include "../../printer/text_printer.h"
#include "../../support/utils.h"
#include "../op/memory/memory.h"
#include "../op/vm/vm.h"
#include "./expr_subst.h"
#include "./function_pointer_analysis.h"
#include "./map_set.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {

typedef std::unordered_set<relay::Var, ObjectPtrHash, ObjectPtrEqual> RelayVarSet;
typedef std::unordered_map<tir::Var, runtime::vm::DBArgAccessMode, ObjectPtrHash, ObjectPtrEqual>
    AccessModesMap;
typedef std::unordered_map<GlobalVar, AccessModesMap, ObjectPtrHash, ObjectPtrEqual>
    FunctionsAccessModesMap;

class LeafTensorAccessModeCalculator : public tir::StmtExprVisitor {
 public:
  LeafTensorAccessModeCalculator(const tir::PrimFunc& func) : func_(func) {}

  AccessModesMap Compute() {
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
    if (op->scatter_buffer_var.defined()) {
      MergeAndSet(op->scatter_buffer_var, runtime::vm::kInput);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tir::StoreNode* op) {
    MergeAndSet(op->buffer_var, runtime::vm::kOutput);
    if (op->scatter_buffer_var.defined()) {
      MergeAndSet(op->scatter_buffer_var, runtime::vm::kOutput);
    }
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

std::pair<Array<Integer>, AccessModesMap> ComputeAndVerifyAccessModes(const tir::PrimFunc& func) {
  auto original_access_modes =
      func->GetAttr(tir::attr::kDBArgAccessModes, Array<Integer>()).value();
  auto access_modes_map = LeafTensorAccessModeCalculator(func).Compute();
  if (original_access_modes.size() > 0) {
    ICHECK_EQ(original_access_modes.size(), func->params.size());

    auto access_modes_map = LeafTensorAccessModeCalculator(func).Compute();
    Array<Integer> computed_access_modes;
    for (size_t i = 0; i < func->params.size(); ++i) {
      auto param = func->params[i];
      auto computed_access_mode = static_cast<int>(
          access_modes_map.count(param) ? access_modes_map.at(param) : runtime::vm::kInput);
      computed_access_modes.push_back(computed_access_mode);
      ICHECK_LE(original_access_modes[i]->value, computed_access_mode);
    }
    return std::make_pair(original_access_modes, access_modes_map);
  } else {
    Array<Integer> access_modes;
    for (size_t i = 0; i < func->params.size(); ++i) {
      auto param = func->params[i];
      access_modes.push_back(Integer(static_cast<int>(
          access_modes_map.count(param) ? access_modes_map.at(param) : runtime::vm::kInput)));
    }
    return std::make_pair(access_modes, access_modes_map);
  }
}

class CoarsenedTensorAccessModeCalculator : public tir::StmtExprVisitor {
 public:
  CoarsenedTensorAccessModeCalculator(const IRModule& mod) : mod_(mod) {}

  AccessModesMap Compute(const tir::Stmt& body) {
    // std::cout << "[COR] Computing access modes\n" << body << std::endl;
    body_ = body;
    VisitStmt(body);
    return access_modes_map_;
  }

 private:
  void MergeAndSet(const tir::Var& var, runtime::vm::DBArgAccessMode mode) {
    // std::cout << "[COR]  Setting mode " << var << " " << mode << std::endl;
    auto iit = access_modes_map_.find(var);
    if (iit == access_modes_map_.end()) {
      access_modes_map_[var] = mode;
    } else {
      auto old_mode = iit->second;
      if (old_mode != mode) {
        // ICHECK(old_mode == runtime::vm::kOutput && mode == runtime::vm::kInput)
        //     << "Modes found: " << old_mode << " " << mode << ". Reused tensor: " << var << " in "
        //     << body_;
        access_modes_map_[var] = runtime::vm::kInputOutput;
      }
    }
  }

  void VisitExpr_(const tir::CallNode* op) {
    // std::cout << "[COR]  Visiting call " << GetRef<PrimExpr>(op) << std::endl;
    ICHECK_EQ(op->op, tir::builtin::tvm_call_unpacked_from_packed());
    auto base_leaf_callee = mod_->Lookup(Downcast<tir::StringImm>(op->args[0])->value);
    ICHECK(base_leaf_callee.as<tir::PrimFuncNode>());
    auto leaf_callee = Downcast<tir::PrimFunc>(base_leaf_callee);

    auto res = ComputeAndVerifyAccessModes(leaf_callee);
    auto func_access_modes = res.first;
    auto func_access_modes_map = res.second;
    // std::cout << "[COR]   Callee " << leaf_callee << std::endl;

    size_t ctr = 1;
    for (size_t i = 1; i <= leaf_callee->params.size(); ++i) {
      auto arg = op->args[ctr];
      auto param = leaf_callee->params[i - 1];
      auto it = func_access_modes_map.find(param);
      if (it != func_access_modes_map.end()) {
        auto mode = static_cast<runtime::vm::DBArgAccessMode>(func_access_modes[i - 1]->value);
        MergeAndSet(Downcast<tir::Var>(arg), mode);
        ctr++;
      }
    }
  }

  const IRModule mod_;
  AccessModesMap access_modes_map_;
  tir::Stmt body_;
};

void FlattenVar(const relay::Var& var, Map<relay::Var, Array<Expr>>* p_tuple_var_values,
                std::vector<relay::Var>* p_flattened_free_vars,
                std::vector<Expr>* p_flattened_call_args) {
  Map<relay::Var, Array<Expr>>& tuple_var_values = *p_tuple_var_values;
  std::vector<relay::Var>& flattened_free_vars = *p_flattened_free_vars;
  std::vector<Expr>& flattened_call_args = *p_flattened_call_args;

  if (auto ttn = GetVarType(var).as<TupleTypeNode>()) {
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
  Array<Integer> arg_modes;
  std::vector<int> param_positions;
  Map<GlobalVar, tir::PrimFunc> contiguous_functions;
};

class AbstractTIRLowerer {
 public:
  AbstractTIRLowerer(const IRModule& mod, const FunctionsAccessModesMap& prim_funcs_access_modes,
                     Map<relay::Var, Array<Expr>> tuple_var_values,
                     Map<relay::Var, tir::Var> var_to_var_mapping, const std::string& name,
                     bool scattered_kernels)
      : mod_(mod),
        prim_funcs_access_modes_(prim_funcs_access_modes),
        tuple_var_values_(tuple_var_values),
        var_to_var_mapping_(var_to_var_mapping),
        name_(name),
        scattered_kernels_(scattered_kernels) {}

  ~AbstractTIRLowerer() {}

  virtual TIRLowererResult LowerToTIR(const std::vector<Var>& flattened_free_vars,
                                      const std::vector<Expr>& calls) = 0;

 protected:
  virtual PrimExpr ConvertTVMOpInvoke(const relay::CallNode* call) = 0;

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
    } else if (expr.as<relay::ConstantNode>()) {
      tir_fields.push_back(expr);
    } else {
      ICHECK(false) << "Don't know how to flatten " << expr;
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
  const FunctionsAccessModesMap& prim_funcs_access_modes_;
  Map<relay::Var, Array<Expr>> tuple_var_values_;
  Map<relay::Var, tir::Var> var_to_var_mapping_;
  const std::string& name_;
  bool scattered_kernels_;
};

struct GroupSchedulerResult {
  std::vector<std::vector<Expr>> groups;
  Map<relay::Var, tir::Var> var_to_var_mapping;
  std::vector<bool> increment_depth;
};

class GroupStaticScheduler : public AbstractTIRLowerer {
 public:
  GroupStaticScheduler(const IRModule& mod, const FunctionsAccessModesMap& prim_funcs_access_modes,
                       Map<relay::Var, Array<Expr>> tuple_var_values, bool scattered_kernels)
      : AbstractTIRLowerer(mod, prim_funcs_access_modes, tuple_var_values, {}, "",
                           scattered_kernels) {}

  GroupSchedulerResult Schedule(const std::vector<Var>& flattened_free_vars,
                                const std::vector<std::pair<relay::Var, Expr>>& bindings) {
    std::vector<tir::Var> prim_func_params_vec;
    std::vector<Type> prim_func_param_types_vec;
    std::vector<int> param_poses;
    for (size_t i = 0; i < flattened_free_vars.size(); ++i) {
      auto rvar = flattened_free_vars[i];
      prim_func_params_vec.push_back(CreateTIRVar(rvar));
      prim_func_param_types_vec.push_back(GetVarType(rvar));
      param_poses.push_back(i);
    }

    std::vector<Expr> calls;
    for (auto pair : bindings) {
      auto rvar = pair.first;
      auto rvalue = pair.second;

      if (auto call = rvalue.as<CallNode>()) {
        ICHECK(call->op == GetInvokeTVMOp()) << rvalue;
        calls.push_back(HandleTVMOpInvoke(call));
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

    std::vector<std::vector<Expr>> groups;
    std::vector<bool> increment_depth;
    if (transform::PassContext::Current()
            ->GetConfig<Bool>("relay.db_perform_static_scheduling", Bool(false))
            .value()) {
      std::unordered_map<const Object*, int> depths;
      std::vector<std::unordered_map<const Object*, std::vector<Expr>>> depth2func2calls;
      for (auto param : flattened_free_vars) {
        depths[param.get()] = 0;
      }
      for (auto e : calls) {
        auto call = e.as<CallNode>();
        int depth = 0;
        for (auto arg : call->args[0].as<TupleNode>()->fields) {
          ICHECK(depths.count(arg.get())) << arg;
          depth = std::max(depth, depths.at(arg.get()));
        }
        depth2func2calls.resize(depth + 1);
        depth2func2calls[depth][call->op.get()].push_back(e);
        depth++;
        for (auto arg : call->args[1].as<TupleNode>()->fields) {
          depths[arg.get()] = depth;
        }
      }
      std::vector<Expr> current_group;
      for (auto func2calls : depth2func2calls) {
        for (auto kv : func2calls) {
          auto calls = kv.second;
          if (calls.size() > 1) {
            if (current_group.size() > 0) {
              groups.push_back(current_group);
              increment_depth.push_back(true);
              current_group = std::vector<Expr>();
            }
            for (size_t i = 0; i < calls.size(); ++i) {
              groups.push_back({calls[i]});
              increment_depth.push_back(i == (calls.size() - 1));
            }
          } else {
            ICHECK_EQ(calls.size(), 1);
            current_group.push_back(calls[0]);
          }
        }
      }
      if (current_group.size() > 0) {
        groups.push_back(current_group);
        increment_depth.push_back(true);
      }
    } else {
      groups.push_back(calls);
      increment_depth.push_back(true);
    }
    return {groups, var_to_var_mapping_, increment_depth};
  }

  TIRLowererResult LowerToTIR(const std::vector<Var>& flattened_free_vars,
                              const std::vector<Expr>& calls) {
    return {};
  }

 private:
  PrimExpr ConvertTVMOpInvoke(const relay::CallNode* call) final { return {}; }

  Expr HandleTVMOpInvoke(const relay::CallNode* call) {
    // std::cout << "[CG] Lowering call " << GetRef<Expr>(call) << std::endl;
    auto callee_gv = Downcast<GlobalVar>(call->args[0]);
    auto callee = Downcast<tir::PrimFunc>(mod_->Lookup(callee_gv));
    ICHECK(prim_funcs_access_modes_.count(callee_gv)) << callee_gv << " " << callee_gv.get();
    auto access_modes_map = prim_funcs_access_modes_.at(callee_gv);

    Array<Expr> args;
    ICHECK_GE(call->args.size(), 3) << GetRef<Expr>(call);
    int ctr = 0;
    auto push_args = [&](const Expr& tuple) {
      Array<Expr> fields;
      for (auto arg : FlattenTuple(tuple)) {
        // std::cout << "[CG]  arg " << arg << std::endl;
        if (access_modes_map.count(callee->params[ctr++])) {
          fields.push_back(arg);
        }
      }
      args.push_back(Tuple(fields));
    };
    push_args(call->args[1]);
    push_args(call->args[2]);
    return Call(callee_gv, args, call->attrs, call->type_args, call->span);
  }
};

class TIRLowererUnbatched : public AbstractTIRLowerer {
 public:
  TIRLowererUnbatched(const IRModule& mod, const FunctionsAccessModesMap& prim_funcs_access_modes,
                      Map<relay::Var, Array<Expr>> tuple_var_values,
                      Map<relay::Var, tir::Var> var_to_var_mapping, const std::string& name,
                      bool scattered_kernels)
      : AbstractTIRLowerer(mod, prim_funcs_access_modes, tuple_var_values, var_to_var_mapping, name,
                           scattered_kernels) {}

  TIRLowererResult LowerToTIR(const std::vector<Var>& flattened_free_vars,
                              const std::vector<Expr>& calls) final {
    std::vector<tir::Var> prim_func_params_vec;
    std::vector<Type> prim_func_param_types_vec;
    std::vector<int> param_poses;
    for (size_t i = 0; i < flattened_free_vars.size(); ++i) {
      auto rvar = flattened_free_vars[i];
      prim_func_params_vec.push_back(GetTIRVar(rvar));
      prim_func_param_types_vec.push_back(GetVarType(rvar));
      param_poses.push_back(i);
    }

    Array<tir::Stmt> tir_stmts;
    for (auto expr : calls) {
      auto call = expr.as<CallNode>();
      ICHECK(call) << expr;
      tir_stmts.push_back(tir::Evaluate(ConvertTVMOpInvoke(call)));
    }

    tir::Stmt prim_func_body = tir::SeqStmt(tir_stmts);

    AccessModesMap access_modes_map =
        CoarsenedTensorAccessModeCalculator(mod_).Compute(prim_func_body);

    auto get_access_mode = [&access_modes_map](const tir::Var& var) {
      auto it = access_modes_map.find(var);
      auto mode = tvm::runtime::vm::kUnused;
      if (it != access_modes_map.end()) {
        mode = it->second;
      }
      return mode;
    };

    auto sorting_lambda = [&](int idx1, int idx2) {
      const tir::Var& var1 = prim_func_params_vec[idx1];
      const tir::Var& var2 = prim_func_params_vec[idx2];
      auto mode1 = get_access_mode(var1);
      auto mode2 = get_access_mode(var2);
      if (mode1 == mode2) {
        auto name1 = var1->name_hint;
        auto name2 = var2->name_hint;
        return (name1.compare(name2) < 0);
      } else {
        return (static_cast<int>(mode1) < static_cast<int>(mode2));
      }
    };

    std::sort(param_poses.begin(), param_poses.end(), sorting_lambda);

    Array<tir::Var> prim_func_params;
    Array<Type> prim_func_param_types;
    Array<Integer> prim_func_arg_access_modes;
    for (size_t j = 0; j < param_poses.size(); ++j) {
      auto pos = param_poses[j];
      auto& param = prim_func_params_vec[pos];
      if (get_access_mode(param) == tvm::runtime::vm::kUnused) {
        param_poses[j] = -1;
      } else {
        auto& type = prim_func_param_types_vec[pos];
        prim_func_params.push_back(param);
        prim_func_param_types.push_back(type);
        prim_func_arg_access_modes.push_back(Integer(static_cast<int>(get_access_mode(param))));
      }
    }

    auto func = tir::PrimFunc(prim_func_params, prim_func_body, VoidType());
    func->checked_type_ = FuncType(prim_func_param_types, VoidType(), {}, {});
    func = WithAttr(func, tir::attr::kDBArgAccessModes, prim_func_arg_access_modes);
    return TIRLowererResult({func, Array<Integer>(), param_poses});
  }

 private:
  PrimExpr ConvertTVMOpInvoke(const relay::CallNode* call) final {
    auto callee_gv = Downcast<GlobalVar>(call->op);
    Array<PrimExpr> args;
    args.push_back(tir::StringImm(callee_gv->name_hint));

    ICHECK_GE(call->args.size(), 2) << GetRef<Expr>(call);
    auto push_args = [&](const Expr& expr) {
      for (auto arg : expr.as<TupleNode>()->fields) {
        ICHECK(arg.as<relay::VarNode>()) << GetRef<Expr>(call);
        args.push_back(GetTIRVar(Downcast<relay::Var>(arg)));
      }
    };
    push_args(call->args[0]);
    push_args(call->args[1]);
    return tir::Call(DataType::Int(32), tir::builtin::tvm_call_unpacked_from_packed(), args,
                     call->span);
  }
};

using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;
using PointsToMap = std::unordered_map<Var, ExprSet, ObjectPtrHash, ObjectPtrEqual>;

struct MakeContiguousResult {
  tir::PrimFunc contiguous_function;
  Array<Integer> reuse_modes;
};

class TIRContiguousTensorMutator : public tir::StmtExprMutator {
 public:
  TIRContiguousTensorMutator(const tir::PrimFunc& func,
                             const std::unordered_set<std::string>& to_make_contiguous,
                             const Array<Integer>& orig_reuse_modes)
      : func_(func), to_make_contiguous_(to_make_contiguous), orig_reuse_modes_(orig_reuse_modes) {}

  MakeContiguousResult MakeContiguous() {
    std::cout << "[MAKE CONTIG] " << std::endl;
    for (auto var_name : to_make_contiguous_) {
      std::cout << "[TO MAKE CONTIG]  " << var_name << std::endl;
    }
    std::cout << func_ << std::endl;
    std::cout << func_->scatter_buffer_map << std::endl;
    auto body = this->VisitStmt(func_->body);

    Array<tir::Var> params;
    for (size_t i = 0; i < func_->params.size(); ++i) {
      params.push_back(func_->params[i]);
      if (to_make_contiguous_.count(func_->params[i]->name_hint)) {
        ++i;
      }
    }

    Map<tir::Var, tir::Buffer> scatter_buffer_map;
    for (auto kv : func_->scatter_buffer_map) {
      if (!to_make_contiguous_.count(kv.first->name_hint)) {
        scatter_buffer_map.Set(kv.first, kv.second);
      }
    }

    auto orig_io_modes =
        func_->GetAttr<Array<Integer>>("db.args_access_modes", Array<Integer>()).value();

    Array<Integer> updated_reuse_modes;
    Array<Integer> updated_io_modes;
    updated_io_modes.push_back(orig_io_modes[0]);
    int ctr = 0;
    for (size_t i = 1; i < func_->params.size();) {
      switch (orig_reuse_modes_[ctr++]) {
        case tvm::runtime::vm::kIgnore:
          updated_reuse_modes.push_back(Integer(static_cast<int>(tvm::runtime::vm::kIgnore)));
          updated_io_modes.push_back(orig_io_modes[i]);
          break;
        case tvm::runtime::vm::kReuse:
          updated_reuse_modes.push_back(Integer(static_cast<int>(tvm::runtime::vm::kReuse)));
          updated_io_modes.push_back(orig_io_modes[i]);
          i += 1;
          break;
        case tvm::runtime::vm::kScatter:
          updated_io_modes.push_back(orig_io_modes[i]);
          if (to_make_contiguous_.count(func_->params[i]->name_hint)) {
            updated_reuse_modes.push_back(Integer(static_cast<int>(tvm::runtime::vm::kContiguous)));
            updated_io_modes.push_back(orig_io_modes[i + 1]);
          } else {
            updated_reuse_modes.push_back(Integer(static_cast<int>(tvm::runtime::vm::kScatter)));
          }
          i += 2;
          break;
        case tvm::runtime::vm::kContiguous:
          ICHECK(false);
          i += 1;
          break;
        case tvm::runtime::vm::kConcat:
          updated_reuse_modes.push_back(Integer(static_cast<int>(tvm::runtime::vm::kConcat)));
          updated_io_modes.push_back(orig_io_modes[i]);
          i += 1;
          break;
      }
    }

    auto contiguous_function = tir::PrimFunc(params, body, func_->ret_type, func_->buffer_map,
                                             scatter_buffer_map, func_->attrs, func_->span);

    contiguous_function = WithAttr(contiguous_function, "db.args_access_modes", updated_io_modes);
    std::cout << "[MAKE CONTIG RES] " << std::endl;
    std::cout << contiguous_function << std::endl;
    return {contiguous_function, updated_reuse_modes};
  }

  PrimExpr VisitExpr_(const tir::LoadNode* op) override {
    auto expr = StmtExprMutator::VisitExpr_(op);
    auto load = expr.as<tir::LoadNode>();
    ICHECK(load) << load << " " << GetRef<PrimExpr>(op);
    // std::cout << "[LOAD] " << GetRef<PrimExpr>(load) << " " << load->buffer_var << " "
    // << load->buffer_var.get() << " " << op->buffer_var.get() << std::endl;
    if (load->scatter_buffer_var.defined() &&
        to_make_contiguous_.count(load->buffer_var->name_hint)) {
      // auto res = tir::Load(load->dtype, load->buffer_var, load->index, load->predicate,
      // load->scatter_buffer_var, load->scatter_batch_index,
      // load->scatter_elem_index, load->span);
      auto res = tir::Load(load->dtype, load->buffer_var, load->index, load->predicate);
      // std::cout << "[UNSCATTERING] " << res << std::endl;
      return res;
    }
    return GetRef<PrimExpr>(load);
  }

  tir::Stmt VisitStmt_(const tir::StoreNode* op) override {
    auto stmt = StmtExprMutator::VisitStmt_(op);
    auto store = stmt.as<tir::StoreNode>();
    ICHECK(store) << stmt << " " << GetRef<tir::Stmt>(op);
    if (store->scatter_buffer_var.defined() &&
        to_make_contiguous_.count(store->buffer_var->name_hint)) {
      // return tir::Store(store->buffer_var, store->value, store->index, store->predicate,
      // store->scatter_buffer_var, store->scatter_batch_index,
      // store->scatter_elem_index, store->span);
      return tir::Store(store->buffer_var, store->value, store->index, store->predicate);
    }
    return GetRef<tir::Stmt>(store);
  }

  const tir::PrimFunc& func_;
  const std::unordered_set<std::string>& to_make_contiguous_;
  const Array<Integer>& orig_reuse_modes_;
};

class TIRLowererBatched : public AbstractTIRLowerer {
 public:
  TIRLowererBatched(const IRModule& mod, const FunctionsAccessModesMap& prim_funcs_access_modes,
                    Map<relay::Var, Array<Expr>> tuple_var_values,
                    Map<relay::Var, tir::Var> var_to_var_mapping, const std::string& name,
                    bool scattered_kernels)
      : AbstractTIRLowerer(mod, prim_funcs_access_modes, tuple_var_values, var_to_var_mapping, name,
                           scattered_kernels) {
    batch_size_var_ = tir::Var("batch_size", DataType::Int(32));
  }

  TIRLowererResult LowerToTIR(const std::vector<Var>& flattened_free_vars,
                              const std::vector<Expr>& calls) final {
    for (auto var : flattened_free_vars) {
      std::cout << "[FREE_VAR] " << var->vid->name_hint << std::endl;
    }

    Map<Var, Bool> contiguous_tensors;

    for (auto expr : calls) {
      std::cout << "[GROUP CALL] " << PrettyPrint(expr) << std::endl;

      auto call = expr.as<CallNode>();
      for (auto arg : FlattenTuple(call->args[1])) {
        ICHECK(arg.as<relay::VarNode>());
        auto arg_var = Downcast<relay::Var>(arg);
        MapSet::Insert(contiguous_tensors, arg_var);
      }
    }

    for (auto kv : contiguous_tensors) {
      std::cout << "[CONTIGUOUS] " << kv.first->vid->name_hint << std::endl;
    }

    bool print = false;
    if (print) {
      std::cout << "[CALL] YELLOWYELLOWYELLOWYELLOWYELLOWYELLOWYELLOWYELLOW" << std::endl;
      for (auto call : calls) {
        std::cout << "[CALL] " << PrettyPrint(call) << std::endl;
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

    for (auto it = calls.rbegin(); it != calls.rend(); ++it) {
      auto call = it->as<CallNode>();
      ICHECK(call);
      if (print) {
        std::cout << "[CG]  Visiting: " << *it << " " << FlattenTuple(call->args[1]) << std::endl;
      }
      auto callee_gv = Downcast<GlobalVar>(call->op);
      auto iit = mod_->batched_prim_funcs.find(callee_gv);
      ICHECK(iit != mod_->batched_prim_funcs.end()) << callee_gv->name_hint;
      auto batched_callee_gv = (*iit).second;
      auto iiit = mod_->batched_arg_modes.find(batched_callee_gv);
      ICHECK(iiit != mod_->batched_arg_modes.end()) << batched_callee_gv->name_hint;
      auto& arg_modes = (*iiit).second;
      int idx = 0;
      auto handle_arg = [&](const Expr& tuple) {
        for (auto arg : FlattenTuple(tuple)) {
          ICHECK(arg.as<relay::VarNode>());
          auto arg_var = Downcast<relay::Var>(arg);
          auto arg_mode = arg_modes[idx++]->value;
          if (arg_mode == static_cast<int>(runtime::vm::DBBatchedArgMode::kIgnore)) {
            continue;
          }
          merge_and_set_arg_mode(arg_var, arg_mode);
        }
      };
      handle_arg(call->args[0]);
      handle_arg(call->args[1]);
    }

    /* Final forward pass *************************************************************/
    var_to_var_mapping_.clear();

    Array<tir::Var> prim_func_params;
    Array<Type> prim_func_param_types;
    Array<Integer> prim_func_arg_modes;
    prim_func_params.push_back(batch_size_var_);
    prim_func_param_types.push_back(batch_size_var_->type_annotation);
    for (auto rvar : flattened_free_vars) {
      if (print) {
        std::cout << "[CG]  Creating updated var for: " << rvar->vid->name_hint << std::endl;
      }

      auto param = CreateTIRVarWithUpdatedType(rvar);
      prim_func_param_types.push_back(GetVarType(rvar));

      auto iit = arg_mode_states_.find(rvar);
      auto arg_mode = (iit != arg_mode_states_.end()
                           ? (*iit).second
                           : (scattered_kernels_ ? runtime::vm::DBBatchedArgMode::kScatter
                                                 : runtime::vm::DBBatchedArgMode::kConcat));

      if (arg_mode != runtime::vm::DBBatchedArgMode::kIgnore) {
        prim_func_params.push_back(param);
        std::cout << "[ARG MODE] " << rvar->vid->name_hint << " " << arg_mode << std::endl;
        if (MapSet::Contains(contiguous_tensors, rvar) &&
            arg_mode == runtime::vm::DBBatchedArgMode::kScatter) {
          std::cout << "[ARG MODE]   Can be contiguous" << std::endl;
          prim_func_arg_modes.push_back(runtime::vm::DBBatchedArgMode::kContiguous);
        } else {
          prim_func_arg_modes.push_back(arg_mode);
          MapSet::Remove(contiguous_tensors, rvar);
        }
      }
      if (print) {
        std::cout << "[CG]  ArgMode: " << arg_mode << " " << (iit != arg_mode_states_.end())
                  << std::endl;
      }
    }

    /* Make Contiguous *************************************************************/
    std::unordered_map<std::string, std::string> contiguous_specification_strings;
    std::unordered_map<std::string, std::unordered_set<std::string>> contiguous_specifications;

    for (auto it = calls.rbegin(); it != calls.rend(); ++it) {
      auto call = it->as<CallNode>();
      ICHECK(call);
      if (print) {
        std::cout << "[CG]  Visiting: " << *it << " " << FlattenTuple(call->args[1]) << std::endl;
      }
      auto callee_gv = Downcast<GlobalVar>(call->op);
      auto iit = mod_->batched_prim_funcs.find(callee_gv);
      ICHECK(iit != mod_->batched_prim_funcs.end()) << callee_gv->name_hint;
      auto batched_callee_gv = (*iit).second;
      auto batched_callee = Downcast<tir::PrimFunc>(mod_->Lookup(batched_callee_gv));
      auto iiit = mod_->batched_arg_modes.find(batched_callee_gv);
      ICHECK(iiit != mod_->batched_arg_modes.end()) << batched_callee_gv->name_hint;
      auto& arg_modes = (*iiit).second;

      std::unordered_set<std::string> to_make_contiguous;
      std::stringstream ss;
      int param_idx = 1;
      int dedup_idx = 0;
      auto handle_arg = [&](const Expr& tuple) {
        for (auto arg : FlattenTuple(tuple)) {
          ICHECK(arg.as<relay::VarNode>());
          auto arg_var = Downcast<relay::Var>(arg);
          if (MapSet::Contains(contiguous_tensors, arg_var)) {
            to_make_contiguous.insert(batched_callee->params[param_idx]->name_hint);
            ss << batched_callee->params[param_idx]->name_hint << "***";
          }
          if (arg_modes[dedup_idx++]->value ==
              static_cast<int>(runtime::vm::DBBatchedArgMode::kScatter)) {
            param_idx += 2;
          } else {
            param_idx++;
          }
        }
      };
      handle_arg(call->args[0]);
      handle_arg(call->args[1]);

      {
        auto it = contiguous_specification_strings.find(batched_callee_gv->name_hint);
        if (it != contiguous_specification_strings.end()) {
          ICHECK_EQ(ss.str(), it->second);
        } else {
          contiguous_specification_strings[batched_callee_gv->name_hint] = ss.str();
        }
      }
      contiguous_specifications[batched_callee_gv->name_hint] = to_make_contiguous;
    }

    for (auto kv : contiguous_specifications) {
      auto batched_name = kv.first;
      auto to_make_contiguous = kv.second;
      auto batched_callee_gv = mod_->GetGlobalVar(batched_name);
      auto batched_callee = Downcast<tir::PrimFunc>(mod_->Lookup(batched_callee_gv));
      auto iiit = mod_->batched_arg_modes.find(batched_callee_gv);
      ICHECK(iiit != mod_->batched_arg_modes.end()) << batched_callee_gv->name_hint;
      auto& arg_modes = (*iiit).second;

      auto contiguous_res =
          TIRContiguousTensorMutator(batched_callee, to_make_contiguous, arg_modes)
              .MakeContiguous();
      auto batched_contiguous_callee = contiguous_res.contiguous_function;
      auto updated_reuse_modes = contiguous_res.reuse_modes;
      mod_->batched_arg_modes.Set(batched_callee_gv, updated_reuse_modes);
      ICHECK(!contiguous_functions_.count(batched_callee_gv))
          << "Function has been made contiguous before";
      contiguous_functions_.Set(batched_callee_gv, batched_contiguous_callee);
    }

    Array<tir::Stmt> tir_stmts;
    for (auto call_expr : calls) {
      auto call = call_expr.as<CallNode>();
      ICHECK(call);
      tir_stmts.push_back(tir::Evaluate(ConvertTVMOpInvoke(call)));
    }

    /* Epilogue: Create function *************************************************************/

    tir::Stmt prim_func_body = tir::SeqStmt(tir_stmts);

    auto func = tir::PrimFunc(prim_func_params, prim_func_body, VoidType());
    func->checked_type_ = FuncType(prim_func_param_types, VoidType(), {}, {});
    return TIRLowererResult({func, prim_func_arg_modes, {}, contiguous_functions_});
  }

 private:
  PrimExpr ConvertTVMOpInvoke(const relay::CallNode* call) final {
    auto callee_gv = Downcast<GlobalVar>(call->op);
    std::string batched_name = runtime::vm::GetBatchedName(callee_gv->name_hint);
    Array<PrimExpr> args;
    args.push_back(tir::StringImm(batched_name));
    args.push_back(batch_size_var_);

    auto push_args = [&](const Expr& tuple) {
      for (auto arg : FlattenTuple(tuple)) {
        ICHECK(arg.as<relay::VarNode>());
        args.push_back(GetTIRVar(Downcast<relay::Var>(arg)));
      }
    };
    ICHECK_GE(call->args.size(), 2) << GetRef<Expr>(call);
    push_args(call->args[0]);
    push_args(call->args[1]);
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

    std::string name = rvar->vid->name_hint + (scattered_tensor ? "_ptr" : "_cc");
    tir::Var tvar = tir::Var(name, RelayTypeToTIRType(var_type, scattered_tensor));
    var_to_var_mapping_.Set(rvar, tvar);
    return tvar;
  }

  tir::Var batch_size_var_;
  Map<relay::Var, Integer> arg_mode_states_;
  Map<GlobalVar, tir::PrimFunc> contiguous_functions_;
};

class PointerAnalysis
    : public ExprFunctor<std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>(const Expr&)> {
 public:
  PointerAnalysis() {}

  PointsToMap RunPointerAnalysis(const Function& func) {
    this->VisitExpr(func->body);
    return points_to;
  }

 private:
  ExprSet VisitExpr_(const ConstantNode* op) { return ExprSet(); };

  ExprSet VisitExpr_(const TupleNode* op) {
    ExprSet res;
    for (auto field : op->fields) {
      auto field_res = this->VisitExpr(field);
      res.insert(field_res.begin(), field_res.end());
    }
    return res;
  };

  ExprSet VisitExpr_(const VarNode* op) { return points_to[GetRef<Var>(op)]; };

  ExprSet VisitExpr_(const GlobalVarNode* op) { return ExprSet(); };

  ExprSet VisitExpr_(const FunctionNode* op) { return ExprSet(); };

  ExprSet VisitExpr_(const CallNode* op) {
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      return VisitExpr(on_device_props.body);
    }

    if (op->op == MemoryAllocTensorOp()) {
      return {GetRef<Expr>(op)};
    }
    return ExprSet();
  };

  ExprSet VisitExpr_(const LetNode* op) {
    auto value_res = this->VisitExpr(op->value);
    points_to[op->var].insert(value_res.begin(), value_res.end());
    // std::cout << "PA " << op->var->vid->name_hint << std::endl;
    for (auto e : value_res) {
      // std::cout << "  VAL " << PrettyPrint(e) << std::endl;
    }
    return VisitExpr(op->body);
  };

  ExprSet VisitExpr_(const IfNode* op) {
    ExprSet res;
    this->VisitExpr(op->cond);
    auto then_res = this->VisitExpr(op->true_branch);
    auto else_res = this->VisitExpr(op->false_branch);
    res.insert(then_res.begin(), then_res.end());
    res.insert(else_res.begin(), else_res.end());
    return res;
  };

  ExprSet VisitExpr_(const OpNode* op) { return ExprSet(); };

  ExprSet VisitExpr_(const TupleGetItemNode* op) { return VisitExpr(op->tuple); };

  ExprSet VisitExpr_(const RefCreateNode* op) {
    ICHECK(false);
    return ExprSet();
  };

  ExprSet VisitExpr_(const RefReadNode* op) {
    ICHECK(false);
    return ExprSet();
  };

  ExprSet VisitExpr_(const RefWriteNode* op) {
    ICHECK(false);
    return ExprSet();
  };

  ExprSet VisitExpr_(const ConstructorNode* op) {
    ICHECK(false);
    return ExprSet();
  };

  ExprSet VisitExpr_(const MatchNode* op) {
    auto value_res = this->VisitExpr(op->data);

    ExprSet res;
    for (auto clause : op->clauses) {
      auto pattern_vars = CollectPatternVars(clause->lhs);
      for (auto& var : pattern_vars) {
        points_to[var].insert(value_res.begin(), value_res.end());
      }
      auto clause_res = this->VisitExpr(clause->rhs);
      res.insert(clause_res.begin(), clause_res.end());
    }
    return res;
  };

  PointsToMap points_to;
};

class Coarsener : public ExprMutator {
 public:
  Coarsener(const IRModule& mod, const FunctionsAccessModesMap& prim_funcs_access_modes,
            bool batched_execution, bool scattered_kernels)
      : mod_(mod),
        prim_funcs_access_modes_(prim_funcs_access_modes),
        batched_execution_(batched_execution),
        scattered_kernels_(scattered_kernels) {}

  Function Coarsen(const Function& func, bool print) {
    print_ = print;
    auto expr = func->body;
    // expr = transform::ToANormalForm(expr);
    expr = LiftLetsOutOfValues(expr);
    // points_to_map_ = PointerAnalysis().RunPointerAnalysis(func);
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

  RelayVarSet GetFreeVarsInGroup(std::vector<Expr> bindings) {
    Expr body = MakeConstantScalar(DataType::Int(32), 0);
    for (int i = static_cast<int>(bindings.size()) - 1; i >= 0; --i) {
      auto& call = bindings[i];
      body = Let(Var("dummy" + std::to_string(i), VoidType()), call, body);
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

  int GetStaticGraphDepth(const CallNode* op) {
    if (op->attrs.defined() && op->attrs->IsInstance<DictAttrsNode>()) {
      auto dict_attrs = Downcast<DictAttrs>(op->attrs);
      return dict_attrs.GetAttr(tir::attr::kDBGraphDepth, Integer(-1)).value()->value;
    }
    return -1;
  }

  bool IsStorageType(const Type& type) {
    if (auto tc = type.as<TypeCallNode>()) {
      return IsStorageType(tc->func);
    } else if (auto td = type.as<TypeDataNode>()) {
      return td->header->name_hint == "Storage";
    } else if (auto tv = type.as<GlobalTypeVarNode>()) {
      return tv->name_hint == "Storage";
    } else {
      return false;
    }
  }

  std::vector<std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>> EscapeAnalysis(
      const std::vector<std::pair<Var, Expr>>& flattened, const Expr& body) {
    std::vector<std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>> live_vars(flattened.size());
    live_vars[flattened.size() - 1] = FreeVarsDedup(body);
    for (int i = flattened.size() - 1; i >= 1; --i) {
      auto& after_live = live_vars[i];
      std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> before_live(after_live.begin(),
                                                                         after_live.end());
      auto& p = flattened[i];
      auto& value = p.second;
      Expr cleaned_value = value;
      if (auto vn = value.as<CallNode>()) {
        auto on_device_props = GetOnDeviceProps(vn);
        if (on_device_props.body.defined()) {
          cleaned_value = on_device_props.body;
        }
      }
      before_live.erase(p.first);
      auto free_vars = FreeVarsDedup(cleaned_value);
      before_live.insert(free_vars.begin(), free_vars.end());

      live_vars[i - 1] = before_live;
      // std::cout << "Live after " << p.first->vid->name_hint << std::endl;
      for (auto var : after_live) {
        // std::cout << "  AFTER " << var->vid->name_hint << std::endl;
      }
    }
    return live_vars;
  }

  Expr VisitExpr_(const LetNode* op) final {
    std::vector<std::pair<Var, Expr>> flattened;
    auto body = FlattenLets(GetRef<Expr>(op), &flattened);

    // auto live_vars_after = EscapeAnalysis(flattened, body);

    size_t start = -1;
    bool in_group = false;
    int op_calls_in_group = 0;
    int last_call_id = -1;
    bool in_static_group = false;
    int static_group_depth = std::numeric_limits<int>::max();
    // end --> begin index mapping for groups
    std::unordered_map<size_t, size_t> groups;
    std::unordered_map<size_t, int> group_static_depths;

    auto start_or_continue_group = [&](size_t i, bool op_call = false, bool static_call = false) {
      if (!in_group) {
        start = i;
        in_group = true;
        in_static_group = static_call;
        if (print_) {
          std::cout << "\n\n[CG] GROUPSTART" << std::endl;
        }
      }
      if (op_call) {
        last_call_id = i;
        op_calls_in_group++;
        // std::cout << "[CG]   Incrementing " << op_calls_in_group << std::endl;
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
          if (in_static_group) {
            group_static_depths[last_call_id] = static_group_depth;
          }
        }
        if (print_) {
          std::cout << "[CG] GROUPEND " << (op_calls_in_group > 0) << std::endl;
        }
        start = -1;
        op_calls_in_group = 0;
        in_group = false;
        static_group_depth = std::numeric_limits<int>::max();
        in_static_group = false;
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
        if (IsStorageType(cleaned_value->checked_type_)) {
          end_group(i);
        } else {
          start_or_continue_group(i);
        }
      } else if (cleaned_value.as<GlobalVarNode>()) {
        end_group(i);
      } else if (cleaned_value.as<FunctionNode>()) {
        end_group(i);
      } else if (auto vn = cleaned_value.as<CallNode>()) {
        auto marked_scalar = IsMarkedScalarOp(vn);
        // std::cout << "[CG] Call  " << vn->op << " " << marked_scalar << std::endl;
        auto static_depth = GetStaticGraphDepth(vn);
        if (vn->op == GetInvokeTVMOp() && static_depth >= 0 && !marked_scalar) {
          if (in_group && !in_static_group) {
            end_group(i);
          }
          start_or_continue_group(i, true, true);
          if (in_group && in_static_group) {
            static_group_depth = std::min(static_group_depth, static_depth);
          }
        } else if (vn->op == GetInvokeTVMOp() && !marked_scalar) {
          if (in_static_group) {
            end_group(i);
          }
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
        for (int j = start; j <= i; ++j) {
          auto value = RemoveOnDeviceCalls(flattened[j].second);
          bindings.push_back(std::make_pair(flattened[j].first, value));
        }

        std::vector<Var> flattened_free_vars;
        std::vector<Expr> call_args_unsorted;
        Map<relay::Var, Array<Expr>> tuple_var_values;

        // std::cout << "[CG] New Group" << std::endl;
        // for (auto pair : bindings) {
        //   std::cout << "[GROUP] " << pair.first->vid->name_hint << std::endl;
        // }

        auto free_vars_set = GetFreeVarsInGroup(bindings);
        for (auto var : free_vars_set) {
          FlattenVar(var, &tuple_var_values, &flattened_free_vars, &call_args_unsorted);
        }

        //////////////////////////////////////////////////
        auto group_res = GroupStaticScheduler(mod_, prim_funcs_access_modes_, tuple_var_values,
                                              scattered_kernels_)
                             .Schedule(flattened_free_vars, bindings);
        auto call_groups = group_res.groups;
        auto groups_increment_depth = group_res.increment_depth;
        auto var_to_var_mapping = group_res.var_to_var_mapping;
        //////////////////////////////////////////////////

        for (int j = static_cast<int>(bindings.size()) - 1; j >= 0; --j) {
          auto& p = bindings[j];
          if (p.second.as<TupleNode>() || p.second.as<VarNode>() ||
              p.second.as<TupleGetItemNode>()) {
            body = Let(p.first, p.second, body);
          }
        }

        for (int k = static_cast<int>(call_groups.size()) - 1; k >= 0; --k) {
          auto group = call_groups[k];

          // for (auto call : group) {
          //   std::cout << PrettyPrint(call) << std::endl;
          // }
          std::string name = "prim_func" + std::to_string(prim_func_ctr++);
          // std::cout << "Creating prim function " << name << " " << groups_increment_depth[k]
          // << std::endl;
          auto group_free_vars_set = GetFreeVarsInGroup(group);
          std::vector<Var> group_flattened_free_vars;
          std::vector<Expr> group_call_args_unsorted;
          for (size_t i = 0; i < flattened_free_vars.size(); ++i) {
            if (group_free_vars_set.count(flattened_free_vars[i])) {
              group_flattened_free_vars.push_back(flattened_free_vars[i]);
              group_call_args_unsorted.push_back(call_args_unsorted[i]);
            }
          }

          auto res = TIRLowererUnbatched(mod_, prim_funcs_access_modes_, tuple_var_values,
                                         var_to_var_mapping, name, scattered_kernels_)
                         .LowerToTIR(group_flattened_free_vars, group);

          std::vector<int> param_positions = res.param_positions;
          std::vector<Expr> call_args;

          for (size_t j = 0; j < param_positions.size(); ++j) {
            auto pos = param_positions[j];
            if (pos >= 0) {
              call_args.push_back(group_call_args_unsorted[pos]);
            }
          }

          auto prim_func = res.func;
          GlobalVar prim_func_var(name, prim_func->checked_type_);
          auto input_args_tuple = Tuple(call_args);
          auto output_args_tuple = Tuple(Array<Expr>());
          Map<String, ObjectRef> attrs_map = {
              {attr::kPrimitive, tvm::Integer(1)},
              {tir::attr::kDBCoarseWrapperPrimFunc, Integer(1)},
              {tir::attr::kDBIncrementDepth, Bool(groups_increment_depth[k])}};
          auto it = group_static_depths.find(i);
          if (it != group_static_depths.end()) {
            attrs_map.Set(tir::attr::kDBGraphDepth, Integer(it->second));
          }
          bool has_static_output = false;
          // std::cout << "[TENET] Group " << std::endl;
          for (auto e : group) {
            // std::cout << "[TENET]  Call " << e << std::endl;
            auto cn = e.as<CallNode>();
            has_static_output =
                has_static_output || Downcast<DictAttrs>(cn->attrs)
                                         .GetAttr(tir::attr::kDBScalarOutputOp, Bool(false))
                                         .value()
                                         ->value;
            // std::cout << "[TENET]    " << has_static_output << std::endl;
          }
          if (has_static_output) {
            attrs_map.Set(tir::attr::kDBScalarOutputOp, Bool(true));
          }
          auto call =
              InvokeTVMOp(prim_func_var, input_args_tuple, output_args_tuple, DictAttrs(attrs_map));
          prim_func = AddAttrsToWrapperFunc(prim_func, name, false);
          prim_funcs_.push_back(std::make_pair(prim_func_var, prim_func));

          if (batched_execution_) {
            std::vector<Var> group_flattened_free_vars_sorted;
            for (size_t j = 0; j < param_positions.size(); ++j) {
              auto pos = param_positions[j];
              if (pos >= 0) {
                group_flattened_free_vars_sorted.push_back(group_flattened_free_vars[pos]);
              }
            }

            std::string batched_name = runtime::vm::GetBatchedName(name);
            // std::cout << "Creating batched prim function " << batched_name << std::endl;
            auto batched_res =
                TIRLowererBatched(mod_, prim_funcs_access_modes_, tuple_var_values,
                                  var_to_var_mapping, batched_name, scattered_kernels_)
                    .LowerToTIR(group_flattened_free_vars_sorted, group);
            auto batched_func = batched_res.func;
            GlobalVar batched_func_var(batched_name, prim_func->checked_type_);
            batched_func = AddAttrsToWrapperFunc(batched_func, batched_name, true);
            prim_funcs_.push_back(std::make_pair(batched_func_var, batched_func));
            batched_func_pairs_.push_back(std::make_pair(prim_func_var, batched_func_var));
            batched_arg_modes_.push_back(std::make_pair(batched_func_var, batched_res.arg_modes));

            for (auto kv : batched_res.contiguous_functions) {
              ICHECK(!contiguous_functions_.count(kv.first))
                  << "Function has been made contiguous before";
              contiguous_functions_.Set(kv.first, kv.second);
            }
          }

          body = Let(Var("dummy", VoidType()), call, body);
        }

        i = start - 1;
      }
    }

    return body;
  }

  bool print_;
  const IRModule& mod_;
  const FunctionsAccessModesMap& prim_funcs_access_modes_;

  const bool batched_execution_;
  const bool scattered_kernels_;
  static int prim_func_ctr;

  PointsToMap points_to_map_;

 public:
  std::vector<std::pair<GlobalVar, tir::PrimFunc>> prim_funcs_;
  std::vector<std::pair<GlobalVar, GlobalVar>> batched_func_pairs_;
  std::vector<std::pair<GlobalVar, Array<Integer>>> batched_arg_modes_;
  Map<GlobalVar, tir::PrimFunc> contiguous_functions_;
};
int Coarsener::prim_func_ctr = 0;

IRModule CoarsenGranularity(IRModule& mod, bool batched_execution, bool scattered_kernels) {
  // std::cout << "==============================================================" << std::endl;
  // std::cout << "[CG] Coarsening now!" << std::endl;
  tvm::Map<GlobalVar, Function> updates;
  tvm::Map<GlobalVar, tir::PrimFunc> new_prim_funcs;

  auto funcs = mod->functions;

  FunctionsAccessModesMap prim_funcs_access_modes;
  for (const auto& it : funcs) {
    if (it.second.as<tir::PrimFuncNode>()) {
      auto func = Downcast<tir::PrimFunc>(it.second);
      auto access_modes_map = ComputeAndVerifyAccessModes(func).second;
      prim_funcs_access_modes[it.first] = access_modes_map;
    }
  }

  for (const auto& it : funcs) {
    if (const auto* n = it.second.as<FunctionNode>()) {
      ICHECK_EQ(FreeVars(it.second).size(), 0);
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
      Function func = GetRef<Function>(n);

      bool print = false;  // (support::StartsWith(it.first->name_hint, "lifted_name"));
      // std::cout << "[CG] Func " << it.first->name_hint << std::endl;
      Coarsener coarsener(mod, prim_funcs_access_modes, batched_execution, scattered_kernels);
      Function ret = coarsener.Coarsen(func, print);

      updates.Set(it.first, ret);

      for (auto it : coarsener.prim_funcs_) {
        new_prim_funcs.Set(it.first, it.second);
      }

      for (auto it : coarsener.contiguous_functions_) {
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
    }
  }

  for (auto pair : updates) {
    mod->Add(pair.first, pair.second, true);
  }

  for (auto pair : new_prim_funcs) {
    mod->Add(pair.first, pair.second, true);
  }
  new_prim_funcs.clear();

  Map<GlobalVar, Array<Integer>> updated_batched_arg_modes;

  // Remove unused args in prim funcs
  for (const auto& it : funcs) {
    if (it.second.as<tir::PrimFuncNode>() &&
        !it.second->HasNonzeroAttr(tir::attr::kDBCoarseWrapperPrimFunc)) {
      bool print = (it.first->name_hint == "vm_mod_fused_nn_contrib_dense_pack_add_1");
      auto func = Downcast<tir::PrimFunc>(it.second);
      auto access_modes_map = prim_funcs_access_modes.at(it.first);
      Array<tir::Var> used_params;
      size_t scalar_args_start = 0;
      if (func->HasNonzeroAttr(tir::attr::kDBBatchedPrimFunc)) {
        scalar_args_start = 1;
        used_params.push_back(func->params[0]);
      }
      for (size_t i = scalar_args_start; i < func->params.size(); ++i) {
        auto& var = func->params[i];
        if (access_modes_map.count(var)) {
          used_params.push_back(var);
        }
      }
      if (batched_execution && !it.second->HasNonzeroAttr(tir::attr::kDBBatchedPrimFunc)) {
        auto func_gv = it.first;
        ICHECK(mod->batched_prim_funcs.count(func_gv)) << func_gv;
        auto batched_func_gv = mod->batched_prim_funcs.at(func_gv);
        auto batched_arg_modes = mod->batched_arg_modes.at(batched_func_gv);
        Array<Integer> updated_arg_modes;
        for (size_t i = scalar_args_start; i < func->params.size(); ++i) {
          auto& var = func->params[i];
          if (access_modes_map.count(var)) {
            updated_arg_modes.push_back(batched_arg_modes[i]);
          }
        }
        updated_batched_arg_modes.Set(batched_func_gv, updated_arg_modes);
      }
      if (used_params.size() != func->params.size()) {
        new_prim_funcs.Set(it.first,
                           tir::PrimFunc(used_params, func->body, func->ret_type, func->buffer_map,
                                         func->scatter_buffer_map, func->attrs, func->span));
      }
    }
  }

  for (auto kv : updated_batched_arg_modes) {
    mod->batched_arg_modes.Set(kv.first, kv.second);
  }

  for (auto pair : new_prim_funcs) {
    mod->Add(pair.first, pair.second, true);
  }

  return mod;
}

IRModule ComputeAccessModes(IRModule& mod) {
  tvm::Map<GlobalVar, tir::PrimFunc> new_prim_funcs;
  auto funcs = mod->functions;
  FunctionsAccessModesMap prim_funcs_access_modes;
  // std::cout << "[CG] AccessModeMaps" << std::endl;
  for (const auto& it : funcs) {
    if (it.second.as<tir::PrimFuncNode>()) {
      auto func = Downcast<tir::PrimFunc>(it.second);
      Array<Integer> access_modes = ComputeAndVerifyAccessModes(func).first;
      func = WithAttr(func, tir::attr::kDBArgAccessModes, access_modes);
      // std::cout << "[AccessMode] " << it.first << " " << access_modes << std::endl;
      new_prim_funcs.Set(it.first, func);
    }
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

TVM_REGISTER_GLOBAL("relay._transform.CoarsenPrimitiveFuncGranularity")
    .set_body_typed(CoarsenPrimitiveFuncGranularity);

Pass ComputePrimFuncAccessModes() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return ComputeAccessModes(m); };
  return CreateModulePass(pass_func, 0, "ComputePrimFuncAccessModes", {});
}

TVM_REGISTER_GLOBAL("relay._transform.ComputePrimFuncAccessModes")
    .set_body_typed(ComputePrimFuncAccessModes);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
