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

#include "./function_pointer_analysis.h"

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/vm/dynamic_batching.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include <limits>

#include "../../printer/text_printer.h"
#include "../analysis/call_graph.h"
#include "../op/annotation/annotation.h"
#include "../op/memory/memory.h"
#include "../op/vm/vm.h"
#include "./map_set.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {

std::vector<Var> CollectPatternVars(const Pattern& p) {
  class PatternVarCollector : public ExprVisitor {
   public:
    void VisitPattern(const Pattern& p) override {
      if (auto pvn = p.as<PatternVarNode>()) {
        vars_.push_back(pvn->var);
      } else if (auto pcn = p.as<PatternConstructorNode>()) {
        for (auto pattern : pcn->patterns) {
          this->VisitPattern(pattern);
        }
      } else if (auto ptn = p.as<PatternTupleNode>()) {
        for (auto pattern : ptn->patterns) {
          this->VisitPattern(pattern);
        }
      }
    }

    std::vector<Var> vars_;
  };

  PatternVarCollector collector;
  collector.VisitPattern(p);
  return collector.vars_;
}

namespace {

using BaseExprFunctor = ExprFunctor<FunctionSet(const Expr& n, FPAContextT)>;

using FPAContextT = const Object*;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

using FunctionEnvironmentMap = std::unordered_map<const FunctionNode*, Map<Var, FunctionSet>>;
using OnStackSet =
    std::unordered_set<std::pair<FPAContextT, const FunctionNode*>, PairHash, PairEquals>;

class FunctionPointerAnalysis : public BaseExprFunctor {
 public:
  FunctionPointerAnalysis(const IRModule& mod) : mod_(mod) {}

  void PerformAnalysis() {
    RunAnalysis();

    for (auto kv : callees_map_) {
      auto context = kv.first.first;
      auto callsite = kv.first.second;
      auto callees = kv.second;
      callees_per_callsite_[callsite][context].insert(callees.begin(), callees.end());
    }
  }

  PreciseCallGraph GetPreciseCallGraph() {
    PreciseCallGraph call_graph;
    for (auto kv : mod_->functions) {
      if (auto fn = kv.second.as<FunctionNode>()) {
        PostOrderVisit(fn->body, [&](const Expr& expr) {
          if (auto callsite = expr.as<CallNode>()) {
            auto it = callees_per_callsite_.find(callsite);
            if (it != callees_per_callsite_.end()) {
              auto callees_per_context = it->second;
              for (auto kvkv : callees_per_context) {
                call_graph[fn].insert(kvkv.second.begin(), kvkv.second.end());
              }
            }
          }
        });
      }
    }
    return call_graph;
  }

  FunctionSet GetRecursiveFunctions() {
    auto call_graph = GetPreciseCallGraph();

    FunctionSet recursive_functions;
    for (auto kv : mod_->functions) {
      if (auto fn = kv.second.as<FunctionNode>()) {
        OrderedFunctionSet visited;
        if (Reachable(call_graph, fn, fn, visited, true)) {
          MapSet::Insert(recursive_functions, GetRef<Function>(fn));
          // std::cout << "[ADF] Recursive func " << fn << " " << func_name_map_[fn] << std::endl;
        }
      }
    }
    return recursive_functions;
  }

  CalleesMap GetCalleesMap() { return callees_map_; }

  FPAVarStateMap GetVarPointsToMap() { return var_states_; }

 private:
  void RunAnalysis() {
    auto main_func = Downcast<Function>(mod_->Lookup("main"));

    ICHECK_EQ(main_func->type_params.size(), 0);
    for (auto param : main_func->params) {
      ICHECK(!param->checked_type().as<FuncTypeNode>());
      Add(param, GetInitialContext(), FunctionSet(), "Main param");
    }

    for (int i = 0; i < max_iterations_; ++i) {
      // std::cout << "[FPA] ITERATION " << i << std::endl;
      this->Reset();
      type_environment_stack_.push_back(Map<TypeVar, Type>());
      environment_stack_.push_back(Map<Var, FunctionSet>());
      auto context_fn_key = std::make_pair(GetInitialContext(), main_func.get());
      on_stack_.insert(context_fn_key);
      function_stack_.push_back(main_func);
      auto full_taint = this->VisitExpr(main_func->body, GetInitialContext());
      function_stack_.pop_back();
      on_stack_.erase(context_fn_key);
      environment_stack_.pop_back();
      type_environment_stack_.pop_back();
    }
  }

  std::string GetFunctionName(const Function& fn) {
    return fn->GetAttr<String>("db.function_name", String("")).value();
  }

  void Reset() {
    on_stack_.clear();
    environment_stack_.clear();
    type_environment_stack_.clear();
  }

  FunctionSet MergeFunctionSets(const FunctionSet& set1, const FunctionSet& set2) {
    return MapSet::Merge<Function>(set1, set2);
  }

  FunctionSet Merge(const Array<FunctionSet>& full_taints) {
    FunctionSet function_points_to;
    for (auto& full_taint : full_taints) {
      function_points_to = MergeFunctionSets(function_points_to, full_taint);
    }
    return function_points_to;
  }

  template <typename T, typename MapType>
  FunctionSet Add(MapType& map, const T& obj, FPAContextT context, const FunctionSet& to_add) {
    auto key = std::make_pair(context, obj.get());
    auto it = map.find(key);
    if (it == map.end()) {
      map[key] = to_add;
      return to_add;
    } else {
      auto merged = Merge(Array<FunctionSet>({it->second, to_add}));
      map[key] = merged;
      return merged;
    }
  }

  FunctionSet Add(const Var& var, FPAContextT current_context, const FunctionSet& to_add,
                  const std::string& reason) {
    return Add<Var, FPAVarStateMap>(var_states_, var, current_context, to_add);
  }

  FunctionSet Add(const Function& fn, FPAContextT current_context, const FunctionSet& to_add) {
    return Add<Function, FPAFunctionStateMap>(function_states_, fn, current_context, to_add);
  }

  bool Reachable(const PreciseCallGraph& call_graph, const FunctionNode* src,
                 const FunctionNode* dst, OrderedFunctionSet& visited, bool start = false) {
    if (!start && src == dst) {
      return true;
    }
    if (visited.count(src)) {
      return false;
    }
    visited.insert(src);
    auto it = call_graph.find(src);
    if (it != call_graph.end()) {
      auto callees = it->second;
      for (auto callee : callees) {
        if (Reachable(call_graph, callee, dst, visited)) {
          return true;
        }
      }
    }
    return false;
  }

  FunctionSet GetOrCreateFunctionState(const Function& fn, FPAContextT context) {
    auto key = std::make_pair(context, fn.get());
    auto it = function_states_.find(key);
    if (it == function_states_.end()) {
      return FunctionSet();
    } else {
      return it->second;
    }
  }

  FPAContextT GetInitialContext() { return nullptr; }

  FunctionSet VisitExpr_(const ConstantNode* op, FPAContextT current_context) {
    return FunctionSet();
  }

  FunctionSet VisitExpr_(const TupleNode* op, FPAContextT current_context) {
    FunctionSet function_points_to;
    for (auto field : op->fields) {
      auto field_full_taint = this->VisitExpr(field, current_context);
      function_points_to = MergeFunctionSets(function_points_to, field_full_taint);
    }
    return function_points_to;
  }

  FunctionSet VisitExpr_(const VarNode* op, FPAContextT current_context) {
    bool print = false;  // op->checked_type().as<TypeCallNode>();

    if (print) {
      std::cout << "[MPT] Getting var " << current_context << " " << op->vid->name_hint
                << std::endl;
    }

    auto var = GetRef<Var>(op);
    auto current_environment = environment_stack_.back();
    auto it = current_environment.find(var);
    if (it != current_environment.end()) {
      auto full_taint = (*it).second;
      if (print) {
        std::cout << "[MPT]  In env " << MapSet::ToString(full_taint) << std::endl;
      }
      return full_taint;
    }

    auto key = std::make_pair(current_context, op);
    auto iit = var_states_.find(key);
    if (iit == var_states_.end()) {
      auto taint = FunctionSet();
      var_states_[key] = taint;
      if (print) {
        std::cout << "[MPT]  Not found" << std::endl;
      }
      return taint;
    } else {
      if (print) {
        std::cout << "[MPT]  Found " << MapSet::ToString(iit->second) << std::endl;
      }
      return iit->second;
    }
  }

  FunctionSet VisitExpr_(const GlobalVarNode* op, FPAContextT current_context) {
    auto function_points_to = FunctionSet();
    auto base_func = mod_->Lookup(GetRef<GlobalVar>(op));
    if (base_func.as<FunctionNode>()) {
      MapSet::Insert(function_points_to, Downcast<Function>(base_func));
    }
    return function_points_to;
  }

  VarSet GetFreeVars(const Function& expr) {
    auto it = free_vars_map_.find(expr.get());
    if (it == free_vars_map_.end()) {
      auto free_vars = FreeVarsDedup(expr);
      free_vars_map_[expr.get()] = free_vars;
      return free_vars;
    } else {
      return it->second;
    }
  }

  FunctionSet VisitExpr_(const FunctionNode* op, FPAContextT current_context) {
    auto function = GetRef<Function>(op);
    auto free_vars = GetFreeVars(function);
    for (auto free_var : free_vars) {
      auto taint = this->VisitExpr(free_var, current_context);
      auto it = function_environments_[op].find(free_var);
      if (it != function_environments_[op].end()) {
        auto preexisting_taint = (*it).second;
        taint = Merge(Array<FunctionSet>({taint, preexisting_taint}));
      }
      function_environments_[op].Set(free_var, taint);
    }

    FunctionSet function_points_to;
    MapSet::Insert(function_points_to, function);
    return function_points_to;
  }

  FunctionSet VisitBody(const Function& fn, FPAContextT context, const CallNode* call) {
    // std::cout << "[MPT] Visiting body " << GetFunctionName(fn) << std::endl;
    Map<TypeVar, Type> type_environment;
    ICHECK_EQ(fn->type_params.size(), call->type_args.size());
    for (size_t i = 0; i < call->type_args.size(); ++i) {
      type_environment.Set(fn->type_params[i], call->type_args[i]);
    }

    type_environment_stack_.push_back(type_environment);
    environment_stack_.push_back(function_environments_[fn.get()]);
    auto context_fn_key = std::make_pair(context, fn.get());
    on_stack_.insert(context_fn_key);
    function_stack_.push_back(fn);
    auto full_taint = this->VisitExpr(fn->body, context);
    function_stack_.pop_back();
    on_stack_.erase(context_fn_key);
    environment_stack_.pop_back();
    type_environment_stack_.pop_back();
    return full_taint;
  }

  FunctionSet VisitExpr_(const CallNode* op, FPAContextT current_context) {
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      return this->VisitExpr(on_device_props.body, current_context);
    }

    if (op->op.as<OpNode>() || op->op.as<ConstructorNode>()) {
      for (auto arg : op->args) {
        this->VisitExpr(arg, current_context);
      }
      return FunctionSet();
    }

    auto callee_context = op;

    auto callees = this->VisitExpr(op->op, current_context);
    auto key = std::make_pair(current_context, op);

    Array<FunctionSet> callee_taints;
    // std::cout << "[MPT] Call " << op->op << " " << callees.size() << std::endl;
    for (auto kv : callees) {
      auto callee_fn = kv.first;
      callees_map_[key].insert(callee_fn.get());
      // std::cout << "[MPT]  Callee " << GetFunctionName(callee_fn) << std::endl;

      //////////////////////////////
      auto callee_name = callee_fn->GetAttr<String>("db.function_name");
      bool print = false;  //(callee_name == "map" || callee_name == "get_classification_fn_f0");
      if (print) {
        std::cout << "[MPT] Calling " << callee_name << " at " << GetRef<Expr>(op) << std::endl;
      }
      //////////////////////////////

      ICHECK_EQ(callee_fn->params.size(), op->args.size());
      for (size_t i = 0; i < callee_fn->params.size(); ++i) {
        auto arg = op->args[i];
        auto param = callee_fn->params[i];
        // if (op->args.size() == 10) {
        // this->VisitExpr(op->args[9], current_context);
        // }
        auto param_full_taint = this->VisitExpr(arg, current_context);
        auto add_res = Add(param, callee_context, param_full_taint, "function param");

        //////////////////////////////
        if (print) {
          std::string arg_str = "";
          if (auto vn = arg.as<VarNode>()) {
            arg_str = vn->vid->name_hint;
          }
          std::cout << "[MPT]  Param " << current_context << " " << arg_str << " "
                    << param->vid->name_hint << " " << param_full_taint.size() << " "
                    << add_res.size() << std::endl;
        }
        //////////////////////////////
      }

      FunctionSet callee_full_taint;
      if (on_stack_.count(std::make_pair(callee_context, callee_fn.get()))) {
        callee_full_taint = GetOrCreateFunctionState(callee_fn, callee_context);
        if (print) {
          std::cout << "[MPT] Return value 1 " << callee_name << " "
                    << MapSet::ToString(callee_full_taint) << std::endl;
        }
      } else {
        callee_full_taint = VisitBody(callee_fn, callee_context, op);
        callee_full_taint = Add(callee_fn, callee_context, callee_full_taint);
        if (print) {
          std::cout << "[MPT] Return value 2 " << callee_name << " "
                    << MapSet::ToString(callee_full_taint) << std::endl;
        }
      }
      callee_taints.push_back(callee_full_taint);
    }

    return Merge(callee_taints);
  }

  FunctionSet VisitExpr_(const LetNode* op, FPAContextT current_context) {
    auto value_res = this->VisitExpr(op->value, current_context);
    auto added_res = Add(op->var, current_context, value_res, "Let value");
    return this->VisitExpr(op->body, current_context);
  }

  FunctionSet VisitExpr_(const IfNode* op, FPAContextT current_context) {
    auto cond_taint = this->VisitExpr(op->cond, current_context);
    auto true_taint = this->VisitExpr(op->true_branch, current_context);
    auto false_taint = this->VisitExpr(op->false_branch, current_context);
    return Merge(Array<FunctionSet>({true_taint, false_taint}));
  }

  FunctionSet VisitExpr_(const OpNode* op, FPAContextT current_context) { return FunctionSet(); }

  FunctionSet VisitExpr_(const TupleGetItemNode* op, FPAContextT current_context) {
    return this->VisitExpr(op->tuple, current_context);
  }

  FunctionSet VisitExpr_(const RefCreateNode* op, FPAContextT current_context) {
    return UnsupportedFunction(op, current_context);
  }

  FunctionSet VisitExpr_(const RefReadNode* op, FPAContextT current_context) {
    return UnsupportedFunction(op, current_context);
  }

  FunctionSet VisitExpr_(const RefWriteNode* op, FPAContextT current_context) {
    return UnsupportedFunction(op, current_context);
  }

  FunctionSet UnsupportedFunction(const ExprNode* op, FPAContextT current_context) {
    ICHECK(false) << "Do not support " << op->GetTypeKey();
    return FunctionSet();
  }

  FunctionSet VisitExpr_(const ConstructorNode* op, FPAContextT current_context) {
    return FunctionSet();
  }

  FunctionSet VisitExpr_(const MatchNode* op, FPAContextT current_context) {
    auto data_taint = this->VisitExpr(op->data, current_context);

    Array<FunctionSet> clause_taints;
    for (auto clause : op->clauses) {
      auto pattern_vars = CollectPatternVars(clause->lhs);
      for (auto& var : pattern_vars) {
        Add(var, current_context, data_taint, "Match pattern");
      }
      // std::cout << "[MPT]   Visiting clause " << clause->lhs << std::endl;
      clause_taints.push_back(this->VisitExpr(clause->rhs, current_context));
    }

    return Merge(clause_taints);
  }

  const IRModule& mod_;

  FPAVarStateMap var_states_;
  FPAFunctionStateMap function_states_;
  FunctionEnvironmentMap function_environments_;
  std::unordered_map<const FunctionNode*, VarSet> free_vars_map_;
  OnStackSet on_stack_;
  std::vector<Map<Var, FunctionSet>> environment_stack_;
  std::vector<Map<TypeVar, Type>> type_environment_stack_;

  std::vector<Function> function_stack_;
  CalleesMap callees_map_;
  std::unordered_map<const CallNode*, std::unordered_map<FPAContextT, OrderedFunctionSet>>
      callees_per_callsite_;
  int max_iterations_{10};
};

}  // namespace

std::pair<FunctionSet, CalleesMap> GetRecursiveFunctions(const IRModule& mod) {
  FunctionPointerAnalysis analysis(mod);
  analysis.PerformAnalysis();
  auto recursive_functions = analysis.GetRecursiveFunctions();
  auto callees_map = analysis.GetCalleesMap();
  return std::make_pair(recursive_functions, callees_map);
}

PreciseCallGraph GetPreciseCallGraph(const IRModule& mod) {
  FunctionPointerAnalysis analysis(mod);
  analysis.PerformAnalysis();
  return analysis.GetPreciseCallGraph();
}

CalleesMap GetCalleesMap(const IRModule& mod) {
  FunctionPointerAnalysis analysis(mod);
  analysis.PerformAnalysis();
  return analysis.GetCalleesMap();
}

FPAVarStateMap GetVarPointsToMap(const IRModule& mod) {
  FunctionPointerAnalysis analysis(mod);
  analysis.PerformAnalysis();
  return analysis.GetVarPointsToMap();
}

}  // namespace relay
}  // namespace tvm
