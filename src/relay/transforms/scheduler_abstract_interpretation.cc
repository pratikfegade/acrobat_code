
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

#include <limits>

#include "../../printer/text_printer.h"
#include "../../support/arena.h"
#include "../analysis/call_graph.h"
#include "../op/annotation/annotation.h"
#include "../op/memory/memory.h"
#include "../op/vm/vm.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {

class FunctionWithTemplateTypes;
/*! \brief FunctionWithTemplateTypes container. */
class FunctionWithTemplateTypesNode : public ExprNode {
 public:
  Function function;
  tvm::Array<Type> type_args;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("function", &function);
    v->Visit("type_args", &type_args);
  }

  bool SEqualReduce(const FunctionWithTemplateTypesNode* other, SEqualReducer equal) const {
    return equal(function, other->function) && equal(type_args, other->type_args);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(function);
    hash_reduce(type_args);
  }

  static constexpr const char* _type_key = "relay.FunctionWithTemplateTypes";
  TVM_DECLARE_FINAL_OBJECT_INFO(FunctionWithTemplateTypesNode, ExprNode);
};

class FunctionWithTemplateTypes : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param op The operator will be invoked.
   * \param args The arguments of the call.
   * \param attrs The attributes of the call node.
   * \param type_args The type arguments passed to a polymorphic function.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit FunctionWithTemplateTypes(Function function, tvm::Array<Type> type_args) {
    ObjectPtr<FunctionWithTemplateTypesNode> n = make_object<FunctionWithTemplateTypesNode>();
    n->function = std::move(function);
    n->type_args = std::move(type_args);
    data_ = std::move(n);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(FunctionWithTemplateTypes, Expr, FunctionWithTemplateTypesNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(FunctionWithTemplateTypesNode);
};

TVM_REGISTER_NODE_TYPE(FunctionWithTemplateTypesNode);

// using FunctionSet = std::unordered_set<const FunctionNode*>;
using TupleDepth = std::vector<int>;
using SAIBaseExprFunctor = ExprFunctor<TupleDepth(const Expr& n)>;
using SAIVarStateMap = std::unordered_map<Var, TupleDepth, ObjectPtrHash, ObjectPtrEqual>;
using SAIInvokeTVMOpDepthMap = std::unordered_map<const CallNode*, int>;
using SAIFunctionStateMap =
    std::unordered_map<FunctionWithTemplateTypes, TupleDepth, StructuralHash, StructuralEqual>;
using FunctionSet = std::set<const FunctionNode*>;
using FPAVarStateMap = std::unordered_map<Var, FunctionSet, ObjectPtrHash, ObjectPtrEqual>;
using FPAFunctionStateMap =
    std::unordered_map<Function, FunctionSet, ObjectPtrHash, ObjectPtrEqual>;
using FPABaseExprFunctor = ExprFunctor<FunctionSet(const Expr& n)>;
using PreciseCallGraph = std::unordered_map<const FunctionNode*, FunctionSet>;
using CalleesMap = std::unordered_map<const CallNode*, FunctionSet>;

class FunctionPointerAnalysis : public FPABaseExprFunctor {
 public:
  FunctionPointerAnalysis(const IRModule& mod) : mod_(mod) {
    for (auto kv : mod_->functions) {
      func_name_map_[kv.second.get()] = kv.first->name_hint;
    }
  }

  void PerformAnalysis() {
    auto call_graph = CallGraph(mod_);
    size_t i = 0;
    for (; i < max_iterations_; ++i) {
      visited_functions_.clear();
      changed_ = false;
      for (auto cge : call_graph->TopologicalOrder()) {
        auto func = mod_->Lookup(cge->GetGlobalVar());
        if (auto fn = func.as<FunctionNode>()) {
          VisitExpr(fn->body);
        }
      }
      if (!changed_) {
        break;
      }
    }

    std::cout << "[GFP] Iterations: " << i << std::endl;
    for (auto kv : var_states_) {
      if (kv.second.size() > 0) {
        std::cout << "[GFP]  Function pointers: " << kv.first->vid->name_hint << std::endl;
        for (auto& f : kv.second) {
          std::cout << "[GFP]   " << func_name_map_[f] << std::endl;
        }
      }
    }
  }

  FunctionSet GetRecursiveFunctions() {
    PreciseCallGraph call_graph;
    for (auto kv : mod_->functions) {
      FunctionSet callees;
      if (auto fn = kv.second.as<FunctionNode>()) {
        PostOrderVisit(fn->body, [&](const Expr& expr) {
          if (auto cn = expr.as<CallNode>()) {
            auto this_callees = this->VisitExpr(cn->op);
            callees.insert(this_callees.begin(), this_callees.end());
          }
        });
        call_graph[fn] = callees;
      }
    }

    FunctionSet recursive_functions;
    for (auto kv : mod_->functions) {
      if (auto fn = kv.second.as<FunctionNode>()) {
        FunctionSet visited;
        if (Reachable(call_graph, fn, fn, visited, true)) {
          recursive_functions.insert(fn);
          std::cout << "[ADF] Recursive func " << fn << " " << func_name_map_[fn] << std::endl;
        }
      }
    }
    return recursive_functions;
  }

  CalleesMap GetCalleesMap() {
    CalleesMap callees_map;
    for (auto kv : mod_->functions) {
      if (auto fn = kv.second.as<FunctionNode>()) {
        PostOrderVisit(fn->body, [&](const Expr& expr) {
          if (auto cn = expr.as<CallNode>()) {
            callees_map[cn] = this->VisitExpr(cn->op);
          }
        });
      }
    }
    return callees_map;
  }

 private:
  template <typename T>
  void Add(std::unordered_map<T, FunctionSet, ObjectPtrHash, ObjectPtrEqual>& map, const T& key,
           const FunctionSet& to_add) {
    auto old_size = map[key].size();
    map[key].insert(to_add.begin(), to_add.end());
    auto new_size = map[key].size();
    if (old_size != new_size) {
      changed_ = true;
    }
  }

  void Add(const Var& var, const FunctionSet& to_add) { Add<Var>(var_states_, var, to_add); }

  void Add(const Function& var, const FunctionSet& to_add) {
    Add<Function>(function_states_, var, to_add);
  }

  bool Reachable(const PreciseCallGraph& call_graph, const FunctionNode* src,
                 const FunctionNode* dst, FunctionSet& visited, bool start = false) {
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

  FunctionSet VisitExpr_(const ConstantNode* op) override {
    FunctionSet ret;
    return ret;
  }

  FunctionSet VisitExpr_(const TupleNode* op) override {
    FunctionSet ret;
    for (auto field : op->fields) {
      auto field_ret = VisitExpr(field);
      ret.insert(field_ret.begin(), field_ret.end());
    }
    return ret;
  }

  FunctionSet VisitExpr_(const VarNode* op) override {
    auto var = GetRef<Var>(op);
    auto it = var_states_.find(var);
    if (it == var_states_.end()) {
      var_states_[var] = {};
      changed_ = true;
    }
    return var_states_[var];
  }

  FunctionSet VisitExpr_(const GlobalVarNode* op) override {
    return {mod_->Lookup(GetRef<GlobalVar>(op)).as<FunctionNode>()};
  }

  FunctionSet VisitExpr_(const FunctionNode* op) override { return {op}; }

  FunctionSet VisitExpr_(const CallNode* op) override {
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      if (on_device_props.body.as<CallNode>()) {
        op = on_device_props.body.as<CallNode>();
      }
    }
    auto callee_set = VisitExpr(op->op);
    if (callee_set.empty()) {
      this->VisitExpr(op->op);
      for (auto arg : op->args) {
        this->VisitExpr(arg);
      }
    }
    FunctionSet ret;
    for (auto callee : callee_set) {
      auto callee_fn = GetRef<Function>(callee);
      if (visited_functions_.count(callee)) {
        auto it = function_states_.find(callee_fn);
        if (it != function_states_.end()) {
          ret.insert(it->second.begin(), it->second.end());
        }
      } else {
        for (size_t i = 0; i < op->args.size(); ++i) {
          auto param_state = VisitExpr(op->args[i]);
          Add(callee->params[i], param_state);
        }
        visited_functions_.insert(callee);
        auto res = VisitExpr(callee->body);
        Add(callee_fn, res);
        ret.insert(res.begin(), res.end());
      }
    }
    return ret;
  }

  FunctionSet VisitExpr_(const LetNode* op) override {
    auto value_res = VisitExpr(op->value);
    Add(op->var, value_res);
    return VisitExpr(op->body);
  }

  FunctionSet VisitExpr_(const IfNode* op) override {
    VisitExpr(op->cond);
    FunctionSet ret;
    auto then_ret = VisitExpr(op->true_branch);
    ret.insert(then_ret.begin(), then_ret.end());
    auto else_ret = VisitExpr(op->false_branch);
    ret.insert(else_ret.begin(), else_ret.end());
    return ret;
  }

  FunctionSet VisitExpr_(const OpNode* op) override { return FunctionSet(); }

  FunctionSet VisitExpr_(const TupleGetItemNode* op) override { return VisitExpr(op->tuple); }

  FunctionSet VisitExpr_(const RefCreateNode* op) override {
    ICHECK(false) << "Refs not supported for now";
    FunctionSet ret;
    return ret;
  }

  FunctionSet VisitExpr_(const RefReadNode* op) override {
    ICHECK(false) << "Refs not supported for now";
    FunctionSet ret;
    return ret;
  }

  FunctionSet VisitExpr_(const RefWriteNode* op) override {
    ICHECK(false) << "Refs not supported for now";
    FunctionSet ret;
    return ret;
  }

  FunctionSet VisitExpr_(const ConstructorNode* op) override {
    FunctionSet ret;
    return ret;
  }

  FunctionSet VisitExpr_(const MatchNode* op) override {
    FunctionSet ret;
    for (auto clause : op->clauses) {
      auto clause_ret = VisitExpr(clause->rhs);
      ret.insert(clause_ret.begin(), clause_ret.end());
    }
    return ret;
  }

  const IRModule& mod_;
  FPAVarStateMap var_states_;
  FPAFunctionStateMap function_states_;
  std::unordered_set<const Object*> visited_functions_;
  size_t max_iterations_{50};
  bool changed_{false};
  std::unordered_map<const BaseFuncNode*, std::string> func_name_map_;
};

std::pair<FunctionSet, CalleesMap> GetRecursiveFunctions(const IRModule& mod) {
  FunctionPointerAnalysis analysis(mod);
  analysis.PerformAnalysis();
  auto recursive_functions = analysis.GetRecursiveFunctions();
  auto callees_map = analysis.GetCalleesMap();
  return std::make_pair(recursive_functions, callees_map);
}

constexpr int MAX_DEPTH_VALUE = 1 << 4;
class SchedulingAbstractInterpreter : public SAIBaseExprFunctor {
 public:
  SchedulingAbstractInterpreter(const IRModule& mod, const FunctionSet& recursive_functions,
                                const CalleesMap& callees_map)
      : mod_(mod), recursive_functions_(recursive_functions), callees_map_(callees_map) {
    for (auto kv : mod_->functions) {
      func_name_map_[kv.second.get()] = kv.first->name_hint;
    }
  }

  void PerformAnalysis() {
    auto main_func = Downcast<Function>(mod_->Lookup("main"));
    for (auto arg : main_func->params) {
      var_states_[arg] = TupleDepth(GetTypeSize(arg->checked_type()), 0);
    }

    auto call_graph = CallGraph(mod_);
    size_t i = 0;
    for (; i < max_iterations_; ++i) {
      visited_functions_.clear();
      changed_ = false;
      for (auto cge : call_graph->TopologicalOrder()) {
        auto func = mod_->Lookup(cge->GetGlobalVar());
        if (auto fn = func.as<FunctionNode>()) {
          VisitExpr(fn->body);
        }
      }
      if (!changed_) {
        break;
      }
    }

    std::cout << "[SAI] Iterations: " << i << std::endl;
    for (auto kv : prim_func_call_depths_) {
      std::cout << "[SAI]  Call Depths: " << kv.first->op << " " << kv.second << std::endl;
    }
  }

 private:
  size_t GetTypeSize(const Type& type) {
    size_t state_size = 1;
    if (auto ttn = type.as<TupleTypeNode>()) {
      state_size = ttn->fields.size();
    }
    return state_size;
  }
  std::pair<TupleDepth, bool> Merge(const TupleDepth& vals1, const TupleDepth& vals2) {
    bool changed = false;
    ICHECK_EQ(vals1.size(), vals2.size());
    TupleDepth result(vals1.size());
    for (size_t i = 0; i < vals1.size(); ++i) {
      int new_val = std::max(vals1[i], vals2[i]);
      if (new_val != vals1[i]) {
        changed = true;
      }
      result[i] = new_val;
    }
    return std::make_pair(result, changed);
  }

  int Collapse(const TupleDepth& vals) {
    int result = 0;
    for (auto v : vals) {
      result = std::max(result, v);
    }
    return result;
  }

  template <typename T, typename MapType>
  void Add(MapType& map, const T& key, const TupleDepth& to_add) {
    auto it = map.find(key);
    if (it == map.end()) {
      map[key] = to_add;
      changed_ = true;
    } else {
      auto res = Merge(map[key], to_add);
      map[key] = res.first;
      changed_ |= res.second;
    }
  }

  void Add(const Var& var, const TupleDepth& to_add) {
    Add<Var, SAIVarStateMap>(var_states_, var, to_add);
  }

  void Add(const FunctionWithTemplateTypes& var, const TupleDepth& to_add) {
    Add<FunctionWithTemplateTypes, SAIFunctionStateMap>(function_states_, var, to_add);
  }

  int CappedAdd(int n, int incr) {
    if (n == MAX_DEPTH_VALUE) {
      return n;
    } else {
      ICHECK_LT(static_cast<long>(n) + static_cast<long>(incr), static_cast<long>(MAX_DEPTH_VALUE));
      return n + incr;
    }
  }

  TupleDepth VisitBody(const CallNode* op, Function& func) {
    Map<TypeVar, Type> type_scope_map;
    auto substituted_type_args = GetSubstitutedTypes(op);
    ICHECK_EQ(op->type_args.size(), func->type_params.size());
    for (size_t i = 0; i < op->type_args.size(); ++i) {
      type_scope_map.Set(func->type_params[i], substituted_type_args[i]);
    }
    type_scopes_.push(type_scope_map);
    auto res = VisitExpr(func->body);
    type_scopes_.pop();
    return res;
  }

  Array<Type> GetSubstitutedTypes(const CallNode* op) {
    Array<Type> ret;
    auto& current = type_scopes_.top();
    for (size_t i = 0; i < op->type_args.size(); ++i) {
      ret.push_back(TypeSubst(op->type_args[i], current));
    }
    return ret;
  }

  TupleDepth VisitExpr_(const ConstantNode* op) { return {0}; }

  TupleDepth VisitExpr_(const TupleNode* op) {
    TupleDepth tuple_depth(op->fields.size());
    for (size_t i = 0; i < op->fields.size(); ++i) {
      tuple_depth[i] = Collapse(VisitExpr(op->fields[i]));
    }
    return tuple_depth;
  }

  TupleDepth VisitExpr_(const VarNode* op) {
    auto var = GetRef<Var>(op);
    auto it = var_states_.find(var);
    if (it == var_states_.end()) {
      var_states_[var] = TupleDepth(GetTypeSize(op->checked_type()), 0);
      changed_ = true;
    }
    return var_states_[var];
  }

  TupleDepth VisitExpr_(const GlobalVarNode* op) { return {0}; }

  TupleDepth VisitExpr_(const FunctionNode* op) { return {0}; }

  TupleDepth VisitExpr_(const CallNode* op) {
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      return VisitExpr(on_device_props.body);
    }

    if (op->op == GetInvokeTVMOp()) {
      auto callee_prim_func = mod_->Lookup(Downcast<GlobalVar>(op->args[0]));
      auto access_modes_opt =
          callee_prim_func->GetAttr<Array<Integer>>(tir::attr::kDBArgAccessModes);
      ICHECK(access_modes_opt) << "No access modes found for " << op->args[0];
      auto access_modes = access_modes_opt.value();

      auto inputs_tuple = op->args[1];
      auto inputs_depth = VisitExpr(inputs_tuple);
      auto max_inputs_depth = Collapse(inputs_depth);
      auto output_depth = CappedAdd(max_inputs_depth, 1);
      auto outputs_tuple = op->args[2].as<TupleNode>();
      for (auto output : outputs_tuple->fields) {
        ICHECK(output.as<VarNode>());
        auto var = Downcast<Var>(output);
        Add(var, TupleDepth({output_depth}));
      }
      prim_func_call_depths_[op] = std::max(prim_func_call_depths_[op], max_inputs_depth);
      return TupleDepth(GetTypeSize(op->checked_type()), output_depth);
    } else if (op->op.as<OpNode>()) {
      return TupleDepth(GetTypeSize(op->checked_type()), 0);
    } else {
      auto callees = callees_map_.at(op);
      bool callee_may_be_recursive = false;
      for (auto callee : callees) {
        std::cout << "[ADV]   Callee " << func_name_map_[callee] << std::endl;
        auto callee_fn = GetRef<Function>(callee);
        if (recursive_functions_.count(callee)) {
          std::cout << "[ADV]    Recursive" << std::endl;
          callee_may_be_recursive = true;
          break;
        }
      }
      if (callee_may_be_recursive) {
        return TupleDepth(GetTypeSize(op->checked_type()), MAX_DEPTH_VALUE);
      }

      TupleDepth ret(GetTypeSize(op->checked_type()), 0);
      for (auto callee : callees) {
        auto callee_fn = GetRef<Function>(callee);
        FunctionWithTemplateTypes key(callee_fn, GetSubstitutedTypes(op));
        if (visited_functions_.count(key)) {
          auto it = function_states_.find(key);
          if (it != function_states_.end()) {
            ret = Merge(ret, it->second).first;
          }
        } else {
          for (size_t i = 0; i < op->args.size(); ++i) {
            auto param_state = VisitExpr(op->args[i]);
            Add(callee->params[i], param_state);
          }
          visited_functions_.insert(key);
          auto res = VisitBody(op, callee_fn);
          Add(key, res);
          if (ret.size() != res.size()) {
            std::cout << "[SAI] Mismatched " << op->checked_type() << " " << callee->ret_type
                      << std::endl;
          }
          ret = Merge(ret, res).first;
        }
      }
      return ret;
    }
  }

  TupleDepth VisitExpr_(const LetNode* op) {
    auto value_res = VisitExpr(op->value);
    Add(op->var, value_res);
    return VisitExpr(op->body);
  }

  TupleDepth VisitExpr_(const IfNode* op) {
    VisitExpr(op->cond);
    auto then_ret = VisitExpr(op->true_branch);
    auto else_ret = VisitExpr(op->false_branch);
    return Merge(then_ret, else_ret).first;
  }

  TupleDepth VisitExpr_(const OpNode* op) { return {0}; }

  TupleDepth VisitExpr_(const TupleGetItemNode* op) {
    auto tuple_res = VisitExpr(op->tuple);
    ICHECK_GT(tuple_res.size(), op->index);
    auto field_type = op->checked_type_.as<TupleTypeNode>()->fields[op->index];
    return TupleDepth(GetTypeSize(field_type), tuple_res[op->index]);
  }

  TupleDepth VisitExpr_(const RefCreateNode* op) { return {}; }
  TupleDepth VisitExpr_(const RefReadNode* op) { return {}; }
  TupleDepth VisitExpr_(const RefWriteNode* op) { return {}; }

  TupleDepth VisitExpr_(const ConstructorNode* op) { return {0}; }

  TupleDepth VisitExpr_(const MatchNode* op) {
    TupleDepth ret(GetTypeSize(op->checked_type()), 0);
    for (auto clause : op->clauses) {
      auto clause_ret = VisitExpr(clause->rhs);
      ret = Merge(ret, clause_ret).first;
    }
    return ret;
  }

  const IRModule& mod_;
  const FunctionSet& recursive_functions_;
  const CalleesMap& callees_map_;

  SAIVarStateMap var_states_;
  SAIFunctionStateMap function_states_;
  SAIInvokeTVMOpDepthMap prim_func_call_depths_;
  std::unordered_set<FunctionWithTemplateTypes, StructuralHash, StructuralEqual> visited_functions_;
  size_t max_iterations_{50};
  bool changed_{false};
  std::unordered_map<const BaseFuncNode*, std::string> func_name_map_;
  std::stack<Map<TypeVar, Type>> type_scopes_;
};

IRModule ComputeConstantDepths(IRModule& mod) {
  auto recursive_res = GetRecursiveFunctions(mod);
  auto recursive_functions = recursive_res.first;
  auto callees_map = recursive_res.second;
  SchedulingAbstractInterpreter analysis(mod, recursive_functions, callees_map);
  analysis.PerformAnalysis();
  return mod;
}

namespace transform {
Pass HoistNonSequentialOps() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return ComputeConstantDepths(m); };
  return CreateModulePass(pass_func, 0, "HoistNonSequentialOps", {});
}

TVM_REGISTER_GLOBAL("relay._transform.HoistNonSequentialOps").set_body_typed(HoistNonSequentialOps);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
