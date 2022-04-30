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
#include "./function_pointer_analysis.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {

constexpr int MAX_DEPTH_VALUE = 1 << 4;

namespace {
using ContextT = const Object*;
using TaintT = Integer;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

using FunctionSet = Map<Function, Bool>;

class FullTaint;
/*!
 * \brief FullTaint tensor type.
 */
class FullTaintNode : public Object {
 public:
  TaintT taint;
  FunctionSet function_points_to;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("taint", &taint);
    v->Visit("function_points_to", &function_points_to);
  }

  bool SEqualReduce(const FullTaintNode* other, SEqualReducer equal) const {
    return equal(taint, other->taint) && equal(function_points_to, other->function_points_to);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(taint);
    hash_reduce(function_points_to);
  }

  static constexpr const char* _type_key = "relay.FullTaint";
  TVM_DECLARE_FINAL_OBJECT_INFO(FullTaintNode, Object);
};

class FullTaint : public ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param data The data of the constant tensor.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit FullTaint(TaintT taint, FunctionSet function_points_to = FunctionSet());

  TVM_DEFINE_OBJECT_REF_METHODS(FullTaint, ObjectRef, FullTaintNode);
};

FullTaint::FullTaint(TaintT taint, FunctionSet function_points_to) {
  ObjectPtr<FullTaintNode> n = make_object<FullTaintNode>();
  n->taint = std::move(taint);
  n->function_points_to = std::move(function_points_to);
  data_ = std::move(n);
}

using BaseExprFunctor = ExprFunctor<FullTaint(const Expr& n, ContextT)>;
using VarStateMap =
    std::unordered_map<std::pair<ContextT, const VarNode*>, FullTaint, PairHash, PairEquals>;
using FunctionStateMap =
    std::unordered_map<std::pair<ContextT, const FunctionNode*>, FullTaint, PairHash, PairEquals>;
using FunctionEnvironmentMap = std::unordered_map<const FunctionNode*, Map<Var, FullTaint>>;
using OnStackSet =
    std::unordered_set<std::pair<ContextT, const FunctionNode*>, PairHash, PairEquals>;

class TaintAnalysis : public BaseExprFunctor {
 public:
  TaintAnalysis(const IRModule& mod) : mod_(mod) {
    if (mod->ContainGlobalVar("map")) {
      map_fun_node_ = mod->Lookup("map").as<FunctionNode>();
    }
  }

  IRModule PerformAnalysis() {
    auto main_func = Downcast<Function>(mod_->Lookup("main"));

    ICHECK_EQ(main_func->type_params.size(), 0);
    for (auto param : main_func->params) {
      ICHECK(!param->checked_type().as<FuncTypeNode>());
      Add(param, GetInitialContext(), FullTaint(Integer(0), FunctionSet()), "Main param");
    }

    for (size_t i = 0; i < MAX_DEPTH_VALUE; ++i) {
      // std::cout << "[MPT] ITERATION " << i << std::endl;

      this->Reset();
      environment_stack_.push_back(Map<Var, FullTaint>());
      auto context_fn_key = std::make_pair(GetInitialContext(), main_func.get());
      on_stack_.insert(context_fn_key);
      function_stack_.push_back(main_func);
      auto full_taint = this->VisitExpr(main_func->body, GetInitialContext());
      function_stack_.pop_back();
      on_stack_.erase(context_fn_key);
      environment_stack_.pop_back();
    }

    std::unordered_map<const VarNode*, FullTaint> merged_var_states;
    for (auto kv : var_states_) {
      auto merged_taint = kv.second;
      auto it = merged_var_states.find(kv.first.second);
      if (it != merged_var_states.end()) {
        merged_taint = Merge(Array<FullTaint>({it->second, kv.second}));
      }
      merged_var_states[kv.first.second] = merged_taint;
    }

    std::unordered_map<const FunctionNode*, FullTaint> merged_function_states;
    for (auto kv : function_states_) {
      auto merged_taint = kv.second;

      auto it = merged_function_states.find(kv.first.second);
      if (it != merged_function_states.end()) {
        merged_taint = Merge(Array<FullTaint>({it->second, kv.second}));
      }
      merged_function_states[kv.first.second] = merged_taint;
    }

    std::unordered_map<const CallNode*, Integer> merged_op_depths;
    for (auto kv : prim_func_call_depths_) {
      auto it = merged_op_depths.find(kv.first.second);
      if (it == merged_op_depths.end()) {
        merged_op_depths[kv.first.second] = kv.second;
      } else {
        merged_op_depths[kv.first.second] = MergeTaints(it->second, kv.second);
      }
      merged_op_depths[kv.first.second] = MergeTaints(merged_op_depths[kv.first.second], kv.second);
    }

    class AddDepthToCalls : public ExprMutator {
     public:
      AddDepthToCalls(const std::unordered_map<const CallNode*, Integer>& merged_op_depths)
          : merged_op_depths_(merged_op_depths) {}

      Expr VisitExpr_(const CallNode* op) {
        auto it = merged_op_depths_.find(op);
        auto call = ExprMutator::VisitExpr_(op);
        if (it != merged_op_depths_.end() && it->second->value < MAX_DEPTH_VALUE) {
          auto new_op = call.as<CallNode>();
          auto depth = it->second;
          ICHECK(new_op->attrs.defined() && new_op->attrs.as<DictAttrsNode>());
          auto attrs = Downcast<DictAttrs>(new_op->attrs);
          auto new_attrs = attrs.WithAttr(tir::attr::kDBGraphDepth, depth);
          return Call(new_op->op, new_op->args, new_attrs, new_op->type_args, new_op->span);
        }
        return call;
      }

      const std::unordered_map<const CallNode*, Integer>& merged_op_depths_;
    };

    AddDepthToCalls mutator(merged_op_depths);
    Map<GlobalVar, Function> new_funcs;
    for (const auto& it : mod_->functions) {
      if (it.second.as<FunctionNode>()) {
        auto func = Downcast<Function>(it.second);
        func = Downcast<Function>(mutator(func));
        new_funcs.Set(it.first, func);
      }
    }
    for (auto pair : new_funcs) {
      mod_->Add(pair.first, pair.second, true);
    }

    return mod_;
  }

 private:
  std::string GetFunctionName(const Function& fn) {
    return fn->GetAttr<String>("db.function_name").value();
  }

  void Reset() {
    on_stack_.clear();
    environment_stack_.clear();
  }

  Integer CappedIncr(Integer n) {
    if (n->value >= MAX_DEPTH_VALUE) {
      return n;
    } else {
      ICHECK_LE(static_cast<long>(n->value) + static_cast<long>(1),
                static_cast<long>(MAX_DEPTH_VALUE));
      return Integer(n->value + 1);
    }
  }

  TaintT MergeTaints(const TaintT& vals1, const TaintT& vals2) {
    return TaintT(std::max(vals1->value, vals2->value));
  }

  FunctionSet MergeFunctionSets(const FunctionSet& set1, const FunctionSet& set2) {
    FunctionSet result;
    for (auto kv : set1) {
      result.Set(kv.first, kv.second);
    }
    for (auto kv : set2) {
      result.Set(kv.first, kv.second);
    }
    return result;
  }

  FullTaint Merge(const Array<FullTaint>& full_taints) {
    TaintT taint = Integer(0);
    FunctionSet function_points_to;
    for (auto& full_taint : full_taints) {
      auto this_taint = full_taint->taint;
      auto this_function_points_to = full_taint->function_points_to;
      taint = MergeTaints(taint, this_taint);
      function_points_to = MergeFunctionSets(function_points_to, this_function_points_to);
    }
    return FullTaint(taint, function_points_to);
  }

  template <typename T, typename MapType>
  FullTaint Add(MapType& map, const T& obj, ContextT context, const FullTaint& to_add) {
    auto key = std::make_pair(context, obj.get());
    auto it = map.find(key);
    if (it == map.end()) {
      map[key] = to_add;
      return to_add;
    } else {
      auto merged = Merge(Array<FullTaint>({it->second, to_add}));
      map[key] = merged;
      return merged;
    }
  }

  FullTaint Add(const Var& var, ContextT current_context, const FullTaint& to_add,
                const std::string& reason) {
    return Add<Var, VarStateMap>(var_states_, var, current_context, to_add);
  }

  FullTaint Add(const Function& fn, ContextT current_context, const FullTaint& to_add) {
    return Add<Function, FunctionStateMap>(function_states_, fn, current_context, to_add);
  }

  FullTaint GetOrCreateFunctionState(const Function& fn, ContextT context) {
    auto key = std::make_pair(context, fn.get());
    auto it = function_states_.find(key);
    if (it == function_states_.end()) {
      auto taint = Integer(0);
      auto function_points_to = FunctionSet();
      auto full_taint = FullTaint(taint, function_points_to);
      return full_taint;
    } else {
      return it->second;
    }
  }

  ContextT GetInitialContext() { return nullptr; }

  FullTaint VisitExpr_(const ConstantNode* op, ContextT current_context) {
    auto taint = TaintT(0);
    auto function_points_to = FunctionSet();
    return FullTaint(taint, function_points_to);
  }

  FullTaint VisitExpr_(const TupleNode* op, ContextT current_context) {
    Array<FullTaint> field_taints;
    for (auto field : op->fields) {
      field_taints.push_back(this->VisitExpr(field, current_context));
    }
    return Merge(field_taints);
  }

  FullTaint VisitExpr_(const VarNode* op, ContextT current_context) {
    auto var = GetRef<Var>(op);
    auto current_environment = environment_stack_.back();
    auto it = current_environment.find(var);
    if (it != current_environment.end()) {
      auto full_taint = (*it).second;
      return full_taint;
    }

    auto key = std::make_pair(current_context, op);
    auto iit = var_states_.find(key);
    if (iit == var_states_.end()) {
      auto taint = FullTaint(Integer(0), FunctionSet());
      var_states_[key] = taint;
      return taint;
    } else {
      return iit->second;
    }
  }

  FullTaint VisitExpr_(const GlobalVarNode* op, ContextT current_context) {
    auto taint = TaintT(0);
    auto function_points_to = FunctionSet();

    auto base_func = mod_->Lookup(GetRef<GlobalVar>(op));
    if (base_func.as<FunctionNode>()) {
      function_points_to.Set(Downcast<Function>(base_func), Bool(true));
    }
    return FullTaint(taint, function_points_to);
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

  FullTaint VisitExpr_(const FunctionNode* op, ContextT current_context) {
    auto function = GetRef<Function>(op);
    auto free_vars = GetFreeVars(function);
    auto function_depth = TaintT(0);
    for (auto free_var : free_vars) {
      auto taint = this->VisitExpr(free_var, current_context);
      auto it = function_environments_[op].find(free_var);
      if (it != function_environments_[op].end()) {
        auto preexisting_taint = (*it).second;
        taint = Merge(Array<FullTaint>({taint, preexisting_taint}));
      }
      function_depth = MergeTaints(function_depth, taint->taint);
      function_environments_[op].Set(free_var, taint);
    }

    FunctionSet function_points_to;
    function_points_to.Set(function, Bool(true));
    return FullTaint(function_depth, function_points_to);
  }

  FullTaint VisitBody(const Function& fn, ContextT context) {
    environment_stack_.push_back(function_environments_[fn.get()]);
    auto context_fn_key = std::make_pair(context, fn.get());
    on_stack_.insert(context_fn_key);
    function_stack_.push_back(fn);
    auto full_taint = this->VisitExpr(fn->body, context);
    function_stack_.pop_back();
    on_stack_.erase(context_fn_key);
    environment_stack_.pop_back();
    return full_taint;
  }

  bool IsMapFuncInModule() { return map_fun_node_ != nullptr; }

  const FunctionNode* GetMapFuncNode() { return map_fun_node_; }

  ContextT GetMapLambdaContext(ContextT map_calling_context) {
    return reinterpret_cast<ContextT>(
        0x1000000000000000 | reinterpret_cast<uint64_t>(const_cast<Object*>(map_calling_context)));
  }

  FullTaint VisitMapBody(const CallNode* map_call, ContextT map_calling_context) {
    auto map_fn_node = GetMapFuncNode();
    auto map_lambda_var = map_fn_node->params[0];
    auto map_list_var = map_fn_node->params[1];

    auto map_list_taint = this->VisitExpr(map_list_var, map_calling_context);
    auto lambda_callees = this->VisitExpr(map_lambda_var, map_calling_context)->function_points_to;
    auto lambda_calling_context = GetMapLambdaContext(map_calling_context);

    Array<FullTaint> callee_taints;
    for (auto kv : lambda_callees) {
      auto callee_fn = kv.first;
      ICHECK_EQ(callee_fn->params.size(), 1);

      {
        auto param = callee_fn->params[0];
        ICHECK(!param->checked_type().as<FuncTypeNode>());
        auto lambda_arg_full_taint = map_list_taint;
        auto lambda_param_full_taint = lambda_arg_full_taint;
        Add(param, lambda_calling_context, lambda_param_full_taint, "map lambda param");
      }

      FullTaint callee_full_taint;
      if (on_stack_.count(std::make_pair(lambda_calling_context, callee_fn.get()))) {
        callee_full_taint = GetOrCreateFunctionState(callee_fn, lambda_calling_context);
      } else {
        callee_full_taint = VisitBody(callee_fn, lambda_calling_context);
        callee_full_taint = Add(callee_fn, lambda_calling_context, callee_full_taint);
      }
      callee_taints.push_back(callee_full_taint);
    }

    return Merge(callee_taints);
  }

  FullTaint VisitExpr_(const CallNode* op, ContextT current_context) {
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      return this->VisitExpr(on_device_props.body, current_context);
    }

    if (op->op == GetInvokeTVMOp()) {
      auto callee_prim_func = mod_->Lookup(Downcast<GlobalVar>(op->args[0]));

      auto inputs_full_taint = this->VisitExpr(op->args[1], current_context);
      auto max_inputs_depth = inputs_full_taint->taint;
      auto output_depth = CappedIncr(max_inputs_depth);
      auto outputs_tuple = op->args[2].as<TupleNode>();
      auto output_full_taint = FullTaint(output_depth, FunctionSet());
      for (auto output : outputs_tuple->fields) {
        ICHECK(output.as<VarNode>());
        auto var = Downcast<Var>(output);
        auto res = Add(var, current_context, output_full_taint, "OpOutput");
      }
      auto op_key = std::make_pair(current_context, op);
      auto it = prim_func_call_depths_.find(op_key);
      if (it == prim_func_call_depths_.end()) {
        prim_func_call_depths_[op_key] = max_inputs_depth;

      } else {
        prim_func_call_depths_[op_key] = MergeTaints(it->second, max_inputs_depth);
      }
      return output_full_taint;
    } else if (op->op.as<OpNode>()) {
      auto taint = TaintT(0);
      auto function_points_to = FunctionSet();
      return FullTaint(taint, function_points_to);
    } else if (op->op.as<ConstructorNode>()) {
      Array<FullTaint> arg_taints;
      for (auto arg : op->args) {
        arg_taints.push_back(this->VisitExpr(arg, current_context));
      }
      return Merge(arg_taints);
    }

    auto caller_context = current_context;
    auto callee_context = op;

    auto op_full_taint = this->VisitExpr(op->op, current_context);
    auto callees = op_full_taint->function_points_to;
    Array<FullTaint> callee_taints;
    for (auto kv : callees) {
      auto callee_fn = kv.first;

      ICHECK_EQ(callee_fn->params.size(), op->args.size());
      for (size_t i = 0; i < callee_fn->params.size(); ++i) {
        auto arg = op->args[i];
        auto param = callee_fn->params[i];
        auto arg_full_taint = this->VisitExpr(arg, current_context);
        auto param_full_taint = arg_full_taint;

        Add(param, callee_context, param_full_taint, "function param");
      }

      FullTaint callee_full_taint;
      if (IsMapFuncInModule() && callee_fn.get() == GetMapFuncNode()) {
        callee_full_taint = VisitMapBody(op, callee_context);
      } else if (on_stack_.count(std::make_pair(callee_context, callee_fn.get()))) {
        callee_full_taint = GetOrCreateFunctionState(callee_fn, callee_context);
      } else {
        callee_full_taint = VisitBody(callee_fn, callee_context);
        callee_full_taint = Add(callee_fn, callee_context, callee_full_taint);
      }
      callee_taints.push_back(callee_full_taint);
    }

    return Merge(callee_taints);
  }

  FullTaint VisitExpr_(const LetNode* op, ContextT current_context) {
    auto value_res = this->VisitExpr(op->value, current_context);
    Add(op->var, current_context, value_res, "Let value");
    return this->VisitExpr(op->body, current_context);
  }

  FullTaint VisitExpr_(const IfNode* op, ContextT current_context) {
    auto cond_taint = this->VisitExpr(op->cond, current_context);
    auto true_taint = this->VisitExpr(op->true_branch, current_context);
    auto false_taint = this->VisitExpr(op->false_branch, current_context);
    return Merge(Array<FullTaint>({true_taint, false_taint}));
  }

  FullTaint VisitExpr_(const OpNode* op, ContextT current_context) {
    auto taint = TaintT(0);
    auto function_points_to = FunctionSet();
    return FullTaint(taint, function_points_to);
  }

  FullTaint VisitExpr_(const TupleGetItemNode* op, ContextT current_context) {
    return this->VisitExpr(op->tuple, current_context);
  }

  FullTaint VisitExpr_(const RefCreateNode* op, ContextT current_context) {
    return UnsupportedFunction(op, current_context);
  }

  FullTaint VisitExpr_(const RefReadNode* op, ContextT current_context) {
    return UnsupportedFunction(op, current_context);
  }

  FullTaint VisitExpr_(const RefWriteNode* op, ContextT current_context) {
    return UnsupportedFunction(op, current_context);
  }

  FullTaint UnsupportedFunction(const ExprNode* op, ContextT current_context) {
    ICHECK(false) << "Do not support " << op->GetTypeKey();
    auto taint = TaintT(MAX_DEPTH_VALUE);
    auto function_points_to = FunctionSet();
    return FullTaint(taint, function_points_to);
  }

  FullTaint VisitExpr_(const ConstructorNode* op, ContextT current_context) {
    auto taint = Integer(0);
    auto function_points_to = FunctionSet();
    return FullTaint(taint, function_points_to);
  }

  FullTaint VisitExpr_(const MatchNode* op, ContextT current_context) {
    ICHECK(!op->data->checked_type().as<TupleTypeNode>());
    auto data_full_taint = this->VisitExpr(op->data, current_context);

    Array<FullTaint> clause_taints;
    for (auto clause : op->clauses) {
      auto pattern_vars = CollectPatternVars(clause->lhs);
      for (auto& var : pattern_vars) {
        Add(var, current_context, data_full_taint, "Match pattern");
      }
      clause_taints.push_back(this->VisitExpr(clause->rhs, current_context));
    }

    return Merge(clause_taints);
  }

  const IRModule& mod_;

  VarStateMap var_states_;
  FunctionStateMap function_states_;
  std::unordered_map<std::pair<ContextT, const CallNode*>, Integer, PairHash, PairEquals>
      prim_func_call_depths_;
  FunctionEnvironmentMap function_environments_;
  std::unordered_map<const FunctionNode*, VarSet> free_vars_map_;
  OnStackSet on_stack_;
  std::vector<Map<Var, FullTaint>> environment_stack_;

  std::vector<Function> function_stack_;
  const FunctionNode* map_fun_node_;
};

}  // namespace

IRModule ComputeConstantDepths(IRModule& mod) {
  std::cout << "[SAI] Starting hoisting" << std::endl;
  TaintAnalysis analysis(mod);
  return analysis.PerformAnalysis();
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
