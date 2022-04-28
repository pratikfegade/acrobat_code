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

#include "model_parameter_taint_analysis.h"

#include <tvm/ir/attrs.h>
#include <tvm/ir/function.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/call.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/dynamic_batching.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>

#include "../../support/utils.h"
#include "../analysis/call_graph.h"
#include "../op/memory/on_device.h"
#include "../transforms/function_pointer_analysis.h"
#include "../transforms/pass_utils.h"

namespace tvm {
namespace relay {
namespace tec {

namespace {
using ContextT = const Object*;
using TaintT = Array<Bool>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

class ExecutableFunction;

/*!
 * \brief ExecutableFunction tensor type.
 */
class ExecutableFunctionNode : public Object {
 public:
  Function function;
  Map<Var, ObjectRef> environment;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("function", &function);
    v->Visit("environment", &environment);
  }

  bool SEqualReduce(const ExecutableFunctionNode* other, SEqualReducer equal) const {
    return equal(function, other->function) && equal(environment, other->environment);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(function);
    hash_reduce(environment);
  }

  static constexpr const char* _type_key = "relay.ExecutableFunction";
  TVM_DECLARE_FINAL_OBJECT_INFO(ExecutableFunctionNode, Object);
};

class ExecutableFunction : public ObjectRef {
 public:
  /*!
   * \brief The constructor
   * \param data The data of the constant tensor.
   * \param span The source span of the expression.
   */
  TVM_DLL explicit ExecutableFunction(Function function,
                                      Map<Var, ObjectRef> environment = Map<Var, ObjectRef>());

  TVM_DEFINE_OBJECT_REF_METHODS(ExecutableFunction, ObjectRef, ExecutableFunctionNode);
};

ExecutableFunction::ExecutableFunction(Function function, Map<Var, ObjectRef> environment) {
  ObjectPtr<ExecutableFunctionNode> n = make_object<ExecutableFunctionNode>();
  n->function = std::move(function);
  n->environment = std::move(environment);
  data_ = std::move(n);
}

using ExecutableFunctionSet = Map<ExecutableFunction, Bool>;

class FullTaint;
/*!
 * \brief FullTaint tensor type.
 */
class FullTaintNode : public Object {
 public:
  TaintT taint;
  ExecutableFunctionSet function_points_to;

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
  TVM_DLL explicit FullTaint(TaintT taint,
                             ExecutableFunctionSet function_points_to = ExecutableFunctionSet());

  TVM_DEFINE_OBJECT_REF_METHODS(FullTaint, ObjectRef, FullTaintNode);
};

FullTaint::FullTaint(TaintT taint, ExecutableFunctionSet function_points_to) {
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

Array<Bool> ConstructBoolArray(size_t size, bool value) {
  Array<Bool> result;
  for (size_t i = 0; i < size; ++i) {
    result.push_back(Bool(value));
  }
  return std::move(result);
}

int GetTypeSize(const Type& type) {
  size_t state_size = 1;
  if (auto ttn = type.as<TupleTypeNode>()) {
    state_size = ttn->fields.size();
  }
  return state_size;
}

Array<Bool> CreateStateForType(const Type& type, bool value) {
  return ConstructBoolArray(GetTypeSize(type), value);
}

Array<Bool> SingletonArray(bool value) { return ConstructBoolArray(1, value); }

class TaintAnalysis : public BaseExprFunctor {
 public:
  TaintAnalysis(const IRModule& mod) : mod_(mod) {}

  Map<Function, Array<Bool>> PerformAnalysis() {
    auto main_func = Downcast<Function>(mod_->Lookup("main"));

    environment_stack_.push(Map<Var, ObjectRef>());

    this->VisitExpr(main_func->body, GetInitialContext());

    std::unordered_map<const VarNode*, FullTaint> merged_var_states;
    for (auto kv : var_states_) {
      auto it = merged_var_states.find(kv.first.second);
      if (it != merged_var_states.end()) {
        merged_var_states.insert(
            std::make_pair(kv.first.second, Merge(Array<FullTaint>({it->second, kv.second}))));
      } else {
        merged_var_states.insert(std::make_pair(kv.first.second, kv.second));
      }
    }

    std::unordered_map<const FunctionNode*, FullTaint> merged_function_states;
    for (auto kv : function_states_) {
      auto it = merged_function_states.find(kv.first.second);
      if (it != merged_function_states.end()) {
        merged_function_states.insert(
            std::make_pair(kv.first.second, Merge(Array<FullTaint>({it->second, kv.second}))));
      } else {
        merged_function_states.insert(std::make_pair(kv.first.second, kv.second));
      }
    }

    if (true) {
      // for (auto kv : merged_var_states) {
      // std::cout << "[MPTA]  Merged Var: " << kv.first->vid->name_hint << " " << kv.second->taint
      // << std::endl;
      // }

      // for (auto kv : merged_function_states) {
      // std::cout << "[MPTA]  Function Depths: " << func_name_map_[kv.first] << " "
      // << support::PrintVector(kv.second) << std::endl;
      // }
    }

    Map<Function, Array<Bool>> results_map;
    for (auto kv : merged_function_states) {
      auto fn = GetRef<Function>(kv.first);
      Array<Bool> param_states;
      for (auto arg : fn->params) {
        auto it = merged_var_states.find(arg.get());
        if (it == merged_var_states.end()) {
          merged_var_states[arg.get()] =
              FullTaint(CreateStateForType(arg->checked_type(), false), ExecutableFunctionSet());
        }
        auto arg_state = merged_var_states[arg.get()]->taint;

        ICHECK_EQ(arg_state.size(), GetTypeSize(arg->checked_type())) << arg->checked_type();

        for (auto s : arg_state) {
          param_states.push_back(s);
        }
      }
      auto iit = merged_function_states.find(fn.get());
      if (iit == merged_function_states.end()) {
        merged_function_states[fn.get()] =
            FullTaint(CreateStateForType(fn->ret_type, false), ExecutableFunctionSet());
      }
      auto fn_state = merged_function_states[fn.get()]->taint;
      for (auto s : fn_state) {
        param_states.push_back(s);
      }

      std::cout << "[MPTA] Function " << fn << " " << param_states << std::endl;
      results_map.Set(fn, param_states);
    }

    return results_map;
  }

 private:
  Array<Bool> MergeTaints(const Array<Bool>& vals1, const Array<Bool>& vals2) {
    ICHECK_EQ(vals1.size(), vals2.size());
    Array<Bool> result;
    for (size_t i = 0; i < vals1.size(); ++i) {
      bool new_val = vals1[i].operator bool() && vals2[i].operator bool();
      result.push_back(Bool(new_val));
    }
    return result;
  }

  ExecutableFunctionSet MergeFunctionSets(const ExecutableFunctionSet& set1,
                                          const ExecutableFunctionSet& set2) {
    ExecutableFunctionSet result(set1);
    for (auto kv : set2) {
      result.Set(kv.first, kv.second);
    }
    return result;
  }

  FullTaint Merge(const Array<FullTaint>& full_taints) {
    TaintT taint = full_taints[0]->taint;
    ExecutableFunctionSet function_points_to = full_taints[0]->function_points_to;
    for (size_t i = 1; i < full_taints.size(); ++i) {
      auto this_taint = full_taints[i]->taint;
      auto this_function_points_to = full_taints[i]->function_points_to;
      taint = MergeTaints(taint, this_taint);
      function_points_to = MergeFunctionSets(function_points_to, this_function_points_to);
    }
    return FullTaint(taint, function_points_to);
  }

  Bool CollapseTaint(const Array<Bool>& vals) {
    bool res = true;
    for (auto b : vals) {
      res = res && b.operator bool();
    }
    return Bool(res);
  }

  template <typename T, typename MapType>
  FullTaint Add(MapType& map, const T& obj, ContextT context, const FullTaint& to_add) {
    auto key = std::make_pair(context, obj.get());
    auto it = map.find(key);
    if (it == map.end()) {
      map.insert(std::make_pair(key, to_add));
    } else {
      map.insert(std::make_pair(key, Merge(Array<FullTaint>({it->second, to_add}))));
    }
    return map.at(key);
  }

  FullTaint Add(const Var& var, ContextT current_context, const FullTaint to_add,
                const std::string& reason) {
    return Add<Var, VarStateMap>(var_states_, var, current_context, to_add);
  }

  FullTaint Add(const Function& fn, ContextT current_context, const FullTaint to_add) {
    return Add<Function, FunctionStateMap>(function_states_, fn, current_context, to_add);
  }

  FullTaint GetOrCreateFunctionState(const Function& fn, ContextT context) {
    auto key = std::make_pair(context, fn.get());
    auto it = function_states_.find(key);
    if (it == function_states_.end()) {
      auto taint = CreateStateForType(fn->ret_type, true);
      auto function_points_to = ExecutableFunctionSet();
      function_states_.insert(std::make_pair(key, FullTaint(taint, function_points_to)));
    }
    return function_states_.at(key);
  }

  ContextT GetInitialContext() { return nullptr; }

  FullTaint VisitExpr_(const ConstantNode* op, ContextT current_context) {
    auto taint = Array<Bool>({Bool(true)});
    auto function_points_to = ExecutableFunctionSet();
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
    auto current_environment = environment_stack_.top();
    auto it = current_environment.find(var);
    if (it != current_environment.end()) {
      return Downcast<FullTaint>((*it).second);
    }

    auto key = std::make_pair(current_context, op);
    auto iit = var_states_.find(key);
    if (iit == var_states_.end()) {
      var_states_.insert(std::make_pair(key, CreateStateForType(op->checked_type(), true)));
    }
    return var_states_.at(key);
  }

  FullTaint VisitExpr_(const GlobalVarNode* op, ContextT current_context) {
    auto taint = Array<Bool>({Bool(false)});
    auto function_points_to = ExecutableFunctionSet();

    auto base_func = mod_->Lookup(GetRef<GlobalVar>(op));
    if (base_func.as<FunctionNode>()) {
      function_points_to.Set(ExecutableFunction(Downcast<Function>(base_func)), Bool(true));
    }
    return FullTaint(taint, function_points_to);
  }

  VarSet GetFreeVars(const Function& expr) {
    auto it = free_vars_map_.find(expr.get());
    if (it == free_vars_map_.end()) {
      free_vars_map_.insert(std::make_pair(expr.get(), FreeVarsDedup(expr)));
    }
    return free_vars_map_.at(expr.get());
  }

  FullTaint VisitExpr_(const FunctionNode* op, ContextT current_context) {
    auto function = GetRef<Function>(op);
    auto free_vars = GetFreeVars(function);
    Map<Var, ObjectRef> environment;
    for (auto free_var : free_vars) {
      environment.Set(free_var, this->VisitExpr(free_var, current_context));
    }
    ExecutableFunction executable_function = ExecutableFunction(function, environment);
    ExecutableFunctionSet function_points_to;
    function_points_to.Set(executable_function, Bool(true));
    TaintT taint = SingletonArray(false);
    return FullTaint(taint, function_points_to);
  }

  FullTaint VisitBody(const ExecutableFunction& fn, ContextT context) {
    environment_stack_.push(fn->environment);
    on_stack_.insert(fn->function.get());
    return this->VisitExpr(fn->function->body, context);
    on_stack_.erase(fn->function.get());
    environment_stack_.pop();
  }

  FullTaint VisitExpr_(const CallNode* op, ContextT current_context) {
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      return this->VisitExpr(on_device_props.body, current_context);
    }

    if (op->op.as<OpNode>() || op->op.as<ConstructorNode>()) {
      auto taint = CreateStateForType(op->checked_type(), false);
      auto function_points_to = ExecutableFunctionSet();
      return FullTaint(taint, function_points_to);
    }

    auto caller_context = current_context;
    auto callee_context = op;

    auto op_full_taint = this->VisitExpr(op->op, current_context);
    auto callees = op_full_taint->function_points_to;
    Array<FullTaint> callee_taints;
    for (auto kv : callees) {
      auto executable_callee = kv.first;
      auto callee_fn = executable_callee->function;

      ICHECK_EQ(callee_fn->params.size(), op->args.size());
      for (size_t i = 0; i < callee_fn->params.size(); ++i) {
        auto arg = op->args[i];
        auto param = callee_fn->params[i];
        auto arg_full_taint = this->VisitExpr(arg, current_context);
        Add(param, callee_context, arg_full_taint, "function param");
      }

      if (on_stack_.count(callee_fn.get())) {
        callee_taints.push_back(GetOrCreateFunctionState(callee_fn, callee_context));
      } else {
        auto callee_full_taint = VisitBody(executable_callee, callee_context);
        callee_full_taint = Add(callee_fn, callee_context, callee_full_taint);
        callee_taints.push_back(callee_full_taint);
      }
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
    auto taint = Array<Bool>({Bool(false)});
    auto function_points_to = ExecutableFunctionSet();
    return FullTaint(taint, function_points_to);
  }

  FullTaint VisitExpr_(const TupleGetItemNode* op, ContextT current_context) {
    auto tuple_taint = this->VisitExpr(op->tuple, current_context);
    ICHECK_GT(tuple_taint->taint.size(), op->index);
    auto taint = Array<Bool>({tuple_taint->taint[op->index]});
    auto function_points_to = tuple_taint->function_points_to;
    return FullTaint(taint, function_points_to);
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
    auto taint = Array<Bool>({Bool(false)});
    auto function_points_to = ExecutableFunctionSet();
    return FullTaint(taint, function_points_to);
  }

  FullTaint VisitExpr_(const ConstructorNode* op, ContextT current_context) {
    auto taint = Array<Bool>({Bool(false)});
    auto function_points_to = ExecutableFunctionSet();
    return FullTaint(taint, function_points_to);
  }

  FullTaint VisitExpr_(const MatchNode* op, ContextT current_context) {
    ICHECK(!op->data->checked_type().as<TupleTypeNode>());
    auto data_full_taint = this->VisitExpr(op->data, current_context);
    auto collapsed_data_taint = CollapseTaint(data_full_taint->taint);

    Array<FullTaint> clause_taints;
    for (auto clause : op->clauses) {
      auto pattern_vars = CollectPatternVars(clause->lhs);
      for (auto& var : pattern_vars) {
        Add(var, current_context,
            FullTaint(Array<Bool>({collapsed_data_taint}), data_full_taint->function_points_to),
            "Match pattern");
      }
      clause_taints.push_back(this->VisitExpr(clause->rhs, current_context));
    }

    return Merge(clause_taints);
  }

  const IRModule& mod_;

  VarStateMap var_states_;
  FunctionStateMap function_states_;
  std::unordered_map<const FunctionNode*, VarSet> free_vars_map_;
  std::unordered_set<const FunctionNode*> on_stack_;
  std::stack<Map<Var, ObjectRef>> environment_stack_;
};

}  // namespace

Map<Function, Array<Bool>> ModelParameterTaintAnalysis(const IRModule& mod) {
  TaintAnalysis analysis(mod);
  auto ret = analysis.PerformAnalysis();
  exit(0);
  return ret;
}

}  // namespace tec
}  // namespace relay
}  // namespace tvm
