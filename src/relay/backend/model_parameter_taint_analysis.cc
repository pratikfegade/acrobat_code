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
#include <tvm/runtime/container/structural_map.h>
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

class TaintAnalysis : public BaseExprFunctor {
 public:
  TaintAnalysis(const IRModule& mod) : mod_(mod) {}

  Map<Function, Array<Bool>> PerformAnalysis() {
    auto main_func = Downcast<Function>(mod_->Lookup("main"));

    ICHECK_EQ(main_func->type_params.size(), 0);
    for (auto param : main_func->params) {
      ICHECK(!param->checked_type().as<FuncTypeNode>());
      Add(param, GetInitialContext(),
          FullTaint(CreateStateForType(param->checked_type(), false), FunctionSet()), "Main param");
    }

    for (size_t i = 0; i < 10; ++i) {
      // std::cout << "[MPT] ITERATION " << i << std::endl;

      this->Reset();
      type_environment_stack_.push_back(Map<TypeVar, Type>());
      environment_stack_.push_back(Map<Var, FullTaint>());
      auto context_fn_key = std::make_pair(GetInitialContext(), main_func.get());
      on_stack_.insert(context_fn_key);
      function_stack_.push_back(main_func);
      auto full_taint = this->VisitExpr(main_func->body, GetInitialContext());
      function_stack_.pop_back();
      on_stack_.erase(context_fn_key);
      environment_stack_.pop_back();
      type_environment_stack_.pop_back();
    }

    // for (auto kv : callees_) {
    //   std::cout << "[MPT] CALLEE: " << kv.first->op << std::endl;
    //   for (auto fn : kv.second) {
    //     std::cout << "[MPT]   " << GetFunctionName(GetRef<Function>(fn)) << std::endl;
    //   }
    // }

    std::unordered_map<const VarNode*, FullTaint> merged_var_states;
    for (auto kv : var_states_) {
      auto merged_taint = kv.second;
      bool print = false;  // kv.first.second->vid->name_hint == "xsm";
      if (print) {
        std::cout << "[MPT] CVar " << kv.first.second << " " << kv.first.second->vid->name_hint
                  << " " << kv.second->taint << std::endl;
      }
      auto it = merged_var_states.find(kv.first.second);
      if (it != merged_var_states.end()) {
        merged_taint = Merge(Array<FullTaint>({it->second, kv.second}));
        if (print) {
          std::cout << "[MPT]  Merging with " << it->second->taint << " " << merged_taint->taint
                    << std::endl;
        }
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
              FullTaint(CreateStateForType(arg->checked_type(), false), FunctionSet());
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
            FullTaint(CreateStateForType(fn->ret_type, false), FunctionSet());
      }
      auto fn_state = merged_function_states[fn.get()]->taint;
      for (auto s : fn_state) {
        param_states.push_back(s);
      }

      // std::cout << "[MPT] Function " << fn->GetAttr<String>("db.function_name") << " "
      // << param_states << std::endl;
      results_map.Set(fn, param_states);
    }

    return results_map;
  }

 private:
  std::string GetFunctionName(const Function& fn) {
    return fn->GetAttr<String>("db.function_name").value();
  }

  void Reset() {
    on_stack_.clear();
    environment_stack_.clear();
    type_environment_stack_.clear();
  }

  Array<Bool> MergeTaints(const Array<Bool>& vals1, const Array<Bool>& vals2) {
    ICHECK_EQ(vals1.size(), vals2.size());
    Array<Bool> result;
    for (size_t i = 0; i < vals1.size(); ++i) {
      bool new_val = vals1[i].operator bool() && vals2[i].operator bool();
      result.push_back(Bool(new_val));
    }
    return result;
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
    TaintT taint = ConstructBoolArray(full_taints[0]->taint.size(), true);
    FunctionSet function_points_to;
    for (auto& full_taint : full_taints) {
      auto this_taint = full_taint->taint;
      auto this_function_points_to = full_taint->function_points_to;
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
    ICHECK_EQ(to_add->taint.size(), GetTypeSize(var->checked_type()))
        << var << " " << to_add->taint << " " << reason;

    // bool print = var->checked_type().as<TypeCallNode>() && current_context;

    auto ret = Add<Var, VarStateMap>(var_states_, var, current_context, to_add);
    // if (print) {
    //   std::cout << "[MPT] Added var " << current_context << " " << var->vid->name_hint << " "
    //             << to_add->taint << " " << ret->taint << " "
    //             << var_states_.at(std::make_pair(current_context, var.get()))->taint <<
    //             std::endl;
    // }
    return ret;
  }

  FullTaint Add(const Function& fn, ContextT current_context, const FullTaint& to_add) {
    ICHECK_EQ(to_add->taint.size(), GetTypeSize(fn->ret_type)) << fn << " " << to_add->taint;
    return Add<Function, FunctionStateMap>(function_states_, fn, current_context, to_add);
  }

  FullTaint GetOrCreateFunctionState(const Function& fn, ContextT context) {
    auto key = std::make_pair(context, fn.get());
    auto it = function_states_.find(key);
    if (it == function_states_.end()) {
      auto taint = CreateStateForType(fn->ret_type, true);
      auto function_points_to = FunctionSet();
      auto full_taint = FullTaint(taint, function_points_to);
      return full_taint;
    } else {
      return it->second;
    }
  }

  ContextT GetInitialContext() { return nullptr; }

  FullTaint ResizeTaint(const FullTaint& taint, const Type& from, const Type& to) {
    size_t from_size = GetTypeSize(from);
    size_t to_size = GetTypeSize(to);
    if (from_size == to_size) {
      return taint;
    }
    return FullTaint(ConstructBoolArray(to_size, CollapseTaint(taint->taint)),
                     taint->function_points_to);
  }

  FullTaint VisitExpr_(const ConstantNode* op, ContextT current_context) {
    auto taint = Array<Bool>({Bool(true)});
    auto function_points_to = FunctionSet();
    return FullTaint(taint, function_points_to);
  }

  FullTaint VisitExpr_(const TupleNode* op, ContextT current_context) {
    TaintT taint;
    FunctionSet function_points_to;
    for (auto field : op->fields) {
      auto field_full_taint = this->VisitExpr(field, current_context);
      taint.push_back(CollapseTaint(field_full_taint->taint));
      function_points_to =
          MergeFunctionSets(function_points_to, field_full_taint->function_points_to);
    }
    return FullTaint(taint, function_points_to);
  }

  FullTaint VisitExpr_(const VarNode* op, ContextT current_context) {
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
        std::cout << "[MPT]  In env " << full_taint->taint << std::endl;
      }
      return full_taint;
    }

    auto key = std::make_pair(current_context, op);
    auto iit = var_states_.find(key);
    if (iit == var_states_.end()) {
      auto taint = FullTaint(CreateStateForType(op->checked_type(), true), FunctionSet());
      var_states_[key] = taint;
      if (print) {
        std::cout << "[MPT]  Not found" << std::endl;
      }
      return taint;
    } else {
      if (print) {
        std::cout << "[MPT]  Found " << iit->second->taint << std::endl;
      }
      return iit->second;
    }
  }

  FullTaint VisitExpr_(const GlobalVarNode* op, ContextT current_context) {
    auto taint = Array<Bool>({Bool(false)});
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
    for (auto free_var : free_vars) {
      auto taint = this->VisitExpr(free_var, current_context);
      auto it = function_environments_[op].find(free_var);
      if (it != function_environments_[op].end()) {
        auto preexisting_taint = (*it).second;
        taint = Merge(Array<FullTaint>({taint, preexisting_taint}));
      }
      function_environments_[op].Set(free_var, taint);
    }

    FunctionSet function_points_to;
    function_points_to.Set(function, Bool(true));
    TaintT taint = ConstructBoolArray(1, false);
    return FullTaint(taint, function_points_to);
  }

  FullTaint VisitBody(const Function& fn, ContextT context, const CallNode* call) {
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

  FullTaint VisitExpr_(const CallNode* op, ContextT current_context) {
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      return this->VisitExpr(on_device_props.body, current_context);
    }

    if (op->op.as<OpNode>() || op->op.as<ConstructorNode>()) {
      auto taint = CreateStateForType(op->checked_type(), false);
      auto function_points_to = FunctionSet();
      return FullTaint(taint, function_points_to);
    }

    auto caller_context = current_context;
    auto callee_context = op;

    auto op_full_taint = this->VisitExpr(op->op, current_context);
    auto callees = op_full_taint->function_points_to;
    Array<FullTaint> callee_taints;
    // std::cout << "[MPT] Call " << op->op << " " << callees.size() << std::endl;
    for (auto kv : callees) {
      auto callee_fn = kv.first;
      // std::cout << "[MPT]  Callee " << GetFunctionName(callee_fn) << std::endl;

      //////////////////////////////
      callees_[op].insert(callee_fn.get());

      auto callee_name = callee_fn->GetAttr<String>("db.function_name");
      bool print = false;  //(callee_name == "map" || callee_name == "lstm_cell");
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
        auto arg_full_taint = this->VisitExpr(arg, current_context);
        auto param_full_taint =
            ResizeTaint(arg_full_taint, arg->checked_type(), param->checked_type());

        //////////////////////////////
        if (print) {
          std::string arg_str = "";
          if (auto vn = arg.as<VarNode>()) {
            arg_str = vn->vid->name_hint;
          }
          std::cout << "[MPT]  Param " << current_context << " " << arg_str << " "
                    << param->vid->name_hint << " " << param_full_taint->taint << std::endl;
        }
        //////////////////////////////

        Add(param, callee_context, param_full_taint, "function param");
      }

      FullTaint callee_return_full_taint;
      if (on_stack_.count(std::make_pair(callee_context, callee_fn.get())) ||
          // callee_name == "lstm_cell") {
          false) {
        callee_return_full_taint = GetOrCreateFunctionState(callee_fn, callee_context);
        if (print) {
          std::cout << "[MPT] Return value 1 " << callee_name << " "
                    << callee_return_full_taint->taint << std::endl;
        }
      } else {
        callee_return_full_taint = VisitBody(callee_fn, callee_context, op);
        callee_return_full_taint = Add(callee_fn, callee_context, callee_return_full_taint);
        if (print) {
          std::cout << "[MPT] Return value 2 " << callee_name << " "
                    << callee_return_full_taint->taint << std::endl;
        }
      }
      auto call_full_taint =
          ResizeTaint(callee_return_full_taint, callee_fn->ret_type, op->checked_type());
      callee_taints.push_back(call_full_taint);
    }

    return Merge(callee_taints);
  }

  FullTaint VisitExpr_(const LetNode* op, ContextT current_context) {
    auto value_res = this->VisitExpr(op->value, current_context);
    auto added_res = Add(op->var, current_context, value_res, "Let value");
    // if (op->var->checked_type().as<TypeCallNode>()) {
    //   std::cout << "[MPT]   Let " << current_context << " " << op->var->vid->name_hint << " "
    //             << value_res->taint << " " << added_res->taint << std::endl;
    // }
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
    auto function_points_to = FunctionSet();
    return FullTaint(taint, function_points_to);
  }

  FullTaint VisitExpr_(const TupleGetItemNode* op, ContextT current_context) {
    auto tuple_taint = this->VisitExpr(op->tuple, current_context);
    ICHECK_GT(tuple_taint->taint.size(), op->index);
    auto taint = CreateStateForType(op->checked_type(), tuple_taint->taint[op->index]);
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
    auto function_points_to = FunctionSet();
    return FullTaint(taint, function_points_to);
  }

  FullTaint VisitExpr_(const ConstructorNode* op, ContextT current_context) {
    auto taint = Array<Bool>({Bool(false)});
    auto function_points_to = FunctionSet();
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
            FullTaint(CreateStateForType(var->checked_type(), collapsed_data_taint),
                      data_full_taint->function_points_to),
            "Match pattern");
      }
      clause_taints.push_back(this->VisitExpr(clause->rhs, current_context));
    }

    return Merge(clause_taints);
  }

  const IRModule& mod_;

  VarStateMap var_states_;
  FunctionStateMap function_states_;
  FunctionEnvironmentMap function_environments_;
  std::unordered_map<const FunctionNode*, VarSet> free_vars_map_;
  OnStackSet on_stack_;
  std::vector<Map<Var, FullTaint>> environment_stack_;
  std::vector<Map<TypeVar, Type>> type_environment_stack_;

  std::vector<Function> function_stack_;
  std::unordered_map<const CallNode*, std::unordered_set<const FunctionNode*>> callees_;
};

}  // namespace

Map<Function, Array<Bool>> ModelParameterTaintAnalysis(const IRModule& mod) {
  TaintAnalysis analysis(mod);
  auto ret = analysis.PerformAnalysis();
  return ret;
}

}  // namespace tec
}  // namespace relay
}  // namespace tvm
