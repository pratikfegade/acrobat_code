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
#include "../op/memory/memory.h"
#include "../op/memory/on_device.h"
#include "../transforms/expr_subst.h"
#include "../transforms/function_pointer_analysis.h"
#include "../transforms/map_set.h"
#include "../transforms/pass_utils.h"

namespace tvm {
namespace relay {
namespace tec {

namespace {

using FunctionSet = Map<Function, Bool>;
using ConstantSet = Map<Expr, Bool>;

using ContextT = const Object*;
using TaintT = Array<ConstantSet>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

class FullTaint;
/*!
 * \brief FullTaint tensor type.
 */
class FullTaintNode : public Object {
 public:
  TaintT taint;
  Bool is_constant;
  FunctionSet function_points_to;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("taint", &taint);
    v->Visit("is_constant", &is_constant);
    v->Visit("function_points_to", &function_points_to);
  }

  bool SEqualReduce(const FullTaintNode* other, SEqualReducer equal) const {
    return equal(taint, other->taint) && equal(is_constant, other->is_constant) &&
           equal(function_points_to, other->function_points_to);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(taint);
    hash_reduce(is_constant);
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
  TVM_DLL explicit FullTaint(TaintT taint, Bool is_constant,
                             FunctionSet function_points_to = FunctionSet());

  TVM_DEFINE_OBJECT_REF_METHODS(FullTaint, ObjectRef, FullTaintNode);
};

FullTaint::FullTaint(TaintT taint, Bool is_constant, FunctionSet function_points_to) {
  ObjectPtr<FullTaintNode> n = make_object<FullTaintNode>();
  n->taint = std::move(taint);
  n->is_constant = std::move(is_constant);
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

Array<ConstantSet> ConstructSetArray(size_t size, ConstantSet value) {
  Array<ConstantSet> result;
  for (size_t i = 0; i < size; ++i) {
    result.push_back(value);
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

Array<ConstantSet> CreateStateForType(const Type& type, ConstantSet value) {
  return ConstructSetArray(GetTypeSize(type), value);
}

template <typename T>
bool IsSuperSet(const std::unordered_set<T>& b, const std::unordered_set<T>& a) {
  // return true if all members of a are also in b
  if (a.size() > b.size()) return false;

  auto const not_found = b.end();
  for (auto const& element : a)
    if (b.find(element) == not_found) return false;

  return true;
}

class TransitiveTensorOpCalls : public ExprVisitor {
 public:
  TransitiveTensorOpCalls(const IRModule& mod) : call_graph_(GetPreciseCallGraph(mod)) {}

  bool CanTransitivelyCallTensorOps(const Function& func) {
    return CanTransitivelyCallTensorOps(func.get());
  }

  bool CanTransitivelyCallTensorOps(const FunctionNode* func) {
    auto it = can_transitively_call_tensor_ops_cache_.find(func);
    if (it != can_transitively_call_tensor_ops_cache_.end()) {
      return it->second;
    }
    if (IncludesTensorOpCalls(func)) {
      can_transitively_call_tensor_ops_cache_[func] = true;
      return true;
    }
    auto iit = call_graph_.find(func);
    if (iit != call_graph_.end()) {
      for (auto callee : iit->second) {
        if (CanTransitivelyCallTensorOps(callee)) {
          can_transitively_call_tensor_ops_cache_[func] = true;
          return true;
        }
      }
    }
    can_transitively_call_tensor_ops_cache_[func] = false;
    return false;
  }
  PreciseCallGraph call_graph_;

  bool IncludesTensorOpCalls(const FunctionNode* func) {
    auto it = includes_tensor_op_calls_cache_.find(func);
    if (it != includes_tensor_op_calls_cache_.end()) {
      return it->second;
    }
    // std::cout << "[TTC] Visiting function " << GetFunctionName(func) << std::endl;
    found_tensor_op_calls_ = false;
    this->VisitExpr(func->body);
    includes_tensor_op_calls_cache_[func] = found_tensor_op_calls_;
    // std::cout << "[TTC]  Res " << found_tensor_op_calls_ << std::endl;
    return found_tensor_op_calls_;
  }

  bool IsAuxiliaryOp(const Expr& op) {
    return op == on_device_op_ || op == alloc_storage_op_ || op == alloc_tensor_op_;
  }

  void VisitExpr_(const CallNode* op) {
    ExprVisitor::VisitExpr_(op);
    if (op->op.as<OpNode>() && !IsAuxiliaryOp(op->op) && !IsMarkedScalarOp(op)) {
      // std::cout << "[TTC]   Found op " << op->op << " " << op->op.as<OpNode>() << std::endl;
      found_tensor_op_calls_ = true;
    }
  }

  void VisitExpr_(const FunctionNode* op) {}

  std::string GetFunctionName(const FunctionNode* fn) {
    return fn->GetAttr<String>("db.function_name", String("")).value();
  }

  bool found_tensor_op_calls_{false};
  const Op& invoke_tvm_op_ = GetInvokeTVMOp();
  const Op& on_device_op_ = OnDeviceOp();
  const Op& alloc_tensor_op_ = MemoryAllocTensorOp();
  const Op& alloc_storage_op_ = MemoryAllocStorageOp();
  std::unordered_map<const FunctionNode*, bool> includes_tensor_op_calls_cache_;
  std::unordered_map<const FunctionNode*, bool> can_transitively_call_tensor_ops_cache_;
};

class TaintAnalysis : public BaseExprFunctor {
 public:
  TaintAnalysis(IRModule& mod) : mod_(mod) {}

  std::unordered_map<const FunctionNode*, std::unordered_set<ContextT>> PerformAnalysisPhase1() {
    phase_ = 1;
    RunAnalysis();

    std::unordered_map<const VarNode*, std::unordered_map<ContextT, FullTaint>>
        aggregate_var_states;
    for (auto kv : var_states_) {
      auto ctx = kv.first.first;
      auto var = kv.first.second;
      auto taint = kv.second;
      aggregate_var_states[var][ctx] = taint;
    }

    std::unordered_set<const FunctionNode*> all_functions;
    for (auto kv : function_states_) {
      all_functions.insert(kv.first.second);
    }

    std::unordered_map<const Object*, const FunctionNode*> expr2functions;
    std::unordered_set<const FunctionNode*> global_functions;
    for (auto kv : mod_->functions) {
      if (auto fn = kv.second.as<FunctionNode>()) {
        global_functions.insert(fn);
        PostOrderVisit(fn->body, [&](const Expr& e) {
          if (e.as<CallNode>() || e.as<FunctionNode>()) {
            expr2functions[e.get()] = fn;
          }
        });
      }
    }

    TransitiveTensorOpCalls tensor_call_finder(mod_);
    std::unordered_map<const FunctionNode*, std::unordered_set<ContextT>> results_map;
    for (auto fn_node : all_functions) {
      auto fn = GetRef<Function>(fn_node);
      // std::cout << "[TSR] TrySplit " << GetFunctionName(fn) << std::endl;
      std::unordered_set<ContextT> split_contexts;
      bool no_split = false;
      for (auto param : fn->params) {
        // std::cout << "[TSR]  Param " << param->vid->name_hint << std::endl;
        auto it = aggregate_var_states.find(param.get());
        if (it == aggregate_var_states.end()) {
          // std::cout << "[TSR]   State not found" << std::endl;
          continue;
        }
        auto param_states_with_recursive_context = it->second;

        std::unordered_map<ContextT, FullTaint> param_states;
        for (auto kv : param_states_with_recursive_context) {
          auto context = kv.first;
          // NB: Does not handle mutual recursion
          if (expr2functions.at(context) != fn_node) {
            param_states[kv.first] = kv.second;
          }
        }

        if (param_states.size() <= 1) {
          // std::cout << "[TSR]   Unique contexts: " << param_states.size() << std::endl;
          continue;
        }
        for (size_t j = 0; j < GetTypeSize(GetVarType(param)); ++j) {
          Array<ConstantSet> all_constants;
          for (auto kv : param_states) {
            all_constants.push_back(kv.second->taint[j]);
          }
          ConstantSet merged_constants_w_scalars = MapSet::Merge(all_constants);
          ConstantSet merged_constants;
          // Heuristic: Do not split on scalar constants
          for (auto kv : merged_constants_w_scalars) {
            if (!kv.first->checked_type_.as<TensorTypeNode>()->db_scalar) {
              MapSet::Insert(merged_constants, kv.first);
            }
          }

          // std::cout << "[TSR]   Merged constants:" << std::endl;
          for (auto kv : merged_constants) {
            // std::cout << "[TSR]   " << kv.first->checked_type_ << " "
            // << kv.first.as<ConstantNode>()->data.Shape() << std::endl;
          }
          if (merged_constants.size() <= 1) {
            // std::cout << "[TSR]   Unique constants: " << merged_constants.size() << std::endl;
            continue;
          }
          std::unordered_set<ContextT> param_split_contexts;
          for (auto kv : param_states) {
            param_split_contexts.insert(kv.first);
          }
          if (split_contexts.size() == 0) {
            split_contexts.insert(param_split_contexts.begin(), param_split_contexts.end());
          }
          if (split_contexts.size() == 0 ||
              IsSuperSet(param_split_contexts, param_split_contexts)) {
          } else {
            // std::cout << "[TSR]   No superset: " << merged_constants.size() << std::endl;
            no_split = true;
            break;
          }
        }
      }
      if (split_contexts.size() <= 1) {
        // std::cout << "[TSR]   Too few contexts: " << split_contexts.size() << std::endl;
        continue;
      }

      // Check that each of the calls to be repeated have only one
      // callee. NB: This assumes that the contexts are callsites
      // TODO: Also check somehow that the callee is either a global
      // function, or a local function defined in immediately before
      // the callsite.
      for (auto cn : split_contexts) {
        auto it = callees_.find(static_cast<const CallNode*>(cn));
        if (it == callees_.end()) {
          // std::cout << "[TSR]   No callees: " << split_contexts.size() << std::endl;
          no_split = true;
          continue;
        }
        auto callees = it->second;
        if (callees.size() != 1 || *(callees.begin()) != fn_node) {
          // std::cout << "[TSR]   Too many callees : " << split_contexts.size() << std::endl;
          no_split = true;
          continue;
        }
        if (!global_functions.count(fn_node) &&
            expr2functions.at(cn) != expr2functions.at(fn_node)) {
          // std::cout << "[TSR]   Non-local callee: " << split_contexts.size() << std::endl;
          no_split = true;
          continue;
        }
      }

      if (!no_split) {
        if (!tensor_call_finder.CanTransitivelyCallTensorOps(fn)) {
          // std::cout << "[TSR]  Does not call tensor op " << GetFunctionName(fn) << std::endl;
          continue;
        }

        results_map[fn_node] = split_contexts;
        // std::cout << "[TSR] Splitting " << GetFunctionName(fn) << std::endl;
        for (auto ctx : split_contexts) {
          // std::cout << "[TSR]  Context: " << ctx << " "
          // << PrettyPrint(GetRef<Expr>(static_cast<const CallNode*>(ctx))) << std::endl;
        }
      }
    }

    return results_map;
  }

  IRModule PerformAnalysisPhase2() {
    phase_ = 2;
    RunAnalysis();

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

    bool print = false;
    Map<Function, Array<Bool>> results_map;
    for (auto kv : merged_function_states) {
      auto fn = GetRef<Function>(kv.first);
      if (print) {
        std::cout << "[MPT] Function " << fn->GetAttr<String>("db.function_name") << " " << kv.first
                  << std::endl;
      }
      Array<Bool> param_states;
      for (auto arg : fn->params) {
        auto it = merged_var_states.find(arg.get());
        if (it == merged_var_states.end()) {
          merged_var_states[arg.get()] = FullTaint(
              CreateStateForType(arg->checked_type(), ConstantSet()), Bool(false), FunctionSet());
        }
        auto arg_state = merged_var_states[arg.get()];

        ICHECK_EQ(arg_state->taint.size(), GetTypeSize(arg->checked_type())) << arg->checked_type();

        for (auto s : arg_state->taint) {
          param_states.push_back(Bool((s.size() == 1) && arg_state->is_constant->value));
          if (print) {
            std::cout << "[MPT]  P " << arg_state->is_constant->value << " " << s.size() << " "
                      << MapSet::ToString(s) << std::endl;
          }
        }
      }
      auto iit = merged_function_states.find(fn.get());
      if (iit == merged_function_states.end()) {
        merged_function_states[fn.get()] =
            FullTaint(CreateStateForType(fn->ret_type, ConstantSet()), Bool(false), FunctionSet());
      }
      auto fn_state = merged_function_states[fn.get()];
      for (auto s : fn_state->taint) {
        param_states.push_back(Bool((s.size() == 1) && fn_state->is_constant->value));
        if (print) {
          std::cout << "[MPT]  O " << s.size() << std::endl;
        }
      }

      results_map.Set(fn, param_states);
    }

    return AddFunctionTaints(results_map, mod_, tir::attr::kDBModelParamterTaints);

    return mod_;
  }

 private:
  void RunAnalysis() {
    auto main_func = Downcast<Function>(mod_->Lookup("main"));

    ICHECK_EQ(main_func->type_params.size(), 0);
    for (auto param : main_func->params) {
      ICHECK(!param->checked_type().as<FuncTypeNode>());
      Add(param, GetInitialContext(),
          FullTaint(CreateStateForType(param->checked_type(), ConstantSet()), Bool(false),
                    FunctionSet()),
          "Main param");
    }

    for (size_t i = 0; i < max_iterations_; ++i) {
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
  }

  std::string GetFunctionName(const Function& fn) {
    return fn->GetAttr<String>("db.function_name", String("")).value();
  }

  void Reset() {
    on_stack_.clear();
    environment_stack_.clear();
    type_environment_stack_.clear();
  }

  Array<ConstantSet> MergeTaints(const Array<ConstantSet>& vals1, const Array<ConstantSet>& vals2) {
    ICHECK_EQ(vals1.size(), vals2.size());
    Array<ConstantSet> result;
    for (size_t i = 0; i < vals1.size(); ++i) {
      result.push_back(MapSet::Merge<Expr>(vals1[i], vals2[i]));
    }
    return result;
  }

  std::string TaintToString(const Array<ConstantSet>& taint) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < taint.size(); ++i) {
      ss << MapSet::ToString(taint[i]);
      if (i < taint.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    return ss.str();
  }

  FunctionSet MergeFunctionSets(const FunctionSet& set1, const FunctionSet& set2) {
    return MapSet::Merge<Function>(set1, set2);
  }

  FullTaint Merge(const Array<FullTaint>& full_taints) {
    TaintT taint = ConstructSetArray(full_taints[0]->taint.size(), ConstantSet());
    bool is_constant = true;
    FunctionSet function_points_to;
    for (auto& full_taint : full_taints) {
      auto this_taint = full_taint->taint;
      auto this_is_constant = full_taint->is_constant;
      auto this_function_points_to = full_taint->function_points_to;
      taint = MergeTaints(taint, this_taint);
      is_constant = is_constant && this_is_constant.operator bool();
      function_points_to = MergeFunctionSets(function_points_to, this_function_points_to);
    }
    return FullTaint(taint, Bool(is_constant), function_points_to);
  }

  ConstantSet CollapseTaint(const Array<ConstantSet>& vals) { return MapSet::Merge(vals); }

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
      auto taint = CreateStateForType(fn->ret_type, ConstantSet());
      auto function_points_to = FunctionSet();
      auto full_taint = FullTaint(taint, Bool(true), function_points_to);
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
    return FullTaint(ConstructSetArray(to_size, CollapseTaint(taint->taint)), taint->is_constant,
                     taint->function_points_to);
  }

  FullTaint VisitExpr_(const ConstantNode* op, ContextT current_context) {
    auto taint = Array<ConstantSet>({MapSet::Create({GetRef<Expr>(op)})});
    auto function_points_to = FunctionSet();
    return FullTaint(taint, Bool(true), function_points_to);
  }

  FullTaint VisitExpr_(const TupleNode* op, ContextT current_context) {
    TaintT taint;
    bool is_constant = true;
    FunctionSet function_points_to;
    for (auto field : op->fields) {
      auto field_full_taint = this->VisitExpr(field, current_context);
      taint.push_back(CollapseTaint(field_full_taint->taint));
      is_constant = is_constant && field_full_taint->is_constant;
      function_points_to =
          MergeFunctionSets(function_points_to, field_full_taint->function_points_to);
    }
    return FullTaint(taint, Bool(is_constant), function_points_to);
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
      auto taint = FullTaint(CreateStateForType(op->checked_type(), ConstantSet()), Bool(true),
                             FunctionSet());
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
    auto taint = Array<ConstantSet>({ConstantSet()});
    auto function_points_to = FunctionSet();

    auto base_func = mod_->Lookup(GetRef<GlobalVar>(op));
    if (base_func.as<FunctionNode>()) {
      function_points_to.Set(Downcast<Function>(base_func), Bool(true));
    }
    return FullTaint(taint, Bool(false), function_points_to);
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
    MapSet::Insert(function_points_to, function);
    TaintT taint = ConstructSetArray(1, ConstantSet());
    return FullTaint(taint, Bool(false), function_points_to);
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
      auto taint = CreateStateForType(op->checked_type(), ConstantSet());
      auto function_points_to = FunctionSet();
      return FullTaint(taint, Bool(false), function_points_to);
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
      bool print =
          false;  //(phase_ == 2) && (callee_name == "rnn_f0_dup1" || callee_name == "rnn_dup0");
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

        auto add_res = Add(param, callee_context, param_full_taint, "function param");

        //////////////////////////////
        if (print) {
          std::string arg_str = "";
          if (auto vn = arg.as<VarNode>()) {
            arg_str = vn->vid->name_hint;
          }
          std::cout << "[MPT]  Param " << current_context << " " << arg_str << " "
                    << param->vid->name_hint << " " << TaintToString(param_full_taint->taint) << " "
                    << TaintToString(add_res->taint) << std::endl;
        }
        //////////////////////////////
      }

      FullTaint callee_return_full_taint;
      if (on_stack_.count(std::make_pair(callee_context, callee_fn.get()))) {
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
    return this->VisitExpr(op->body, current_context);
  }

  FullTaint VisitExpr_(const IfNode* op, ContextT current_context) {
    auto cond_taint = this->VisitExpr(op->cond, current_context);
    auto true_taint = this->VisitExpr(op->true_branch, current_context);
    auto false_taint = this->VisitExpr(op->false_branch, current_context);
    return Merge(Array<FullTaint>({true_taint, false_taint}));
  }

  FullTaint VisitExpr_(const OpNode* op, ContextT current_context) {
    auto taint = Array<ConstantSet>({ConstantSet()});
    auto function_points_to = FunctionSet();
    return FullTaint(taint, Bool(false), function_points_to);
  }

  FullTaint VisitExpr_(const TupleGetItemNode* op, ContextT current_context) {
    auto tuple_taint = this->VisitExpr(op->tuple, current_context);
    ICHECK_GT(tuple_taint->taint.size(), op->index);
    auto taint = CreateStateForType(op->checked_type(), tuple_taint->taint[op->index]);
    auto function_points_to = tuple_taint->function_points_to;
    return FullTaint(taint, tuple_taint->is_constant, function_points_to);
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
    auto taint = Array<ConstantSet>({ConstantSet()});
    auto function_points_to = FunctionSet();
    return FullTaint(taint, Bool(false), function_points_to);
  }

  FullTaint VisitExpr_(const ConstructorNode* op, ContextT current_context) {
    auto taint = Array<ConstantSet>({ConstantSet()});
    auto function_points_to = FunctionSet();
    return FullTaint(taint, Bool(false), function_points_to);
  }

  FullTaint VisitExpr_(const MatchNode* op, ContextT current_context) {
    // ICHECK(!op->data->checked_type().as<TupleTypeNode>());
    auto data_full_taint = this->VisitExpr(op->data, current_context);
    auto collapsed_data_taint = CollapseTaint(data_full_taint->taint);

    Array<FullTaint> clause_taints;
    for (auto clause : op->clauses) {
      auto pattern_vars = CollectPatternVars(clause->lhs);
      for (auto& var : pattern_vars) {
        Add(var, current_context,
            FullTaint(CreateStateForType(var->checked_type(), collapsed_data_taint),
                      data_full_taint->is_constant, data_full_taint->function_points_to),
            "Match pattern");
      }
      clause_taints.push_back(this->VisitExpr(clause->rhs, current_context));
    }

    return Merge(clause_taints);
  }

  IRModule& mod_;

  VarStateMap var_states_;
  FunctionStateMap function_states_;
  FunctionEnvironmentMap function_environments_;
  std::unordered_map<const FunctionNode*, VarSet> free_vars_map_;
  OnStackSet on_stack_;
  std::vector<Map<Var, FullTaint>> environment_stack_;
  std::vector<Map<TypeVar, Type>> type_environment_stack_;

  std::vector<Function> function_stack_;
  std::unordered_map<const CallNode*, std::unordered_set<const FunctionNode*>> callees_;
  int max_iterations_{10};
  int phase_ = 1;
};

class AbstractDuplicator : public ExprMutator {
 protected:
  Expr VisitExpr_(const VarNode* op) override {
    auto var = GetRef<Var>(op);
    auto it = var_rmap_.find(var);
    if (it != var_rmap_.end()) {
      return (*it).second;
    } else {
      auto new_var = Var(var->vid->name_hint + "_d", var->type_annotation, var->span);
      new_var->checked_type_ = var->checked_type();
      var_rmap_.Set(var, new_var);
      return new_var;
    }
  }

  Pattern VisitPattern(const Pattern& p) override {
    if (p.as<PatternWildcardNode>()) {
      return p;
    } else if (auto pv = p.as<PatternVarNode>()) {
      return PatternVar(Downcast<Var>(this->VisitExpr(pv->var)));
    } else if (auto pc = p.as<PatternConstructorNode>()) {
      Array<Pattern> patterns;
      for (auto field : pc->patterns) {
        patterns.push_back(this->VisitPattern(field));
      }
      return PatternConstructor(pc->constructor, patterns);
    } else if (auto pt = p.as<PatternTupleNode>()) {
      Array<Pattern> patterns;
      for (auto field : pt->patterns) {
        patterns.push_back(this->VisitPattern(field));
      }
      return PatternTuple(patterns);
    }
    ICHECK(false) << p;
    return p;
  }

  Map<Var, Var> var_rmap_;
};

class RepeatMutator : public AbstractDuplicator {
 public:
  RepeatMutator(std::unordered_map<const FunctionNode*, std::unordered_set<ContextT>> to_repeat,
                IRModule mod)
      : to_repeat_(to_repeat), mod_(mod) {
    for (auto kv : to_repeat_) {
      for (auto ctx : kv.second) {
        to_repeat_call_nodes_[static_cast<const CallNode*>(ctx)] = GetRef<Function>(kv.first);
      }
    }
  }

  IRModule Repeat() {
    if (to_repeat_call_nodes_.size() > 0) {
      for (auto kv : mod_->functions) {
        auto base_func = kv.second;
        if (base_func.as<FunctionNode>()) {
          auto func = Downcast<Function>(this->Mutate(Downcast<Function>(base_func)));
          new_global_functions_.Set(kv.first, func);
          // std::cout << "[MPTA] New function1 " << kv.first->name_hint << " " << kv.first.get()
          // << std::endl;
        }
      }
      for (auto kv : new_global_functions_) {
        // std::cout << "[MPTA] New function2 " << kv.first->name_hint << " " << kv.first.get()
        // << std::endl;
        mod_->Add(kv.first, kv.second, true);
      }
    }
    return mod_;
  }

 private:
  Expr VisitExpr_(const CallNode* old_op) override {
    auto mutated = ExprMutator::VisitExpr_(old_op);
    auto op = mutated.as<CallNode>();
    ICHECK(op);
    auto it = to_repeat_call_nodes_.find(old_op);
    if (it != to_repeat_call_nodes_.end()) {
      auto callee = it->second;
      if (op->op.as<GlobalVarNode>()) {
        ICHECK_EQ(mod_->Lookup(Downcast<GlobalVar>(op->op)), callee);
        mutated = Call(DuplicateGlobalVar(Downcast<GlobalVar>(op->op)), op->args, op->attrs,
                       op->type_args, op->span);
      } else {
        mutated =
            Call(DuplicateFunction(callee, "", NullValue<GlobalVar>(), NullValue<GlobalVar>()),
                 op->args, op->attrs, op->type_args, op->span);
      }
    }
    mutated->checked_type_ = old_op->checked_type_;
    return mutated;
  }

  Expr DuplicateGlobalVar(const GlobalVar& gv) {
    auto fn = Downcast<Function>(mod_->Lookup(gv));
    std::string new_name = gv->name_hint + "_dup" + std::to_string(ctr_++);
    GlobalVar new_gv = GlobalVar(new_name, gv->checked_type_);
    new_gv->checked_type_ = gv->checked_type_;
    new_global_functions_.Set(new_gv, DuplicateFunction(fn, new_name, gv, new_gv));
    return new_gv;
  }

  // NB: Does not handle mutually recursive functions
  Function DuplicateFunction(const Function& fn, std::string new_name, const GlobalVar& old_gv,
                             const GlobalVar& new_gv) {
    class DuplicateExprSubstituter : public ExprMutator {
     public:
      explicit DuplicateExprSubstituter(
          std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> subst_map, int& ctr)
          : subst_map_(subst_map), ctr_(ctr) {}

      Expr VisitExpr(const Expr& expr) final {
        auto it = subst_map_.find(expr);
        if (it != subst_map_.end()) {
          return ExprMutator::VisitExpr((*it).second);
        }
        return ExprMutator::VisitExpr(expr);
      }

      Expr VisitExpr_(const FunctionNode* op) final {
        // std::cout << "[PANSL] Duplicate visiting " << GetRef<Expr>(op) << std::endl;

        auto visited = ExprMutator::VisitExpr_(op);
        // std::cout << "[PANSL]  visited " << visited << std::endl;
        auto fn = visited.as<FunctionNode>();

        ICHECK(fn);
        auto new_func =
            Function(fn->params, fn->body, fn->ret_type, fn->type_params, fn->attrs, fn->span);

        auto opt_old_name = fn->GetAttr<String>(tir::attr::kDBFunctionName);
        if (opt_old_name) {
          new_func = WithAttr(new_func, tir::attr::kDBFunctionName,
                              opt_old_name.value() + "_dup" + std::to_string(ctr_++));
        }
        new_func = WithAttr(new_func, "db.mpta.original_function", visited);
        new_func->checked_type_ = op->checked_type();
        return new_func;
      }

     private:
      tvm::Map<Expr, Expr> subst_map_;
      int& ctr_;
    };

    if (new_name.size() == 0) {
      auto old_name = fn->GetAttr<String>("db.function_name").value();
      new_name = old_name + "_dup" + std::to_string(ctr_++);
    }
    // std::cout << "[PANSL] Duplicating " << new_name << std::endl;
    Array<Var> new_params;
    std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> rmap;
    if (old_gv.defined()) {
      rmap[old_gv] = new_gv;
    }
    for (auto param : fn->params) {
      auto new_param = Var(param->vid->name_hint + "_d", param->type_annotation, param->span);
      new_param->checked_type_ = param->checked_type();
      new_params.push_back(new_param);
      rmap[param] = new_param;
    }

    auto new_body = DuplicateExprSubstituter(std::move(rmap), ctr_).Mutate(fn->body);
    auto new_func =
        WithAttr(Function(new_params, new_body, fn->ret_type, fn->type_params, fn->attrs, fn->span),
                 tir::attr::kDBFunctionName, String(new_name));
    new_func = WithAttr(new_func, "db.mpta.original_function", fn);
    new_func = Downcast<Function>(AbstractDuplicator()(new_func));
    new_func->checked_type_ = fn->checked_type();
    return new_func;
  }

  Map<GlobalVar, Function> new_global_functions_;
  std::unordered_map<const FunctionNode*, std::unordered_set<ContextT>> to_repeat_;
  IRModule mod_;
  std::unordered_map<const CallNode*, Function> to_repeat_call_nodes_;
  int ctr_{0};
};
}  // namespace

IRModule ModelParameterTaintAnalysis(IRModule& mod, bool repeat) {
  if (repeat) {
    for (int i = 0; i < 2; ++i) {
      auto to_repeat = TaintAnalysis(mod).PerformAnalysisPhase1();
      mod = RepeatMutator(to_repeat, mod).Repeat();
    }
    // std::cout << "[After Repeating]\n" << mod << std::endl;
  }
  return TaintAnalysis(mod).PerformAnalysisPhase2();
}

}  // namespace tec
}  // namespace relay
}  // namespace tvm
