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
inline std::vector<bool> ConstructBoolArray(size_t size, bool value) {
  std::vector<bool> result(size, value);
  return std::move(result);
}

std::pair<Array<Bool>, bool> Merge(const Array<Bool>& vals1, const Array<Bool>& vals2) {
  bool changed = false;
  ICHECK_EQ(vals1.size(), vals2.size());
  Array<Bool> result;
  for (size_t i = 0; i < vals1.size(); ++i) {
    Bool new_val = vals1[i] && vals2[i];
    if (new_val.operator bool() != vals1[i].operator bool()) {
      changed = true;
    }
    result.push_back(new_val);
  }
  return std::make_pair(result, changed);
}

Bool AndState(const Array<Bool>& vals) {
  Bool result(true);
  for (auto v : vals) {
    result = result && v;
  }
  return result;
}

// Array<Bool> CreateStateForType(const Type& type, bool value) {
//   size_t state_size = 1;
//   if (auto ttn = type.as<TupleTypeNode>()) {
//     state_size = ttn->fields.size();
//   }
//   return ConstructBoolArray(state_size, value);
// }

// class ModelParameterTaintVisitor : public ExprFunctor<Array<Bool>(const Expr& n)> {
//  public:
//   ModelParameterTaintVisitor(const IRModule& mod, Map<Var, Array<Bool>>* p_var_state,
//                              Map<Function, Array<Bool>>* p_function_state)
//       : mod_(mod), p_var_state_(p_var_state), p_function_state_(p_function_state) {}

//   void Run(const Function& func, std::string name) {
//     print_ = true;  //(name == "rnn_cell");
//     if (print_) {
//       std::cout << "[MPTA]  Func " << name << std::endl;
//     }
//     this->VisitExpr(func);
//   }

//   template <typename T>
//   Array<Bool> MergeAndSet(Map<T, Array<Bool>>* p_map, T key, Array<Bool> vals2) {
//     auto it = p_map->find(key);
//     if (it == p_map->end()) {
//       changed |= true;
//       p_map->Set(key, vals2);
//       return vals2;
//     } else {
//       Array<Bool> vals1 = (*it).second;
//       auto res = Merge(vals1, vals2);
//       changed |= res.second;
//       p_map->Set(key, res.first);
//       return res.first;
//     }
//   }

//   Array<Bool> MergeAndSet(Var var, Array<Bool> vals2) {
//     auto ret = MergeAndSet<Var>(p_var_state_, var, vals2);
//     std::cout << "[MPTA]   Var " << var->vid->name_hint << " " << vals2 << " " << ret <<
//     std::endl; return ret;
//   }

//   Array<Bool> MergeAndSet(Function func, Array<Bool> vals2) {
//     auto ret = MergeAndSet<Function>(p_function_state_, func, vals2);
//     ICHECK(p_function_state_->count(func));
//     return ret;
//   }

//   Array<Bool> VisitExpr_(const ConstantNode* op) { return Array<Bool>({Bool(true)}); }

//   Array<Bool> VisitExpr_(const TupleNode* op) {
//     Array<Bool> result;
//     for (auto field : op->fields) {
//       Array<Bool> field_state = VisitExpr(field);
//       result.push_back(AndState(field_state));
//     }
//     return result;
//   }

//   Array<Bool> VisitExpr_(const VarNode* op) {
//     auto var = GetRef<Var>(op);
//     // std::cout << "[MPTA]    Visiting var " << var->vid->name_hint << std::endl;
//     auto it = p_var_state_->find(var);
//     if (it == p_var_state_->end()) {
//       Array<Bool> state = CreateStateForType(op->checked_type(), true);
//       changed |= true;
//       p_var_state_->Set(var, state);
//       // std::cout << "[MPTA]     NoFound " << state << std::endl;
//       return state;
//     } else {
//       // std::cout << "[MPTA]     Found " << (*it).second << std::endl;
//       return (*it).second;
//     }
//   };

//   Array<Bool> VisitExpr_(const GlobalVarNode* op) { return Array<Bool>({Bool(false)}); };

//   Array<Bool> VisitExpr_(const FunctionNode* op) {
//     if (print_) {
//       // std::cout << "[FUNC] Visiting function " << op << " " << ObjectHash()(GetRef<Expr>(op))
//       // << std::endl;
//     }
//     auto func = GetRef<Function>(op);
//     if (!visited_functions_.count(func)) {
//       visited_functions_.Set(func, Bool(true));
//       if (print_) {
//         // std::cout << "[FUNC]  Visiting first" << std::endl;
//       }
//       bool all_params_model_params = true;
//       for (auto p : op->params) {
//         auto it = p_var_state_->find(p);
//         if (it == p_var_state_->end() || AndState((*it).second).operator bool()) continue;
//         all_params_model_params = false;
//         break;
//       }
//       if (all_params_model_params) {
//         if (print_) {
//           // std::cout << "[FUNC]   Returning 1 for function " << ret << " "
//           // << p_function_state_->count(func) << std::endl;
//         }
//         return MergeAndSet(func, CreateStateForType(op->ret_type, true));
//       } else {
//         auto ret = MergeAndSet(func, VisitExpr(op->body));
//         if (print_) {
//           // std::cout << "[FUNC]   Returning 2 for function " << ret << " "
//           // << p_function_state_->count(func) << std::endl;
//         }
//         return ret;
//       }
//     } else {
//       auto it = p_function_state_->find(func);
//       if (it == p_function_state_->end()) {
//         // Recursive function visiting twice
//         return CreateStateForType(op->ret_type, true);
//       } else {
//         return (*it).second;
//       }
//     }
//   }

//   Array<Bool> VisitExpr_(const CallNode* op) {
//     auto on_device_props = GetOnDeviceProps(op);
//     if (on_device_props.body.defined()) {
//       return VisitExpr(on_device_props.body);
//     }
//     if (print_) {
//       // std::cout << "[CALL] Call  " << op->op << std::endl;
//     }
//     bool print = false;
//     auto get_function = [&](const Expr& expr) {
//       const FunctionNode* callee_func = nullptr;
//       if (expr.as<GlobalVarNode>()) {
//         // print = true;
//         auto func = this->mod_->Lookup(Downcast<GlobalVar>(expr));
//         if (auto fn = func.as<FunctionNode>()) {
//           callee_func = fn;
//         }
//       } else if (auto fn = expr.as<FunctionNode>()) {
//         callee_func = fn;
//       }
//       return callee_func;
//     };

//     const FunctionNode* callee_func = nullptr;
//     if (auto vn = op->op.as<VarNode>()) {
//       auto it = let_shortcuts_.find(Downcast<Var>(op->op));
//       if (it != let_shortcuts_.end()) {
//         callee_func = get_function(it->second);
//       }
//     }
//     if (callee_func == nullptr) {
//       callee_func = get_function(op->op);
//     }

//     if (callee_func == nullptr) {
//       return CreateStateForType(op->checked_type(), false);
//     }

//     if (print) {
//       std::cout << "[CALL] Call to callee" << std::endl;
//     }

//     for (size_t i = 0; i < op->args.size(); ++i) {
//       Array<Bool> param_state = VisitExpr(op->args[i]);
//       auto res = MergeAndSet(callee_func->params[i], param_state);
//       if (print) {
//         std::cout << "[CALL]   Setting callee arg " << callee_func->params[i]->vid->name_hint <<
//         ""
//                   << param_state << " " << res << std::endl;
//       }
//     }

//     // if (print) {
//     // std::cout << "[CALL]  Visiting function" << std::endl;
//     // std::cout << GetRef<Function>(callee_func) << std::endl;
//     // }
//     auto res = VisitExpr(GetRef<Function>(callee_func));
//     if (print) {
//       std::cout << "[CALL]  Call res " << res << std::endl;
//     }
//     return res;
//   }

//   Array<Bool> VisitExpr_(const LetNode* op) {
//     // std::cout << "[MPTA] Let " << op->var << std::endl;
//     if (op->value.as<GlobalVarNode>() || op->value.as<VarNode>() || op->value.as<FunctionNode>())
//     {
//       let_shortcuts_.insert({op->var, op->value});
//     }
//     Array<Bool> value_taints = VisitExpr(op->value);
//     MergeAndSet(op->var, value_taints);
//     return VisitExpr(op->body);
//   }

//   Array<Bool> VisitExpr_(const IfNode* op) {
//     VisitExpr(op->cond);
//     return Merge(VisitExpr(op->true_branch), VisitExpr(op->false_branch)).first;
//   }

//   Array<Bool> VisitExpr_(const OpNode* op) { return Array<Bool>({Bool(false)}); }

//   Array<Bool> VisitExpr_(const TupleGetItemNode* op) {
//     Array<Bool> tuple_state = VisitExpr(op->tuple);
//     ICHECK_GE(tuple_state.size(), op->index);
//     return CreateStateForType(op->checked_type(), tuple_state[op->index].operator bool());
//   }

//   Array<Bool> VisitExpr_(const RefCreateNode* op) { return Array<Bool>({Bool(false)}); }

//   Array<Bool> VisitExpr_(const RefReadNode* op) { return Array<Bool>({Bool(false)}); }

//   Array<Bool> VisitExpr_(const RefWriteNode* op) { return Array<Bool>({Bool(false)}); }

//   Array<Bool> VisitExpr_(const ConstructorNode* op) { return Array<Bool>({Bool(false)}); }

//   Array<Bool> VisitExpr_(const MatchNode* op) {
//     VisitExpr(op->data);
//     Array<Bool> res = CreateStateForType(op->checked_type(), true);
//     for (auto clause : op->clauses) {
//       res = Merge(res, VisitExpr(clause->rhs)).first;
//     }
//     return res;
//   }

//   bool isConstantPrimFunc(const tir::PrimFunc& func) { return (func->params.size() == 0); }

//   bool changed{false};

//  private:
//   const IRModule& mod_;
//   Map<Var, Array<Bool>>* p_var_state_;
//   Map<Function, Array<Bool>>* p_function_state_;
//   std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> let_shortcuts_;
//   Map<Function, Bool> visited_functions_;
//   Map<tir::PrimFunc, Bool> constant_prim_funcs_;
//   bool print_;
// };

// class ResultGatherer : public ExprVisitor {
//  public:
//   ResultGatherer(const Map<Var, Array<Bool>>& var_states,
//                  const Map<Function, Array<Bool>>& function_states)
//       : var_states_(var_states), function_states_(function_states) {}

//   void Flatten(Array<Bool>* p_param_states, const Var& var) {
//     size_t state_size = 1;
//     if (auto ttn = var->checked_type().as<TupleTypeNode>()) {
//       state_size = ttn->fields.size();
//     }
//     auto it = var_states_.find(var);
//     Array<Bool> state =
//         (it != var_states_.end()) ? (*it).second : ConstructBoolArray(state_size, false);
//     ICHECK_EQ(state.size(), state_size);
//     p_param_states->push_back_all(state);
//   }

//   size_t FlattenTupleType(const Type& type) {
//     if (auto ttn = type.as<TupleTypeNode>()) {
//       size_t sum = 0;
//       for (auto field : ttn->fields) {
//         sum += FlattenTupleType(field);
//       }
//       return sum;
//     } else {
//       return 1;
//     }
//   }

//   void VisitExpr_(const FunctionNode* op) {
//     auto hash = op->GetAttr<String>("hash", "no_hash");
//     Array<Bool> param_states;
//     for (auto param : op->params) {
//       Flatten(&param_states, param);
//     }

//     auto it = function_states_.find(GetRef<Function>(op));
//     if (it == function_states_.end()) {
//       size_t flattened_size = FlattenTupleType(op->ret_type);
//       param_states.push_back_all(ConstructBoolArray(flattened_size, false));
//     } else {
//       auto output_state = (*it).second;
//       if (auto ttn = op->ret_type.as<TupleTypeNode>()) {
//         ICHECK_EQ(output_state.size(), ttn->fields.size());
//         for (size_t i = 0; i < ttn->fields.size(); ++i) {
//           size_t flattened_size = FlattenTupleType(ttn->fields[i]);
//           param_states.push_back_all(
//               ConstructBoolArray(flattened_size, output_state[i].operator bool()));
//         }
//       } else {
//         param_states.push_back_all(output_state);
//       }
//     }

//     result.Set(GetRef<Function>(op), param_states);
//     ExprVisitor::VisitExpr_(op);
//   }

//   const Map<Var, Array<Bool>>& var_states_;
//   const Map<Function, Array<Bool>>& function_states_;
//   Map<Function, Array<Bool>> result;
// };

using MPTAVarKey = std::pair<const FunctionNode*, const VarNode*>;
using MPTAFunctionKey = std::pair<const FunctionNode*, const FunctionNode*>;
using MPTAOpKey = std::pair<const FunctionNode*, const CallNode*>;

using MPTABaseExprFunctor = ExprFunctor<std::vector<bool>(const Expr& n)>;
using MPTAVarStateMap = std::unordered_map<MPTAVarKey, std::vector<bool>, PairHash, PairEquals>;
using MPTAInvokeTVMOpDepthMap =
    std::unordered_map<MPTAOpKey, std::vector<bool>, PairHash, PairEquals>;
using MPTAFunctionStateMap =
    std::unordered_map<MPTAFunctionKey, std::vector<bool>, PairHash, PairEquals>;

std::vector<bool> CreateStateForType(const Type& type, bool value) {
  size_t state_size = 1;
  if (auto ttn = type.as<TupleTypeNode>()) {
    state_size = ttn->fields.size();
  }
  return ConstructBoolArray(state_size, value);
}

class MPTAnalysis : public MPTABaseExprFunctor {
 public:
  MPTAnalysis(const IRModule& mod, const FunctionSet& recursive_functions,
              const CalleesMap& callees_map)
      : mod_(mod), recursive_functions_(recursive_functions), callees_map_(callees_map) {
    for (auto kv : mod_->functions) {
      func_name_map_[kv.second.get()] = kv.first->name_hint;
      // std::cout << "[FUNC_NAME] " << kv.second.get() << " " << kv.first->name_hint << std::endl;
    }
    func_name_map_[nullptr] = "START";
    stack_.push_back(nullptr);

    if (mod->ContainGlobalVar("map")) {
      map_fun_node_ = mod->Lookup("map").as<FunctionNode>();
    }
  }

  Map<Function, Array<Bool>> PerformAnalysis() {
    auto main_func = Downcast<Function>(mod_->Lookup("main"));

    for (auto arg : main_func->params) {
      var_states_[std::make_pair(nullptr, arg.get())] =
          CreateStateForType(arg->checked_type(), false);
    }

    auto call_graph = CallGraph(mod_);
    size_t i = 0;
    for (; i < max_iterations_; ++i) {
      std::cout << "[MPTA] ITERATION " << i << std::endl;
      visited_functions_.clear();
      changed_ = false;
      auto entry_func = Downcast<Function>(mod_->Lookup("main"));
      VisitBody(entry_func);
      if (!changed_) {
        break;
      }
    }

    std::unordered_map<const VarNode*, std::vector<bool>> merged_var_states;
    for (auto kv : var_states_) {
      std::cout << "[MPTA] MergeVar: " << kv.first.second->vid->name_hint << std::endl;
      auto it = merged_var_states.find(kv.first.second);
      if (it != merged_var_states.end()) {
        merged_var_states[kv.first.second] =
            Merge(merged_var_states[kv.first.second], kv.second).first;
      } else {
        merged_var_states[kv.first.second] = kv.second;
      }
    }

    std::unordered_map<const FunctionNode*, std::vector<bool>> merged_function_states;
    for (auto kv : function_states_) {
      auto it = merged_function_states.find(kv.first.second);
      if (it != merged_function_states.end()) {
        merged_function_states[kv.first.second] =
            Merge(merged_function_states[kv.first.second], kv.second).first;
      } else {
        merged_function_states[kv.first.second] = kv.second;
      }
    }

    if (true) {
      for (auto kv : merged_var_states) {
        std::cout << "[MPTA] MergeVar: " << kv.first->vid->name_hint << std::endl;
        std::cout << "[MPTA]  Var Depths: " << kv.first->vid->name_hint << " "
                  << support::PrintVector(kv.second) << std::endl;
      }

      for (auto kv : merged_function_states) {
        std::cout << "[MPTA]  Function Depths: " << func_name_map_[kv.first] << " "
                  << support::PrintVector(kv.second) << std::endl;
      }
    }

    Map<Function, Array<Bool>> results_map;
    for (auto kv : merged_function_states) {
      auto fn = GetRef<Function>(kv.first);
      Array<Bool> param_states;
      std::vector<std::string> param_names;
      for (auto arg : fn->params) {
        auto it = merged_var_states.find(arg.get());
        if (it == merged_var_states.end()) {
          merged_var_states[arg.get()] = CreateStateForType(arg->checked_type(), false);
        }
        auto arg_state = merged_var_states[arg.get()];

        for (auto s : arg_state) {
          param_states.push_back(Bool(s));
        }
        param_names.push_back(arg->vid->name_hint);
      }
      auto iit = merged_function_states.find(fn.get());
      if (iit == merged_function_states.end()) {
        merged_function_states[fn.get()] = CreateStateForType(fn->ret_type, false);
      }
      auto fn_state = merged_function_states[fn.get()];
      for (auto s : fn_state) {
        param_states.push_back(Bool(s));
      }

      std::cout << "[MPTA] Function " << func_name_map_[fn.get()] << " " << fn.get() << " "
                << support::PrintVector(param_names) << " " << param_states << std::endl;
      results_map.Set(fn, param_states);
    }

    return results_map;
  }

 private:
  const FunctionNode* GetCurrentContext() { return stack_[stack_.size() - 2]; }

  const FunctionNode* GetCurrentFunction() { return stack_.back(); }

  std::pair<std::vector<bool>, bool> Merge(const std::vector<bool>& vals1,
                                           const std::vector<bool>& vals2) {
    bool changed = false;
    ICHECK_EQ(vals1.size(), vals2.size());
    std::vector<bool> result;
    for (size_t i = 0; i < vals1.size(); ++i) {
      bool new_val = vals1[i] && vals2[i];
      if (new_val != vals1[i]) {
        changed = true;
      }
      result.push_back(new_val);
    }
    return std::make_pair(result, changed);
  }

  bool Collapse(const std::vector<bool>& vals) {
    bool res = true;
    for (auto b : vals) {
      res = res && b;
    }
    return res;
  }

  template <typename T, typename MapType>
  std::vector<bool> Add(MapType& map, const T& key, const std::vector<bool>& to_add) {
    auto it = map.find(key);
    if (it == map.end()) {
      map[key] = to_add;
      changed_ = true;
    } else {
      auto res = Merge(map[key], to_add);
      map[key] = res.first;
      changed_ |= res.second;
    }
    return map.at(key);
  }

  std::vector<bool> Add(const MPTAVarKey& var, const std::vector<bool>& to_add,
                        const std::string& reason) {
    auto res = Add<MPTAVarKey, MPTAVarStateMap>(var_states_, var, to_add);
    std::cout << "[MPTA]    Setting " << func_name_map_[var.first] << " "
              << var.second->vid->name_hint << " " << support::PrintVector(to_add) << " "
              << support::PrintVector(res) << " " << reason << " " << changed_ << std::endl;
    return res;
  }

  std::vector<bool> Add(const MPTAFunctionKey& func, const std::vector<bool>& to_add) {
    return Add<MPTAFunctionKey, MPTAFunctionStateMap>(function_states_, func, to_add);
  }

  MPTAVarKey VarKey(const FunctionNode* ctx, const VarNode* var) {
    return std::make_pair(ctx, var);
  }

  MPTAFunctionKey FunctionKey(const FunctionNode* ctx, const FunctionNode* func) {
    return std::make_pair(ctx, func);
  }

  MPTAOpKey OpKey(const FunctionNode* ctx, const CallNode* op) { return std::make_pair(ctx, op); }

  std::vector<bool> VisitBody(const Function& func) {
    std::cout << "[MPTA]  Visiting body " << func_name_map_[func.get()] << std::endl;
    stack_.push_back(func.get());
    auto res = VisitExpr(func->body);
    stack_.pop_back();
    std::cout << "[MPTA]  Visited body " << func_name_map_[func.get()] << std::endl;
    return res;
  }

  std::vector<bool> VisitExpr_(const ConstantNode* op) { return {true}; }

  std::vector<bool> VisitExpr_(const TupleNode* op) {
    std::vector<bool> res;
    for (size_t i = 0; i < op->fields.size(); ++i) {
      res.push_back(Collapse(VisitExpr(op->fields[i])));
    }
    return res;
  }

  std::vector<bool> VisitExpr_(const VarNode* op) {
    if (function_environments_[GetCurrentFunction()].count(op)) {
      return function_environments_[GetCurrentFunction()][op];
    }
    auto key = VarKey(GetCurrentContext(), op);
    if (!var_states_.count(key)) {
      var_states_[key] = CreateStateForType(op->checked_type(), true);
      changed_ = true;
    }
    return var_states_[key];
  }

  std::vector<bool> VisitExpr_(const GlobalVarNode* op) { return {false}; }

  std::vector<bool> VisitExpr_(const FunctionNode* op) { return {false}; }

  bool IsMapFuncInModule() { return map_fun_node_ != nullptr; }

  const FunctionNode* GetMapFuncNode() { return map_fun_node_; }

  std::vector<bool> VisitMapBody(const FunctionNode* map_context) {
    auto map_fn_node = GetMapFuncNode();
    auto lambda_var = map_fn_node->params[0];
    FunctionSet lambda_callees;
    PostOrderVisit(map_fn_node->body, [&](const Expr& expr) {
      if (auto cn = expr.as<CallNode>()) {
        if (cn->op == lambda_var) {
          auto it = callees_map_.find(OpKey(map_context, cn));
          if (it != callees_map_.end()) {
            auto callees = it->second;
            lambda_callees.insert(callees.begin(), callees.end());
          }
        }
      }
    });

    stack_.push_back(map_fn_node);

    auto lambda_context = map_fn_node;
    std::vector<bool> ret{true};
    for (auto callee : lambda_callees) {
      auto name = func_name_map_[callee];
      auto callee_fn = GetRef<Function>(callee);
      ICHECK_EQ(callee->params.size(), 1);
      auto fn_key = FunctionKey(lambda_context, callee);

      bool args_changed = false;
      bool old_changed = false;
      std::swap(old_changed, changed_);
      auto param_key = VarKey(lambda_context, callee->params[0].get());
      auto res = Add(param_key, {false}, "Map list param arg");
      std::swap(old_changed, changed_);
      changed_ = changed_ || old_changed;
      args_changed = args_changed || old_changed;

      bool visited = visited_functions_.count(fn_key);
      if (args_changed || !visited) {
        visited_functions_.insert(fn_key);
        auto res = VisitBody(callee_fn);
        res = Add(fn_key, res);
        ret = Merge(res, ret).first;
      } else if (visited) {
        auto it = function_states_.find(fn_key);
        if (it != function_states_.end()) {
          ret = Merge(it->second, ret).first;
        }
      }
    }
    stack_.pop_back();
    return ret;
  }

  std::vector<bool> VisitExpr_(const CallNode* op) {
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      return VisitExpr(on_device_props.body);
    }

    if (op->op.as<OpNode>() || op->op.as<ConstructorNode>()) {
      return CreateStateForType(op->checked_type(), false);
    } else {
      auto callee_context = GetCurrentFunction();
      auto it = callees_map_.find(OpKey(GetCurrentContext(), op));
      if (it == callees_map_.end()) {
        for (auto kv : callees_map_) {
          std::cout << func_name_map_[kv.first.first] << " " << kv.first.second->op << std::endl;
        }
      }
      ICHECK(it != callees_map_.end())
          << func_name_map_[GetCurrentContext()] << " " << GetCurrentContext() << " " << op->op;
      auto callees = it->second;
      bool callee_may_be_recursive = false;

      auto ret = CreateStateForType(op->checked_type(), false);
      for (auto callee : callees) {
        if (IsMapFuncInModule() && callee == GetMapFuncNode()) {
          auto lambda_state = VisitExpr(op->args[0]);
          auto list_state = VisitExpr(op->args[1]);
          return VisitMapBody(GetCurrentFunction());
        }

        auto name = func_name_map_[callee];
        bool print = false;  //(name == "map");
        if (print) {
          std::cout << "[MPTA]  map call " << PrettyPrint(RemoveOnDeviceCalls(GetRef<Expr>(op)))
                    << std::endl;
        }
        auto callee_fn = GetRef<Function>(callee);
        auto fn_key = FunctionKey(callee_context, callee);
        bool args_changed = false;
        auto op_state = VisitExpr(op->op);
        for (size_t i = 0; i < op->args.size(); ++i) {
          auto param_state = VisitExpr(op->args[i]);
          bool old_changed = false;
          std::swap(old_changed, changed_);
          auto param_key = VarKey(callee_context, callee->params[i].get());
          std::stringstream arg_str;
          arg_str << op->args[i];
          auto res = Add(
              param_key, param_state,
              "Function param arg " + arg_str.str() + " in " + func_name_map_[GetCurrentContext()]);

          // res = Add(param_key, op_state, "OpState");
          std::swap(old_changed, changed_);
          changed_ = changed_ || old_changed;
          args_changed = args_changed || old_changed;
          if (print && i == op->args.size() - 1) {
            std::cout << "[MPTA]    map list depth " << support::PrintVector(param_state) << " "
                      << support::PrintVector(res) << " " << op->args[i] << " " << callee->params[i]
                      << std::endl;
          }
        }

        bool visited = visited_functions_.count(fn_key);
        int path = -1;
        if (args_changed || !visited) {
          path = 1;
          visited_functions_.insert(fn_key);
          auto res = VisitBody(callee_fn);
          res = Add(fn_key, res);
          ret = Merge(res, ret).first;
        } else if (visited) {
          auto it = function_states_.find(fn_key);
          if (it != function_states_.end()) {
            path = 2;
            ret = Merge(it->second, ret).first;
          } else {
            path = 3;
          }
        }

        if (print) {
          std::cout << "[MPTA]   treelstm call ret " << support::PrintVector(ret) << " "
                    << args_changed << " " << path << std::endl;
        }
      }
      return ret;
    }
  }

  std::vector<bool> VisitExpr_(const LetNode* op) {
    auto value_res = VisitExpr(op->value);
    Add(VarKey(GetCurrentContext(), op->var.get()), value_res, "Let value");
    return VisitExpr(op->body);
  }

  std::vector<bool> VisitExpr_(const IfNode* op) {
    VisitExpr(op->cond);
    auto then_ret = VisitExpr(op->true_branch);
    auto else_ret = VisitExpr(op->false_branch);
    return Merge(then_ret, else_ret).first;
  }

  std::vector<bool> VisitExpr_(const OpNode* op) { return {false}; }

  std::vector<bool> VisitExpr_(const TupleGetItemNode* op) {
    return {VisitExpr(op->tuple)[op->index]};
  }

  std::vector<bool> VisitExpr_(const RefCreateNode* op) { return {false}; }

  std::vector<bool> VisitExpr_(const RefReadNode* op) { return {false}; }

  std::vector<bool> VisitExpr_(const RefWriteNode* op) { return {false}; }

  std::vector<bool> VisitExpr_(const ConstructorNode* op) { return {false}; }

  std::vector<bool> VisitExpr_(const MatchNode* op) {
    auto input_state = Collapse(VisitExpr(op->data));
    std::stringstream data_str;
    data_str << op->data;
    std::vector<bool> ret = CreateStateForType(op->checked_type(), true);
    for (auto clause : op->clauses) {
      auto pattern_vars = CollectPatternVars(clause->lhs);
      for (auto& var : pattern_vars) {
        Add(VarKey(GetCurrentContext(), var.get()),
            CreateStateForType(var->checked_type(), input_state),
            "Match pattern " + data_str.str());
      }
      auto clause_ret = VisitExpr(clause->rhs);
      ret = Merge(ret, clause_ret).first;
    }
    return ret;
  }

  const IRModule& mod_;
  const FunctionSet& recursive_functions_;
  const CalleesMap& callees_map_;

  MPTAVarStateMap var_states_;
  MPTAFunctionStateMap function_states_;
  MPTAInvokeTVMOpDepthMap prim_func_call_depths_;
  std::unordered_set<MPTAFunctionKey, PairHash, PairEquals> visited_functions_;
  size_t max_iterations_{20};
  bool changed_{false};
  std::unordered_map<const BaseFuncNode*, std::string> func_name_map_;
  std::vector<const FunctionNode*> stack_;
  std::unordered_map<const FunctionNode*, std::unordered_map<const VarNode*, std::vector<bool>>>
      function_environments_;
  const FunctionNode* map_fun_node_{nullptr};
};

}  // namespace

Map<Function, Array<Bool>> ModelParameterTaintAnalysis(const IRModule& mod) {
  std::cout << "[MPTA] Starting hoisting" << std::endl;
  auto recursive_res = GetRecursiveFunctions(mod);
  auto recursive_functions = recursive_res.first;
  auto callees_map = recursive_res.second;
  MPTAnalysis analysis(mod, recursive_functions, callees_map);
  return analysis.PerformAnalysis();
}

}  // namespace tec
}  // namespace relay
}  // namespace tvm
