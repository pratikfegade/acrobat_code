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
#include "../transforms/pass_utils.h"

namespace tvm {
namespace relay {
namespace tec {

namespace {
Array<Bool> ConstructBoolArray(size_t size, bool value) {
  Array<Bool> result;
  for (size_t i = 0; i < size; ++i) {
    result.push_back(Bool(value));
  }
  return result;
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

Array<Bool> CreateStateForType(const Type& type, bool value) {
  size_t state_size = 1;
  if (auto ttn = type.as<TupleTypeNode>()) {
    state_size = ttn->fields.size();
  }
  return ConstructBoolArray(state_size, value);
}

class ModelParameterTaintVisitor : public ExprFunctor<Array<Bool>(const Expr& n)> {
 public:
  ModelParameterTaintVisitor(const IRModule& mod, Map<Var, Array<Bool>>* p_var_state,
                             Map<Function, Array<Bool>>* p_function_state)
      : mod_(mod), p_var_state_(p_var_state), p_function_state_(p_function_state) {}

  void Run(const Function& func, std::string name) {
    print_ = false;  //(name == "lstm_cell");
    if (print_) {
      std::cout << "[MPTA]  Fyunc " << name << " " << RemoveOnDeviceCalls(func) << std::endl;
    }
    this->VisitExpr(func);
  }

  template <typename T>
  Array<Bool> MergeAndSet(Map<T, Array<Bool>>* p_map, T key, Array<Bool> vals2) {
    auto it = p_map->find(key);
    if (it == p_map->end()) {
      changed |= true;
      p_map->Set(key, vals2);
      return vals2;
    } else {
      Array<Bool> vals1 = (*it).second;
      auto res = Merge(vals1, vals2);
      changed |= res.second;
      p_map->Set(key, res.first);
      return res.first;
    }
  }

  Array<Bool> MergeAndSet(Var var, Array<Bool> vals2) {
    return MergeAndSet<Var>(p_var_state_, var, vals2);
  }

  Array<Bool> MergeAndSet(Function func, Array<Bool> vals2) {
    auto ret = MergeAndSet<Function>(p_function_state_, func, vals2);
    ICHECK(p_function_state_->count(func));
    return ret;
  }

  Array<Bool> VisitExpr_(const ConstantNode* op) { return Array<Bool>({Bool(true)}); }

  Array<Bool> VisitExpr_(const TupleNode* op) {
    Array<Bool> result;
    for (auto field : op->fields) {
      Array<Bool> field_state = VisitExpr(field);
      result.push_back(AndState(field_state));
    }
    return result;
  }

  Array<Bool> VisitExpr_(const VarNode* op) {
    auto var = GetRef<Var>(op);
    // std::cout << "[MPTA]    Visiting var " << var->vid->name_hint << std::endl;
    auto it = p_var_state_->find(var);
    if (it == p_var_state_->end()) {
      Array<Bool> state = CreateStateForType(op->checked_type(), true);
      changed |= true;
      p_var_state_->Set(var, state);
      // std::cout << "[MPTA]     NoFound " << state << std::endl;
      return state;
    } else {
      // std::cout << "[MPTA]     Found " << (*it).second << std::endl;
      return (*it).second;
    }
  };

  Array<Bool> VisitExpr_(const GlobalVarNode* op) { return Array<Bool>({Bool(false)}); };

  Array<Bool> VisitExpr_(const FunctionNode* op) {
    if (print_) {
      std::cout << "[FUNC] Visiting function " << op << " " << ObjectHash()(GetRef<Expr>(op))
                << std::endl;
    }
    auto func = GetRef<Function>(op);
    if (!visited_functions_.count(func)) {
      visited_functions_.Set(func, Bool(true));
      if (print_) {
        std::cout << "[FUNC]  Visiting first" << std::endl;
      }
      bool all_params_model_params = true;
      for (auto p : op->params) {
        auto it = p_var_state_->find(p);
        if (it == p_var_state_->end() || AndState((*it).second).operator bool()) continue;
        all_params_model_params = false;
        break;
      }
      if (all_params_model_params) {
        if (print_) {
          std::cout << "[FUNC]   Returning 1 for function " << ret << " "
                    << p_function_state_->count(func) << std::endl;
        }
        return MergeAndSet(func, CreateStateForType(op->ret_type, true));
      } else {
        auto ret = MergeAndSet(func, VisitExpr(op->body));
        if (print_) {
          std::cout << "[FUNC]   Returning 2 for function " << ret << " "
                    << p_function_state_->count(func) << std::endl;
        }
        return ret;
      }
    } else {
      auto it = p_function_state_->find(func);
      if (it == p_function_state_->end()) {
        // Recursive function visiting twice
        return CreateStateForType(op->ret_type, true);
      } else {
        return (*it).second;
      }
    }
  }

  Array<Bool> VisitExpr_(const CallNode* op) {
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      return VisitExpr(on_device_props.body);
    }
    if (print_) {
      // std::cout << "[CALL] Call  " << op->op << std::endl;
    }
    bool print = false;
    auto get_function = [&](const Expr& expr) {
      const FunctionNode* callee_func = nullptr;
      if (expr.as<GlobalVarNode>()) {
        // print = true;
        auto func = this->mod_->Lookup(Downcast<GlobalVar>(expr));
        if (auto fn = func.as<FunctionNode>()) {
          callee_func = fn;
        }
      } else if (auto fn = expr.as<FunctionNode>()) {
        callee_func = fn;
      }
      return callee_func;
    };

    const FunctionNode* callee_func = nullptr;
    if (auto vn = op->op.as<VarNode>()) {
      auto it = let_shortcuts_.find(Downcast<Var>(op->op));
      if (it != let_shortcuts_.end()) {
        callee_func = get_function(it->second);
      }
    }
    if (callee_func == nullptr) {
      callee_func = get_function(op->op);
    }

    if (callee_func == nullptr) {
      return CreateStateForType(op->checked_type(), false);
    }

    if (print) {
      std::cout << "[CALL] Call to callee" << std::endl;
    }

    for (size_t i = 0; i < op->args.size(); ++i) {
      Array<Bool> param_state = VisitExpr(op->args[i]);
      auto res = MergeAndSet(callee_func->params[i], param_state);
      if (print) {
        std::cout << "[CALL]   Setting callee arg " << callee_func->params[i]->vid->name_hint << ""
                  << param_state << " " << res << std::endl;
      }
    }

    // if (print) {
    // std::cout << "[CALL]  Visiting function" << std::endl;
    // std::cout << GetRef<Function>(callee_func) << std::endl;
    // }
    auto res = VisitExpr(GetRef<Function>(callee_func));
    if (print) {
      std::cout << "[CALL]  Call res " << res << std::endl;
    }
    return res;
  }

  Array<Bool> VisitExpr_(const LetNode* op) {
    // std::cout << "[MPTA] Let " << op->var << std::endl;
    if (op->value.as<GlobalVarNode>() || op->value.as<VarNode>() || op->value.as<FunctionNode>()) {
      let_shortcuts_.insert({op->var, op->value});
    }
    Array<Bool> value_taints = VisitExpr(op->value);
    MergeAndSet(op->var, value_taints);
    return VisitExpr(op->body);
  }

  Array<Bool> VisitExpr_(const IfNode* op) {
    VisitExpr(op->cond);
    return Merge(VisitExpr(op->true_branch), VisitExpr(op->false_branch)).first;
  }

  Array<Bool> VisitExpr_(const OpNode* op) { return Array<Bool>({Bool(false)}); }

  Array<Bool> VisitExpr_(const TupleGetItemNode* op) {
    Array<Bool> tuple_state = VisitExpr(op->tuple);
    ICHECK_GE(tuple_state.size(), op->index);
    return CreateStateForType(op->checked_type(), tuple_state[op->index].operator bool());
  }

  Array<Bool> VisitExpr_(const RefCreateNode* op) { return Array<Bool>({Bool(false)}); }

  Array<Bool> VisitExpr_(const RefReadNode* op) { return Array<Bool>({Bool(false)}); }

  Array<Bool> VisitExpr_(const RefWriteNode* op) { return Array<Bool>({Bool(false)}); }

  Array<Bool> VisitExpr_(const ConstructorNode* op) { return Array<Bool>({Bool(false)}); }

  Array<Bool> VisitExpr_(const MatchNode* op) {
    VisitExpr(op->data);
    Array<Bool> res = CreateStateForType(op->checked_type(), true);
    for (auto clause : op->clauses) {
      res = Merge(res, VisitExpr(clause->rhs)).first;
    }
    return res;
  }

  bool isConstantPrimFunc(const tir::PrimFunc& func) { return (func->params.size() == 0); }

  bool changed{false};

 private:
  const IRModule& mod_;
  Map<Var, Array<Bool>>* p_var_state_;
  Map<Function, Array<Bool>>* p_function_state_;
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> let_shortcuts_;
  Map<Function, Bool> visited_functions_;
  Map<tir::PrimFunc, Bool> constant_prim_funcs_;
  bool print_;
};

class ResultGatherer : public ExprVisitor {
 public:
  ResultGatherer(const Map<Var, Array<Bool>>& var_states,
                 const Map<Function, Array<Bool>>& function_states)
      : var_states_(var_states), function_states_(function_states) {}

  void Flatten(Array<Bool>* p_param_states, const Var& var) {
    size_t state_size = 1;
    if (auto ttn = var->checked_type().as<TupleTypeNode>()) {
      state_size = ttn->fields.size();
    }
    auto it = var_states_.find(var);
    Array<Bool> state =
        (it != var_states_.end()) ? (*it).second : ConstructBoolArray(state_size, false);
    ICHECK_EQ(state.size(), state_size);
    p_param_states->push_back_all(state);
  }

  size_t FlattenTupleType(const Type& type) {
    if (auto ttn = type.as<TupleTypeNode>()) {
      size_t sum = 0;
      for (auto field : ttn->fields) {
        sum += FlattenTupleType(field);
      }
      return sum;
    } else {
      return 1;
    }
  }

  void VisitExpr_(const FunctionNode* op) {
    auto hash = op->GetAttr<String>("hash", "no_hash");
    Array<Bool> param_states;
    for (auto param : op->params) {
      Flatten(&param_states, param);
    }

    auto it = function_states_.find(GetRef<Function>(op));
    if (it == function_states_.end()) {
      size_t flattened_size = FlattenTupleType(op->ret_type);
      param_states.push_back_all(ConstructBoolArray(flattened_size, false));
    } else {
      auto output_state = (*it).second;
      if (auto ttn = op->ret_type.as<TupleTypeNode>()) {
        ICHECK_EQ(output_state.size(), ttn->fields.size());
        for (size_t i = 0; i < ttn->fields.size(); ++i) {
          size_t flattened_size = FlattenTupleType(ttn->fields[i]);
          param_states.push_back_all(
              ConstructBoolArray(flattened_size, output_state[i].operator bool()));
        }
      } else {
        param_states.push_back_all(output_state);
      }
    }

    result.Set(GetRef<Function>(op), param_states);
    ExprVisitor::VisitExpr_(op);
  }

  const Map<Var, Array<Bool>>& var_states_;
  const Map<Function, Array<Bool>>& function_states_;
  Map<Function, Array<Bool>> result;
};
}  // namespace

Map<Function, Array<Bool>> ModelParameterTaintAnalysis(const IRModule& mod) {
  // std::cout << "[MPTA] Starting taint analysis" << std::endl;
  auto call_graph = CallGraph(mod);

  Map<Var, Array<Bool>> var_states;
  Map<Function, Array<Bool>> function_states;
  for (auto pair : mod->functions) {
    auto function = pair.second;
    // std::cout << "[MPTA]  Initing func params " << pair.first << " " <<
    // function.as<FunctionNode>()
    // << std::endl;
    auto model_parameter_list =
        function->GetAttr<Array<Integer>>("model_parameters", Array<Integer>()).value();
    if (model_parameter_list.size() == 0) {
      continue;
    }
    if (auto func_node = function.as<FunctionNode>()) {
      // std::cout << "[MPTA]   Func " << pair.first->name_hint << std::endl;
      for (size_t i = 0; i < func_node->params.size(); ++i) {
        auto param = func_node->params[i];
        bool is_model_parameter = model_parameter_list[i]->value > 0;
        var_states.Set(param, CreateStateForType(param->checked_type(), is_model_parameter));
        // std::cout << "[MPTA]    Param " << param->vid->name_hint << " " << param.get() << " "
        // << var_states.at(param) << std::endl;
      }
    }
  }

  for (size_t i = 0; i < 50; ++i) {
    ModelParameterTaintVisitor visitor(mod, &var_states, &function_states);
    for (auto cge : call_graph->TopologicalOrder()) {
      auto func = mod->Lookup(cge->GetGlobalVar());
      if (func.as<FunctionNode>()) {
        // std::cout << "[MPTA] Func " << cge->GetGlobalVar() << std::endl;
        visitor.Run(Downcast<Function>(func), cge->GetGlobalVar()->name_hint);
      }
    }
    if (!visitor.changed) {
      break;
    }
  }

  // for (auto p : var_states) {
  // std::cout << p.first->vid->name_hint << " " << p.second << std::endl;
  // }

  // std::cout << "[MPTA]  Gathering final results" << std::endl;
  ResultGatherer gatherer(var_states, function_states);
  for (auto pair : mod->functions) {
    if (pair.second.as<FunctionNode>()) {
      gatherer(Downcast<Function>(pair.second));
    }
  }
  return gatherer.result;
}

}  // namespace tec
}  // namespace relay
}  // namespace tvm
