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

using SAIVarKey = std::pair<const FunctionNode*, const VarNode*>;
using SAIFunctionKey = std::pair<const FunctionNode*, const FunctionNode*>;
using SAIOpKey = std::pair<const FunctionNode*, const CallNode*>;

using SAIBaseExprFunctor = ExprFunctor<int(const Expr& n)>;
using SAIVarStateMap = std::unordered_map<SAIVarKey, int, PairHash, PairEquals>;
using SAIInvokeTVMOpDepthMap = std::unordered_map<SAIOpKey, int, PairHash, PairEquals>;
using SAIFunctionStateMap = std::unordered_map<SAIFunctionKey, int, PairHash, PairEquals>;

constexpr int MAX_DEPTH_VALUE = 1 << 4;
class SchedulingAbstractInterpreter : public SAIBaseExprFunctor {
 public:
  SchedulingAbstractInterpreter(IRModule& mod, const FunctionSet& recursive_functions,
                                const CalleesMap& callees_map)
      : mod_(mod), recursive_functions_(recursive_functions), callees_map_(callees_map) {
    for (auto kv : mod_->functions) {
      func_name_map_[kv.second.get()] = kv.first->name_hint;
      // std::cout << "[FUNC_NAME] " << kv.second.get() << " " << kv.first->name_hint << std::endl;
    }
    stack_.push_back(nullptr);
  }

  IRModule PerformAnalysis() {
    auto main_func = Downcast<Function>(mod_->Lookup("main"));
    for (auto arg : main_func->params) {
      var_states_[VarKey(nullptr, arg.get())] = 0;
    }

    auto call_graph = CallGraph(mod_);
    size_t i = 0;
    for (; i < max_iterations_; ++i) {
      // std::cout << "[SAI] ITERATION " << i << std::endl;
      visited_functions_.clear();
      changed_ = false;
      auto entry_func = Downcast<Function>(mod_->Lookup("main"));
      VisitBody(entry_func);
      if (!changed_) {
        break;
      }
    }

    std::unordered_map<const VarNode*, int> merged_var_states;
    for (auto kv : var_states_) {
      merged_var_states[kv.first.second] =
          Merge(merged_var_states[kv.first.second], kv.second).first;
    }

    std::unordered_map<const FunctionNode*, int> merged_function_states;
    for (auto kv : function_states_) {
      merged_function_states[kv.first.second] =
          Merge(merged_function_states[kv.first.second], kv.second).first;
    }

    CallDepthMap merged_op_depths;
    for (auto kv : prim_func_call_depths_) {
      merged_op_depths[kv.first.second] = Merge(merged_op_depths[kv.first.second], kv.second).first;
    }

    if (false) {
      for (auto kv : merged_var_states) {
        std::cout << "[SAI]  Var Depths: " << kv.first->vid->name_hint << " " << kv.second
                  << std::endl;
      }

      for (auto kv : merged_function_states) {
        std::cout << "[SAI]  Function Depths: " << func_name_map_[kv.first] << " " << kv.second
                  << std::endl;
      }

      std::cout << "[SAI] Iterations: " << i << std::endl;
      for (auto kv : merged_op_depths) {
        if (kv.second < MAX_DEPTH_VALUE) {
          std::cout << "[SAI]  Call Depths: " << PrettyPrint(GetRef<Expr>(kv.first)) << " "
                    << kv.second << std::endl;
        }
      }
    }

    class AddDepthToCalls : public ExprMutator {
     public:
      AddDepthToCalls(const CallDepthMap& merged_op_depths) : merged_op_depths_(merged_op_depths) {}

      Expr VisitExpr_(const CallNode* op) {
        auto it = merged_op_depths_.find(op);
        auto call = ExprMutator::VisitExpr_(op);
        if (it != merged_op_depths_.end() && it->second < MAX_DEPTH_VALUE) {
          auto new_op = call.as<CallNode>();
          auto depth = it->second;
          ICHECK(new_op->attrs.defined() && new_op->attrs.as<DictAttrsNode>());
          auto attrs = Downcast<DictAttrs>(new_op->attrs);
          auto new_attrs = attrs.WithAttr(tir::attr::kDBGraphDepth, Integer(depth));
          return Call(new_op->op, new_op->args, new_attrs, new_op->type_args, new_op->span);
        }
        return call;
      }

      const CallDepthMap& merged_op_depths_;
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
  const FunctionNode* GetCurrentContext() { return stack_[stack_.size() - 2]; }

  const FunctionNode* GetCurrentFunction() { return stack_.back(); }

  std::pair<int, bool> Merge(const int& vals1, const int& vals2) {
    bool changed = false;
    int result = std::max(vals1, vals2);
    if (result != vals1) {
      changed = true;
    }
    return std::make_pair(result, changed);
  }

  int Collapse(const int& vals) { return vals; }

  template <typename T, typename MapType>
  int Add(MapType& map, const T& key, const int& to_add) {
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

  int Add(const SAIVarKey& var, const int& to_add, const std::string& reason) {
    // if (to_add == MAX_DEPTH_VALUE) {
    // std::cout << "Capping " << func_name_map_[var.first] << " " << var.second->vid->name_hint
    // << " " << reason << std::endl;
    // }
    return Add<SAIVarKey, SAIVarStateMap>(var_states_, var, to_add);
  }

  int Add(const SAIFunctionKey& func, const int& to_add) {
    return Add<SAIFunctionKey, SAIFunctionStateMap>(function_states_, func, to_add);
  }

  SAIVarKey VarKey(const FunctionNode* ctx, const VarNode* var) { return std::make_pair(ctx, var); }

  SAIFunctionKey FunctionKey(const FunctionNode* ctx, const FunctionNode* func) {
    return std::make_pair(ctx, func);
  }

  SAIOpKey OpKey(const FunctionNode* ctx, const CallNode* op) { return std::make_pair(ctx, op); }

  int CappedIncr(int n) {
    if (n >= MAX_DEPTH_VALUE) {
      return n;
    } else {
      ICHECK_LE(static_cast<long>(n) + static_cast<long>(1), static_cast<long>(MAX_DEPTH_VALUE));
      return n + 1;
    }
  }

  int VisitBody(const Function& func) {
    stack_.push_back(func.get());
    auto res = VisitExpr(func->body);
    stack_.pop_back();
    return res;
  }

  int VisitExpr_(const ConstantNode* op) { return 0; }

  int VisitExpr_(const TupleNode* op) {
    int tuple_depth = 0;
    for (size_t i = 0; i < op->fields.size(); ++i) {
      tuple_depth = Merge(tuple_depth, VisitExpr(op->fields[i])).first;
    }
    return tuple_depth;
  }

  int VisitExpr_(const VarNode* op) {
    if (function_environments_[GetCurrentFunction()].count(op)) {
      return function_environments_[GetCurrentFunction()][op];
    }

    auto key = VarKey(GetCurrentContext(), op);
    auto var = GetRef<Var>(op);
    auto it = var_states_.find(key);
    if (it == var_states_.end()) {
      var_states_[key] = 0;
      changed_ = true;
    }
    return var_states_[key];
  }

  int VisitExpr_(const GlobalVarNode* op) { return 0; }

  int VisitExpr_(const FunctionNode* op) {
    auto free_vars = FreeVarsDedup(op->body);
    for (auto param : op->params) {
      free_vars.erase(param);
    }
    int depth = 0;
    for (auto var : free_vars) {
      auto res = VisitExpr(var);
      depth = Merge(depth, res).first;

      int old_val = function_environments_[op][var.get()];
      auto merge_res = Merge(old_val, res);
      function_environments_[op][var.get()] = merge_res.first;
      // std::cout << "[SAI] FuncEnv " << var->vid->name_hint << " " << var.get() << " "
      // << merge_res.first << " " << function_environments_[op][var.get()] << std::endl;
      changed_ = changed_ || merge_res.second;
    }
    return depth;
  }

  const FunctionNode* GetMapFuncNode() {
    static const FunctionNode* node = mod_->Lookup("map").as<FunctionNode>();
    ICHECK(node);
    return node;
  }

  bool IsMapFuncInModule() { return mod_->ContainGlobalVar("map"); }

  int VisitMapBody(const int input_depth, const int lambda_depth, const FunctionNode* map_context) {
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
    int ret = 0;
    for (auto callee : lambda_callees) {
      auto name = func_name_map_[callee];
      auto callee_fn = GetRef<Function>(callee);
      ICHECK_EQ(callee->params.size(), 1);
      auto fn_key = FunctionKey(lambda_context, callee);

      bool args_changed = false;
      bool old_changed = false;
      std::swap(old_changed, changed_);
      auto param_key = VarKey(lambda_context, callee->params[0].get());
      auto res = Add(param_key, input_depth, "Map list param arg");
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

  int VisitExpr_(const CallNode* op) {
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      return VisitExpr(on_device_props.body);
    }

    if (op->op == GetInvokeTVMOp()) {
      auto callee_prim_func = mod_->Lookup(Downcast<GlobalVar>(op->args[0]));

      auto inputs_depth = VisitExpr(op->args[1]);

      auto max_inputs_depth = Collapse(inputs_depth);
      auto output_depth = CappedIncr(max_inputs_depth);
      auto outputs_tuple = op->args[2].as<TupleNode>();
      for (auto output : outputs_tuple->fields) {
        ICHECK(output.as<VarNode>());
        auto var = Downcast<Var>(output);
        auto res = Add(VarKey(GetCurrentContext(), var.get()), output_depth, "OpOutput");
      }
      prim_func_call_depths_[OpKey(GetCurrentContext(), op)] =
          std::max(prim_func_call_depths_[OpKey(GetCurrentContext(), op)], max_inputs_depth);
      return output_depth;
    } else if (op->op.as<OpNode>()) {
      return 0;
    } else if (op->op.as<ConstructorNode>()) {
      int args_depth = 0;
      for (auto arg : op->args) {
        args_depth = Merge(args_depth, VisitExpr(arg)).first;
      }
      return args_depth;
    } else {
      auto callee_context = GetCurrentFunction();
      auto it = callees_map_.find(OpKey(GetCurrentContext(), op));
      ICHECK(it != callees_map_.end())
          << func_name_map_[GetCurrentContext()] << " " << GetCurrentContext() << " " << op->op;
      auto callees = it->second;
      bool callee_may_be_recursive = false;

      int ret = 0;
      for (auto callee : callees) {
        if (IsMapFuncInModule() && callee == GetMapFuncNode()) {
          auto lambda_state = VisitExpr(op->args[0]);
          auto list_state = VisitExpr(op->args[1]);
          return VisitMapBody(list_state, lambda_state, GetCurrentFunction());
        }

        auto name = func_name_map_[callee];
        bool print = false;  //(name == "map");
        if (print) {
          std::cout << "[SAI]  map call " << PrettyPrint(RemoveOnDeviceCalls(GetRef<Expr>(op)))
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
            std::cout << "[SAI]    map list depth " << param_state << " " << res << " "
                      << op->args[i] << " " << callee->params[i] << std::endl;
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
          std::cout << "[SAI]   treelstm call ret " << ret << " " << args_changed << " " << path
                    << std::endl;
        }
      }
      return ret;
    }
  }

  int VisitExpr_(const LetNode* op) {
    auto value_res = VisitExpr(op->value);
    Add(VarKey(GetCurrentContext(), op->var.get()), value_res, "Let value");
    return VisitExpr(op->body);
  }

  int VisitExpr_(const IfNode* op) {
    VisitExpr(op->cond);
    auto then_ret = VisitExpr(op->true_branch);
    auto else_ret = VisitExpr(op->false_branch);
    return Merge(then_ret, else_ret).first;
  }

  int VisitExpr_(const OpNode* op) { return 0; }

  int VisitExpr_(const TupleGetItemNode* op) {
    auto tuple_res = VisitExpr(op->tuple);
    return tuple_res;
  }

  int VisitExpr_(const RefCreateNode* op) { return 0; }

  int VisitExpr_(const RefReadNode* op) { return 0; }

  int VisitExpr_(const RefWriteNode* op) { return 0; }

  int VisitExpr_(const ConstructorNode* op) { return 0; }

  int VisitExpr_(const MatchNode* op) {
    auto input_depth = Collapse(VisitExpr(op->data));
    std::stringstream data_str;
    data_str << op->data;
    int ret = 0;
    for (auto clause : op->clauses) {
      auto pattern_vars = CollectPatternVars(clause->lhs);
      for (auto& var : pattern_vars) {
        Add(VarKey(GetCurrentContext(), var.get()), input_depth, "Match pattern " + data_str.str());
      }
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
  std::unordered_set<SAIFunctionKey, PairHash, PairEquals> visited_functions_;
  size_t max_iterations_{20};
  bool changed_{false};
  std::unordered_map<const BaseFuncNode*, std::string> func_name_map_;
  std::vector<const FunctionNode*> stack_;
  std::unordered_map<const FunctionNode*, std::unordered_map<const VarNode*, int>>
      function_environments_;
};

IRModule ComputeConstantDepths(IRModule& mod) {
  auto recursive_res = GetRecursiveFunctions(mod);
  auto recursive_functions = recursive_res.first;
  auto callees_map = recursive_res.second;
  SchedulingAbstractInterpreter analysis(mod, recursive_functions, callees_map);
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
