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

using TDCOTAVarKey = std::pair<const FunctionNode*, const VarNode*>;
using TDCOTAFunctionKey = std::pair<const FunctionNode*, const FunctionNode*>;
using TDCOTAOpKey = std::pair<const FunctionNode*, const CallNode*>;
using TDCOTABaseExprFunctor = ExprFunctor<bool(const Expr& n)>;
using TDCOTAVarStateMap = std::unordered_map<TDCOTAVarKey, bool, PairHash, PairEquals>;
using TDCOTAInvokeTVMOpDepthMap = std::unordered_map<TDCOTAOpKey, bool, PairHash, PairEquals>;
using TDCOTAFunctionStateMap = std::unordered_map<TDCOTAFunctionKey, bool, PairHash, PairEquals>;

class TensorDependentControlOpsTaintAnalysis : public TDCOTABaseExprFunctor {
 public:
  TensorDependentControlOpsTaintAnalysis(IRModule& mod, const CalleesMap& callees_map)
      : mod_(mod), callees_map_(callees_map) {
    for (auto kv : mod_->functions) {
      func_name_map_[kv.second.get()] = kv.first->name_hint;
      std::cout << "[FUNC_NAME] " << kv.second.get() << " " << kv.first->name_hint << std::endl;
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
      std::cout << "[TDCOTA] ITERATION " << i << std::endl;
      visited_functions_.clear();
      changed_ = false;
      auto entry_func = Downcast<Function>(mod_->Lookup("main"));
      VisitBody(entry_func);
      if (!changed_) {
        break;
      }
    }

    return mod_;
  }

 private:
  const FunctionNode* GetCurrentContext() { return stack_[stack_.size() - 2]; }

  const FunctionNode* GetCurrentFunction() { return stack_.back(); }

  std::pair<bool, bool> Merge(const bool& vals1, const bool& vals2) {
    bool changed = false;
    bool result = vals1 || vals2;
    if (result != vals1) {
      changed = true;
    }
    return std::make_pair(result, changed);
  }

  bool Collapse(const bool& vals) { return vals; }

  template <typename T, typename MapType>
  bool Add_(MapType& map, const T& key, const bool& to_add) {
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

  bool Add(const TDCOTAVarKey& var, const bool& to_add, const std::string& reason) {
    return Add_<TDCOTAVarKey, TDCOTAVarStateMap>(var_states_, var, to_add);
  }

  bool Add(const TDCOTAFunctionKey& func, const bool& to_add) {
    return Add_<TDCOTAFunctionKey, TDCOTAFunctionStateMap>(function_states_, func, to_add);
  }

  TDCOTAVarKey VarKey(const FunctionNode* ctx, const VarNode* var) {
    return std::make_pair(ctx, var);
  }

  TDCOTAFunctionKey FunctionKey(const FunctionNode* ctx, const FunctionNode* func) {
    return std::make_pair(ctx, func);
  }

  TDCOTAOpKey OpKey(const FunctionNode* ctx, const CallNode* op) { return std::make_pair(ctx, op); }

  bool VisitBody(const Function& func) {
    stack_.push_back(func.get());
    auto res = VisitExpr(func->body);
    stack_.pop_back();
    return res;
  }

  bool VisitExpr_(const ConstantNode* op) { return 0; }

  bool VisitExpr_(const TupleNode* op) {
    int tuple_depth = 0;
    for (size_t i = 0; i < op->fields.size(); ++i) {
      tuple_depth = Merge(tuple_depth, VisitExpr(op->fields[i])).first;
    }
    return tuple_depth;
  }

  bool VisitExpr_(const VarNode* op) {
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

  bool VisitExpr_(const GlobalVarNode* op) { return 0; }

  bool VisitExpr_(const FunctionNode* op) {
    auto free_vars = FreeVarsDedup(op->body);
    for (auto param : op->params) {
      free_vars.erase(param);
    }
    int depth = 0;
    for (auto var : free_vars) {
      auto res = VisitExpr(var);
      depth = Merge(depth, res).first;

      bool old_val = function_environments_[op][var.get()];
      auto merge_res = Merge(old_val, res);
      function_environments_[op][var.get()] = merge_res.first;
      // std::cout << "[TDCOTA] FuncEnv " << var->vid->name_hint << " " << var.get() << " "
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

  bool VisitMapBody(const int input_depth, const int lambda_depth,
                    const FunctionNode* map_context) {
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

  bool VisitExpr_(const CallNode* op) {
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      return VisitExpr(on_device_props.body);
    }

    if (op->op == GetInvokeTVMOp()) {
      // std::cout << "[TDCOTA] OpDepth " << GetCurrentContext() << " "
      // << PrettyPrint(RemoveOnDeviceCalls(GetRef<Expr>(op))) << std::endl;
      auto callee_prim_func = mod_->Lookup(Downcast<GlobalVar>(op->args[0]));
      auto access_modes_opt =
          callee_prim_func->GetAttr<Array<Integer>>(tir::attr::kDBArgAccessModes);
      ICHECK(access_modes_opt) << "No access modes found for " << op->args[0];
      auto access_modes = access_modes_opt.value();

      auto outputs_tuple = op->args[2].as<TupleNode>();
      for (auto output : outputs_tuple->fields) {
        ICHECK(output.as<VarNode>());
        auto var = Downcast<Var>(output);
        auto res = Add(VarKey(GetCurrentContext(), var.get()), true, "OpOutput");
      }
      return true;
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
          std::cout << "[TDCOTA]  map call " << PrettyPrint(RemoveOnDeviceCalls(GetRef<Expr>(op)))
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
            std::cout << "[TDCOTA]    map list depth " << param_state << " " << res << " "
                      << op->args[i] << " " << callee->params[i] << std::endl;
          }
        }

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

        if (print) {
          std::cout << "[TDCOTA]   treelstm call ret " << ret << " " << args_changed << std::endl;
        }
      }
      return ret;
    }
  }

  bool VisitExpr_(const LetNode* op) {
    auto value_res = VisitExpr(op->value);
    Add(VarKey(GetCurrentContext(), op->var.get()), value_res, "Let value");
    return VisitExpr(op->body);
  }

  bool VisitExpr_(const IfNode* op) {
    if (VisitExpr(op->cond)) {
      std::cout << "[TDCOTA] Dependent control stmt " << op->cond << std::endl;
      dependent_control_stmts_.insert(GetRef<Expr>(op));
    }
    auto then_ret = VisitExpr(op->true_branch);
    auto else_ret = VisitExpr(op->false_branch);
    return Merge(then_ret, else_ret).first;
  }

  bool VisitExpr_(const OpNode* op) { return 0; }

  bool VisitExpr_(const TupleGetItemNode* op) { return VisitExpr(op->tuple); }

  bool VisitExpr_(const RefCreateNode* op) { return 0; }

  bool VisitExpr_(const RefReadNode* op) { return 0; }

  bool VisitExpr_(const RefWriteNode* op) { return 0; }

  bool VisitExpr_(const ConstructorNode* op) { return 0; }

  bool VisitExpr_(const MatchNode* op) {
    auto input_taint = Collapse(VisitExpr(op->data));
    std::stringstream data_str;
    data_str << op->data;
    int ret = 0;
    for (auto clause : op->clauses) {
      auto pattern_vars = CollectPatternVars(clause->lhs);
      for (auto& var : pattern_vars) {
        Add(VarKey(GetCurrentContext(), var.get()), input_taint, "Match pattern " + data_str.str());
      }
      auto clause_ret = VisitExpr(clause->rhs);
      ret = Merge(ret, clause_ret).first;
    }
    return ret;
  }

  const IRModule& mod_;
  const CalleesMap& callees_map_;

  TDCOTAVarStateMap var_states_;
  TDCOTAFunctionStateMap function_states_;
  std::unordered_set<TDCOTAFunctionKey, PairHash, PairEquals> visited_functions_;
  size_t max_iterations_{20};
  bool changed_{false};
  std::unordered_map<const BaseFuncNode*, std::string> func_name_map_;
  std::vector<const FunctionNode*> stack_;
  std::unordered_map<const FunctionNode*, std::unordered_map<const VarNode*, int>>
      function_environments_;
  std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> dependent_control_stmts_;
};

IRModule IdentifyTensorDependentControlOps(IRModule& mod) {
  auto callees_map = GetCalleesMap(mod);
  TensorDependentControlOpsTaintAnalysis analysis(mod, callees_map);
  return analysis.PerformAnalysis();
}

namespace transform {
Pass TensorDependentControlIdentifierPass() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return IdentifyTensorDependentControlOps(m); };
  return CreateModulePass(pass_func, 0, "TensorDependentControlIdentifierPass", {});
}

TVM_REGISTER_GLOBAL("relay._transform.IdentifyTensorDependentControlOps")
    .set_body_typed(TensorDependentControlIdentifierPass);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
