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
        for (auto pattern : pcn->patterns) {
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

class FunctionPointerAnalysis : public FPABaseExprFunctor {
 public:
  FunctionPointerAnalysis(const IRModule& mod) : mod_(mod) {
    for (auto kv : mod_->functions) {
      func_name_map_[kv.second.get()] = kv.first->name_hint;
    }
    stack_.push_back(nullptr);
  }

  void PerformAnalysis() {
    auto call_graph = CallGraph(mod_);
    size_t i = 0;
    for (; i < max_iterations_; ++i) {
      // std::cout << "[FPA] ITERATION " << i << std::endl;
      visited_functions_.clear();
      changed_ = false;
      VisitBody(Downcast<Function>(mod_->Lookup("main")));
      if (!changed_) {
        break;
      }
    }

    for (auto kv : callees_map_) {
      auto context = kv.first.first;
      auto callsite = kv.first.second;
      auto callees = kv.second;
      callees_per_callsite_[callsite][context].insert(callees.begin(), callees.end());
    }
  }

  FunctionSet GetRecursiveFunctions() {
    PreciseCallGraph call_graph;
    for (auto kv : mod_->functions) {
      if (auto fn = kv.second.as<FunctionNode>()) {
        PostOrderVisit(fn->body, [&](const Expr& expr) {
          if (auto callsite = expr.as<CallNode>()) {
            FunctionSet callees;
            auto it = callees_per_callsite_.find(callsite);
            if (it == callees_per_callsite_.end()) {
              return;
            }
            auto callees_per_context = it->second;
            for (auto kvkv : callees_per_context) {
              callees.insert(kvkv.second.begin(), kvkv.second.end());
            }
            call_graph[fn] = callees;
          }
        });
      }
    }

    FunctionSet recursive_functions;
    for (auto kv : mod_->functions) {
      if (auto fn = kv.second.as<FunctionNode>()) {
        FunctionSet visited;
        if (Reachable(call_graph, fn, fn, visited, true)) {
          recursive_functions.insert(fn);
          // std::cout << "[ADF] Recursive func " << fn << " " << func_name_map_[fn] << std::endl;
        }
      }
    }
    return recursive_functions;
  }

  CalleesMap GetCalleesMap() {
    // for (auto kv : callees_map_) {
    // std::cout << "[FPA] Callees " << func_name_map_[kv.first.first] << " "
    // << func_name_map_[callsite_to_function_[kv.first.second]] << " "
    // << kv.first.second->op << std::endl;
    // for (auto f : kv.second) {
    // std::cout << "[FPA]    " << func_name_map_[f] << std::endl;
    // }
    // }

    return callees_map_;
  }

 private:
  const FunctionNode* GetCurrentContext() {
    ICHECK_GE(stack_.size(), 2);
    return stack_[stack_.size() - 2];
  }

  const FunctionNode* GetCurrentFunction() { return stack_.back(); }

  std::pair<FunctionSet, bool> Merge(const FunctionSet& s1, const FunctionSet& s2) {
    FunctionSet ret;
    ret.insert(s1.begin(), s1.end());
    ret.insert(s2.begin(), s2.end());
    bool changed = (s1.size() != ret.size());
    return std::make_pair(ret, changed);
  }

  template <typename T, typename MapType>
  FunctionSet Add(MapType& map, const T& key, const FunctionSet& to_add) {
    auto old_size = map[key].size();
    map[key].insert(to_add.begin(), to_add.end());
    auto new_size = map[key].size();
    if (old_size != new_size) {
      changed_ = true;
    }
    return map.at(key);
  }

  FunctionSet Add(const FPAVarKey& var, const FunctionSet& to_add) {
    return Add<FPAVarKey, FPAVarStateMap>(var_states_, var, to_add);
  }

  FunctionSet Add(const FPAFunctionKey& func, const FunctionSet& to_add) {
    return Add<FPAFunctionKey, FPAFunctionStateMap>(function_states_, func, to_add);
  }

  FPAVarKey VarKey(const FunctionNode* ctx, const VarNode* var) { return std::make_pair(ctx, var); }

  FPAFunctionKey FunctionKey(const FunctionNode* ctx, const FunctionNode* func) {
    return std::make_pair(ctx, func);
  }

  FPAOpKey OpKey(const FunctionNode* ctx, const CallNode* op) { return std::make_pair(ctx, op); }

  FunctionSet VisitBody(const Function& func) {
    // std::cout << "[FPA] Visiting function " << func_name_map_[func.get()] << std::endl;
    stack_.push_back(func.get());
    auto res = VisitExpr(func->body);
    stack_.pop_back();
    return res;
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

  FunctionSet VisitExpr_(const ConstantNode* op) override { return {}; }

  FunctionSet VisitExpr_(const TupleNode* op) override {
    FunctionSet ret;
    for (auto field : op->fields) {
      auto field_ret = VisitExpr(field);
      ret.insert(field_ret.begin(), field_ret.end());
    }
    return ret;
  }

  FunctionSet VisitExpr_(const VarNode* op) override {
    auto var_key = VarKey(GetCurrentContext(), op);
    auto it = var_states_.find(var_key);
    if (it == var_states_.end()) {
      var_states_[var_key] = {};
      changed_ = true;
    }
    return var_states_[var_key];
  }

  FunctionSet VisitExpr_(const GlobalVarNode* op) override {
    return {mod_->Lookup(GetRef<GlobalVar>(op)).as<FunctionNode>()};
  }

  FunctionSet VisitExpr_(const FunctionNode* op) override { return {op}; }

  std::string FunctionSetToStr(const FunctionSet& set) {
    std::stringstream os;
    os << "Set(" << set.size() << ")[";
    for (auto f : set) {
      os << func_name_map_[f] << " ";
    }
    os << "]";
    return os.str();
  }

  FunctionSet VisitExpr_(const CallNode* op) override {
    callsite_to_function_[op] = GetCurrentFunction();
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      return VisitExpr(on_device_props.body);
    }
    if (op->op.as<OpNode>()) {
      return {};
    }
    auto callee_set = VisitExpr(op->op);
    callees_map_[OpKey(GetCurrentContext(), op)] =
        Merge(callees_map_[OpKey(GetCurrentContext(), op)], callee_set).first;
    // std::cout << "[FPA] Setting callee " << func_name_map_[GetCurrentContext()] << " "
    // << GetCurrentContext() << " " << op->op << " " << FunctionSetToStr(callee_set)
    // << std::endl;
    if (callee_set.empty()) {
      for (auto arg : op->args) {
        this->VisitExpr(arg);
      }
    }
    FunctionSet ret;
    for (auto callee : callee_set) {
      auto callee_key = FunctionKey(GetCurrentContext(), callee);
      bool print = false;  //(func_name_map_[callee] == "map");
      auto callee_fn = GetRef<Function>(callee);
      bool args_changed = false;
      for (size_t i = 0; i < op->args.size(); ++i) {
        auto param_state = VisitExpr(op->args[i]);
        bool old_changed = false;
        std::swap(old_changed, changed_);
        auto res = Add(VarKey(GetCurrentFunction(), callee->params[i].get()), param_state);
        std::swap(old_changed, changed_);
        changed_ = changed_ || old_changed;
        args_changed = args_changed || old_changed;

        if (print) {
          std::cout << "[FPA]   call " << func_name_map_[callee] << " " << op->args[i] << " "
                    << args_changed << " " << FunctionSetToStr(param_state) << std::endl;
          for (auto pt : res) {
            std::cout << "[FPA]    Pointeee " << func_name_map_[pt] << std::endl;
          }
        }
      }

      bool visited = visited_functions_.count(callee_key);
      if (args_changed || !visited) {
        visited_functions_.insert(callee_key);
        if (print) {
          std::cout << "[FPA]  Visiting body " << func_name_map_[callee] << std::endl;
        }
        auto res = VisitBody(callee_fn);
        Add(callee_key, res);
        ret.insert(res.begin(), res.end());
      } else if (!visited) {
        auto it = function_states_.find(callee_key);
        if (it != function_states_.end()) {
          ret.insert(it->second.begin(), it->second.end());
        }
      }
    }
    return ret;
  }

  FunctionSet VisitExpr_(const LetNode* op) override {
    auto value_res = VisitExpr(op->value);
    // std::cout << "[FPA]   Setting callee " << func_name_map_[GetCurrentFunction()] << " "
    // << op->var->vid->name_hint << " " << FunctionSetToStr(value_res) << std::endl;
    Add(VarKey(GetCurrentContext(), op->var.get()), value_res);
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
    return {};
  }

  FunctionSet VisitExpr_(const RefReadNode* op) override {
    ICHECK(false) << "Refs not supported for now";
    return {};
  }

  FunctionSet VisitExpr_(const RefWriteNode* op) override {
    ICHECK(false) << "Refs not supported for now";
    return {};
  }

  FunctionSet VisitExpr_(const ConstructorNode* op) override { return {}; }

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
  std::unordered_set<FPAFunctionKey, PairHash, PairEquals> visited_functions_;
  size_t max_iterations_{50};
  bool changed_{false};
  std::unordered_map<const BaseFuncNode*, std::string> func_name_map_;
  std::vector<const FunctionNode*> stack_;
  CalleesMap callees_map_;
  std::unordered_map<const CallNode*, std::unordered_map<const FunctionNode*, FunctionSet>>
      callees_per_callsite_;
  std::unordered_map<const CallNode*, const FunctionNode*> callsite_to_function_;
};

std::pair<FunctionSet, CalleesMap> GetRecursiveFunctions(const IRModule& mod) {
  FunctionPointerAnalysis analysis(mod);
  analysis.PerformAnalysis();
  auto recursive_functions = analysis.GetRecursiveFunctions();
  auto callees_map = analysis.GetCalleesMap();
  return std::make_pair(recursive_functions, callees_map);
}

CalleesMap GetCalleesMap(const IRModule& mod) {
  FunctionPointerAnalysis analysis(mod);
  analysis.PerformAnalysis();
  return analysis.GetCalleesMap();
}

}  // namespace relay
}  // namespace tvm
