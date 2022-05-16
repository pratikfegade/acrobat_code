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
 * \file src/relay/transforms/annotate_target.cc
 * \brief Wraps an expr with compiler_begin and compiler_end to indicate that
 * this expr should be handled by the external compiler.
 */

#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include "../../support/utils.h"
#include "../op/random/db_random.h"
#include "../transforms/function_pointer_analysis.h"
#include "pass_utils.h"

namespace tvm {
namespace relay {

namespace {
using TaintT = std::vector<bool>;

using PairHash = support::PairHash;
using PairEquals = support::PairEquals;

using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using BaseExprFunctor = ExprFunctor<void(const Expr&, TaintT)>;
using VarStateMap = std::unordered_map<const VarNode*, TaintT>;
using TupleStateMap = std::unordered_map<const ExprNode*, TaintT>;
using FunctionStateMap = std::unordered_map<const FunctionNode*, TaintT>;
using OnStackSet = std::unordered_set<const FunctionNode*>;

std::vector<bool> ConstructBoolArray(size_t size, bool value) {
  std::vector<bool> result;
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

std::vector<bool> CreateStateForType(const Type& type, bool value) {
  return ConstructBoolArray(GetTypeSize(type), value);
}

class ScalarTaintAnalysis : public BaseExprFunctor {
 public:
  ScalarTaintAnalysis(IRModule& mod, const CalleesMap& callees_map) : mod_(mod) {
    for (auto kv : callees_map) {
      auto call = kv.first.second;
      auto callees = kv.second;
      callees_map_[call].insert(callees.begin(), callees.end());
    }

    if (false) {
      for (auto kv : callees_map_) {
        std::cout << "[STA] Callees Map " << kv.first->op << " " << kv.second.size() << std::endl;
      }
    }
  }

  IRModule PerformAnalysis() {
    RunAnalysis();

    for (auto kv : var_states_) {
      if (IsScalarTensorType(kv.first->checked_type_)) {
        std::cout << "[GDFTR] Var " << kv.first->vid->name_hint << " "
                  << support::PrintVector(kv.second) << std::endl;
      }
    }

    return mod_;
  }

 private:
  void RunAnalysis() {
    auto main_func = Downcast<Function>(mod_->Lookup("main"));

    for (int i = 0; i < max_iterations_; ++i) {
      // std::cout << "[MPT] ITERATION " << i << std::endl;
      this->Reset();
      auto context_fn_key = main_func.get();
      on_stack_.insert(context_fn_key);
      function_stack_.push_back(main_func);
      this->VisitExpr(main_func->body, CreateStateForType(main_func->body->checked_type_, true));
      function_stack_.pop_back();
      on_stack_.erase(context_fn_key);
    }
  }

  void Reset() { on_stack_.clear(); }

  std::string GetFunctionName(const Function& fn) {
    return fn->GetAttr<String>("db.function_name", String("")).value();
  }

  std::vector<bool> Merge(const std::vector<std::vector<bool>>& vals) {
    std::vector<bool> res(vals[0]);
    for (size_t i = 1; i < vals.size(); ++i) {
      for (size_t j = 0; j < vals[i].size(); ++j) {
        res[j] = res[j] && vals[i][j];
      }
    }
    return res;
  }

  template <typename T, typename MapType>
  std::vector<bool> Add(MapType& map, const T& obj, const std::vector<bool>& to_add) {
    auto key = obj.get();
    auto it = map.find(key);
    if (it == map.end()) {
      map[key] = to_add;
      return to_add;
    } else {
      auto merged = Merge({it->second, to_add});
      map[key] = merged;
      return merged;
    }
  }

  std::vector<bool> Add(const Var& var, const std::vector<bool>& to_add,
                        const std::string& reason) {
    std::cout << "[STA]  Adding var " << var->vid->name_hint << " " << support::PrintVector(to_add)
              << std::endl;
    ICHECK_EQ(to_add.size(), GetTypeSize(var->checked_type()))
        << var << " " << support::PrintVector(to_add) << " " << reason;
    return Add<Var, VarStateMap>(var_states_, var, to_add);
  }

  std::vector<bool> Add(const Expr& tuple, const int index, const bool to_add) {
    auto key = tuple.get();
    auto it = tuple_states_.find(key);
    std::vector<bool> taint;
    if (it == tuple_states_.end()) {
      taint = CreateStateForType(tuple->checked_type_, true);
    } else {
      taint = it->second;
    }
    taint[index] = taint[index] && to_add;
    tuple_states_[key] = taint;
    return taint;
  }

  bool CollapseTaint(const TaintT& vals) {
    bool res = true;
    for (auto b : vals) {
      res = res && b;
    }
    return res;
  }

  TaintT ResizeTaint(const TaintT& taint, const Type& from, const Type& to) {
    size_t from_size = GetTypeSize(from);
    size_t to_size = GetTypeSize(to);
    if (from_size == to_size) {
      return taint;
    }
    return ConstructBoolArray(to_size, CollapseTaint(taint));
  }

  std::vector<bool> Add(const Function& fn, const std::vector<bool>& to_add) {
    ICHECK_EQ(to_add.size(), GetTypeSize(fn->ret_type))
        << fn << " " << support::PrintVector(to_add);
    return Add<Function, FunctionStateMap>(function_states_, fn, to_add);
  }

  std::vector<bool> GetOrCreateVarState(const Var& var) {
    auto key = var.get();
    auto it = var_states_.find(key);
    if (it == var_states_.end()) {
      return CreateStateForType(var->checked_type_, true);
    } else {
      return it->second;
    }
  }

  void VisitExpr_(const ConstantNode* op, std::vector<bool> taint) {}

  void VisitExpr_(const TupleNode* op, std::vector<bool> taint) {
    ICHECK_EQ(taint.size(), op->fields.size());
    for (size_t i = 0; i < op->fields.size(); ++i) {
      this->VisitExpr(op->fields[i], CreateStateForType(op->fields[i]->checked_type_, taint[i]));
    }
  }

  void VisitExpr_(const VarNode* op, std::vector<bool> taint) { Add(GetRef<Var>(op), taint, ""); }

  void VisitExpr_(const GlobalVarNode* op, std::vector<bool> taint) {}

  void VisitExpr_(const FunctionNode* op, std::vector<bool> taint) {}

  void VisitBody(const Function& fn, TaintT taint) {
    std::cout << "[STA] Visiting body of " << GetFunctionName(fn) << " "
              << support::PrintVector(taint) << std::endl;
    auto context_fn_key = fn.get();
    on_stack_.insert(context_fn_key);
    function_stack_.push_back(fn);
    this->VisitExpr(fn->body, taint);
    function_stack_.pop_back();
    on_stack_.erase(context_fn_key);
  }

  void VisitExpr_(const CallNode* op, std::vector<bool> taint) {
    auto on_device_props = GetOnDeviceProps(op);
    if (on_device_props.body.defined()) {
      return this->VisitExpr(on_device_props.body, taint);
    }
    std::cout << "[STA] Visiting call " << GetRef<Expr>(op) << std::endl;

    if (op->op == GetInvokeTVMOp() && IsOpOnScalars(op)) {
      // std::cout << "[STA]  Visiting op " << op->args[0] << " " << support::PrintVector(taint)
      // << std::endl;
      auto ins = op->args[1].as<TupleNode>();
      auto outs = op->args[2].as<TupleNode>();
      ICHECK(ins);
      ICHECK(outs);
      std::vector<std::vector<bool>> out_taints;
      for (auto out : outs->fields) {
        if (auto vn = out.as<VarNode>()) {
          auto out_taint = GetOrCreateVarState(GetRef<Var>(vn));
          // std::cout << "[STA]  O " << vn->vid->name_hint << " " <<
          // support::PrintVector(out_taint)
          // << std::endl;
          out_taints.push_back(out_taint);
        } else {
          ICHECK(false) << GetRef<Expr>(op);
        }
      }
      auto input_taint = CollapseTaint(Merge(out_taints));
      std::cout << "[STA]  I " << input_taint << std::endl;
      for (auto in : ins->fields) {
        this->VisitExpr(in, CreateStateForType(in->checked_type_, input_taint));
      }
    } else if (op->op.as<OpNode>()) {
      bool value = true;
      if (op->op == GetDBRandomUniformOp()) {
        value = false;
      }
      for (auto arg : op->args) {
        this->VisitExpr(arg, CreateStateForType(arg->checked_type_, value));
      }
    } else if (op->op.as<ConstructorNode>()) {
      auto collapsed = CollapseTaint(taint);
      for (auto arg : op->args) {
        this->VisitExpr(arg, CreateStateForType(arg->checked_type_, collapsed));
      }
    } else {
      auto it = callees_map_.find(op);
      ICHECK(it != callees_map_.end()) << GetRef<Expr>(op);
      auto callees = it->second;
      std::vector<std::vector<bool>> arg_taints(op->args.size());
      int ctr = 0;
      for (auto callee_node : callees) {
        auto callee = GetRef<Function>(callee_node);
        if (!on_stack_.count(callee.get())) {
          this->VisitBody(callee, ResizeTaint(taint, op->checked_type_, callee->ret_type));
        }

        for (size_t i = 0; i < callee->params.size(); ++i) {
          auto param_taint = GetOrCreateVarState(callee->params[i]);
          auto resized = ResizeTaint(param_taint, callee->params[i]->checked_type_,
                                     op->args[i]->checked_type_);
          if (ctr == 0) {
            arg_taints[i] = resized;
          } else {
            arg_taints[i] = Merge({arg_taints[i], resized});
          }
        }
        ctr++;
      }

      for (size_t i = 0; i < op->args.size(); ++i) {
        std::cout << "[STA]  Arg taint " << op->args[i] << " "
                  << support::PrintVector(arg_taints[i]) << std::endl;
        this->VisitExpr(op->args[i], arg_taints[i]);
      }
    }
  }

  void VisitExpr_(const LetNode* op, std::vector<bool> taint) {
    this->VisitExpr(op->body, taint);
    this->VisitExpr(op->value, GetOrCreateVarState(op->var));
  }

  void VisitExpr_(const IfNode* op, std::vector<bool> taint) {
    this->VisitExpr(op->true_branch, taint);
    this->VisitExpr(op->false_branch, taint);
    this->VisitExpr(op->cond, CreateStateForType(op->cond->checked_type_, false));
  }

  void VisitExpr_(const OpNode* op, std::vector<bool> taint) {}

  void VisitExpr_(const TupleGetItemNode* op, std::vector<bool> taint) {
    auto tuple_taint = Add(op->tuple, op->index, CollapseTaint(taint));
    this->VisitExpr(op->tuple, tuple_taint);
  }

  void VisitExpr_(const RefCreateNode* op, std::vector<bool> taint) { UnsupportedFunction(op); }

  void VisitExpr_(const RefReadNode* op, std::vector<bool> taint) {}

  void VisitExpr_(const RefWriteNode* op, std::vector<bool> taint) {}

  void UnsupportedFunction(const ExprNode* op) {
    ICHECK(false) << "Unsupported expression type " << GetRef<Expr>(op);
  }

  void VisitExpr_(const ConstructorNode* op, std::vector<bool> taint) {}

  void VisitExpr_(const MatchNode* op, std::vector<bool> taint) {
    // std::cout << "[STA]  Visiting match " << op->data << std::endl;
    std::vector<bool> pattern_var_taints;
    for (auto clause : op->clauses) {
      // std::cout << "[STA]   Visiting clause " << clause->lhs << std::endl;
      this->VisitExpr(clause->rhs, taint);
      auto pattern_vars = CollectPatternVars(clause->lhs);
      for (auto& var : pattern_vars) {
        pattern_var_taints.push_back(CollapseTaint(GetOrCreateVarState(var)));
      }
    }
    this->VisitExpr(op->data,
                    CreateStateForType(op->data->checked_type_, CollapseTaint(pattern_var_taints)));
  }

  IRModule& mod_;

  VarStateMap var_states_;
  TupleStateMap tuple_states_;
  FunctionStateMap function_states_;
  std::unordered_map<const FunctionNode*, VarSet> free_vars_map_;
  OnStackSet on_stack_;

  std::vector<Function> function_stack_;
  std::unordered_map<const CallNode*, std::unordered_set<const FunctionNode*>> callees_map_;
  int max_iterations_{2};
};

Function MarkScalarCallsInFunc(const Function& func) {
  class Marker : public ExprMutator {
   public:
    Expr VisitExpr_(const CallNode* op) {
      auto on_device_props = GetOnDeviceProps(op);
      if (on_device_props.body.defined()) {
        return VisitExpr(on_device_props.body);
      } else if (op->op == GetDBRandomUniformOp()) {
        return ExprMutator::VisitExpr_(op);
      }
      auto mutated = ExprMutator::VisitExpr_(op);
      auto mutated_node = mutated.as<CallNode>();
      auto callee_op_node = op->op.as<OpNode>();
      if (callee_op_node && IsOpOnScalars(mutated_node)) {
        // std::cout << "[SCA] Marking scalar op " << mutated << std::endl;
        auto new_attrs =
            DictAttrs::WithAttr(mutated_node->attrs, tir::attr::kDBScalarCall, mutated);
        auto ret = Call(mutated_node->op, mutated_node->args, new_attrs, mutated_node->type_args,
                        mutated_node->span);
        ret->checked_type_ = op->checked_type_;
        return ret;
      }
      return mutated;
    }
  };

  Marker marker;
  auto mutated = Function(func->params, marker(func->body), func->ret_type, func->type_params,
                          func->attrs, func->span);
  mutated->checked_type_ = func->checked_type_;
  return mutated;
}

}  // namespace

namespace transform {

Pass MarkScalarCalls() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return MarkScalarCallsInFunc(f); };
  auto func_pass = CreateFunctionPass(pass_func, 0, "MarkScalarCallsFunc", {"InferType"});
  return transform::Sequential({func_pass, InferType()}, "MarkScalarCalls");
}

TVM_REGISTER_GLOBAL("relay._transform.MarkScalarCalls").set_body_typed(MarkScalarCalls);

Pass MarkScalarVars() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule m,
                                                                            PassContext pc) {
    ScalarTaintAnalysis analysis(m, GetCalleesMap(m));
    analysis.PerformAnalysis();
    return m;
  };
  return CreateModulePass(pass_func, 0, "MarkScalarVars", {});
}

TVM_REGISTER_GLOBAL("relay._transform.MarkScalarVars").set_body_typed(MarkScalarVars);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
