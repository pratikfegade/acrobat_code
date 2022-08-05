
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
#include <tvm/ir/transform.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/vm/dynamic_batching.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include "../../printer/text_printer.h"
#include "../../support/utils.h"
#include "../analysis/call_graph.h"
#include "../op/db/db_ops.h"
#include "../op/memory/memory.h"
#include "../op/vm/vm.h"
#include "./expr_subst.h"
#include "./function_pointer_analysis.h"
#include "./map_set.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

namespace tvm {
namespace relay {

namespace {

using NodeT = const FunctionNode*;

class SelfPrimCallCounter : public ExprFunctor<double(const Expr& n)> {
 public:
  SelfPrimCallCounter() : look_at_callees_(false) {}

  SelfPrimCallCounter(bool look_at_callees, const IRModule& mod,
                      const std::unordered_map<const FunctionNode*, double> all_calls)
      : look_at_callees_(look_at_callees), mod_(mod), all_calls_(all_calls) {}

  double VisitExpr_(const ConstantNode* op) override { return 0; }

  double VisitExpr_(const TupleNode* op) override {
    double res = 0;
    for (auto field : op->fields) {
      res += this->VisitExpr(field);
    }
    return res;
  }

  double VisitExpr_(const VarNode* op) override { return 0; }

  double VisitExpr_(const GlobalVarNode* op) override { return 0; }

  double VisitExpr_(const FunctionNode* op) override { return 0; }

  double VisitExpr_(const CallNode* op) override {
    int res = this->VisitExpr(op->op);
    for (auto arg : op->args) {
      res += this->VisitExpr(arg);
    }
    if (op->op == invoke_tvm_op_ && !Downcast<DictAttrs>(op->attrs)
                                         .GetAttr(tir::attr::kDBScalarCall, NullValue<Expr>())
                                         .defined()) {
      return res + 1;
    } else if (look_at_callees_ && op->op.as<GlobalVarNode>()) {
      auto base_func = mod_->Lookup(Downcast<GlobalVar>(op->op));
      if (auto fn = base_func.as<FunctionNode>()) {
        res += all_calls_.at(fn);
      }
    }
    return res;
  }

  double VisitExpr_(const LetNode* op) override {
    return this->VisitExpr(op->value) + this->VisitExpr(op->body);
  }

  double VisitExpr_(const IfNode* op) override {
    auto then_case = this->VisitExpr(op->true_branch);
    auto else_case = this->VisitExpr(op->false_branch);
    if (then_case == else_case) {
      return then_case + this->VisitExpr(op->cond);
    } else {
      return std::numeric_limits<double>::infinity();
    }
  }

  double VisitExpr_(const OpNode* op) override { return 0; }

  double VisitExpr_(const TupleGetItemNode* op) override { return this->VisitExpr(op->tuple); }

  double VisitExpr_(const RefCreateNode* op) override {
    ICHECK(false);
    return 0;
  }

  double VisitExpr_(const RefReadNode* op) override {
    ICHECK(false);
    return 0;
  }

  double VisitExpr_(const RefWriteNode* op) override {
    ICHECK(false);
    return 0;
  }

  double VisitExpr_(const ConstructorNode* op) override { return 0; }

  double VisitExpr_(const MatchNode* op) override {
    std::vector<double> calls;
    for (auto clause : op->clauses) {
      calls.push_back(this->VisitExpr(clause->rhs));
    }
    for (auto call : calls) {
      if (call != calls[0]) {
        return std::numeric_limits<double>::infinity();
      }
    }
    return calls[0];
  }

  bool look_at_callees_ = false;
  const IRModule mod_;
  const std::unordered_map<const FunctionNode*, double> all_calls_;

  const Op invoke_tvm_op_ = Op::Get("vm.invoke_tvm_op");
};

double CountTransitiveCalls(const Function& func,
                            const std::unordered_map<const FunctionNode*, double>& self_calls,
                            const IRModule& mod, std::unordered_set<const Object*>& on_stack) {
  if (on_stack.count(func.get())) {
    return std::numeric_limits<double>::infinity();
  } else {
    on_stack.insert(func.get());
    double res = self_calls.at(func.get());
    PostOrderVisit(func->body, [&](const Expr& e) {
      if (auto op = e.as<CallNode>()) {
        if (op->op.as<GlobalVarNode>()) {
          auto callee = mod->Lookup(Downcast<GlobalVar>(op->op));
          if (callee.as<FunctionNode>()) {
            res += CountTransitiveCalls(Downcast<Function>(callee), self_calls, mod, on_stack);
          }
        }
      }
    });
    return res;
  }
}

class GhostOpAdder : public ExprMutator {
 public:
  GhostOpAdder(const IRModule& mod, const std::unordered_map<const FunctionNode*, double> all_calls)
      : mod_(mod), all_calls_(all_calls) {}

  Function AddGhostOps(const Function& func) {
    auto body = this->VisitExpr(func->body);
    if (body.same_as(func->body)) {
      return func;
    } else {
      auto res =
          Function(func->params, body, func->ret_type, func->type_params, func->attrs, func->span);
      res->checked_type_ = func->checked_type_;
      return res;
    }
  }

 private:
  Expr VisitExpr_(const IfNode* op) override {
    auto add_ghost_ops = [](const Expr& e, int num) {
      auto current = e;
      for (int i = 0; i < num; ++i) {
        current = Let(Var("tmp" + std::to_string(i), VoidType()), MakeDBGhostOp(), current);
      }
      return current;
    };

    const Op invoke_tvm_op = Op::Get("vm.invoke_tvm_op");
    // std::cout << "[GH]    Visiting " << PrettyPrint(op->cond) << std::endl;
    double then_calls = SelfPrimCallCounter(true, mod_, all_calls_)(op->true_branch);
    double else_calls = SelfPrimCallCounter(true, mod_, all_calls_)(op->false_branch);

    Expr new_true = this->VisitExpr(op->true_branch);
    Expr new_false = this->VisitExpr(op->false_branch);
    Expr new_condition = this->VisitExpr(op->cond);
    if (0 <= then_calls && then_calls < MAX_GHOST_OPS_ && 0 <= else_calls &&
        else_calls < MAX_GHOST_OPS_ && then_calls != else_calls) {
      // std::cout << "[GH]     Adding " << then_calls << " " << else_calls << std::endl;
      if (then_calls > else_calls) {
        new_false = add_ghost_ops(new_false, then_calls - else_calls);
      } else if (then_calls < else_calls) {
        new_true = add_ghost_ops(new_true, else_calls - then_calls);
      }
    }
    auto res = If(new_condition, new_true, new_false, op->span);
    res->checked_type_ = op->checked_type_;
    return res;
  }

  static constexpr int MAX_GHOST_OPS_ = 5;
  const IRModule& mod_;
  const std::unordered_map<const FunctionNode*, double> all_calls_;
};

}  // namespace

IRModule GhostOps(IRModule& mod) {
  // std::cout << "GHOSTING============================================================== "
  // << std::endl;
  CallGraph cg(mod);

  auto funcs = mod->functions;

  std::unordered_map<const BaseFuncNode*, std::string> func_name_map;
  std::unordered_map<const FunctionNode*, double> self_calls;
  for (const auto& it : funcs) {
    if (const auto* n = it.second.as<FunctionNode>()) {
      func_name_map[it.second.get()] = it.first->name_hint;
      ICHECK_EQ(FreeVars(it.second).size(), 0);
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
      Function func = GetRef<Function>(n);

      self_calls[func.get()] = SelfPrimCallCounter()(func->body);
      // std::cout << "[GH] Looking at " << it.first->name_hint << " " << self_calls[func.get()]
      // << std::endl;
    }
  }

  std::unordered_map<const FunctionNode*, double> all_calls;
  for (const auto& it : funcs) {
    if (const auto* n = it.second.as<FunctionNode>()) {
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
      Function func = GetRef<Function>(n);

      std::unordered_set<const Object*> on_stack;
      all_calls[func.get()] = CountTransitiveCalls(func, self_calls, mod, on_stack);

      // std::cout << "[GH] All Looking at " << it.first->name_hint << " " << all_calls[func.get()]
      // << std::endl;
    }
  }

  tvm::Map<GlobalVar, Function> updates;
  for (const auto& it : funcs) {
    if (const auto* n = it.second.as<FunctionNode>()) {
      if (n->GetAttr<String>(attr::kCompiler).defined()) continue;
      std::cout << "[GH] Adding ghost ops to " << it.first->name_hint << std::endl;
      Function func = GetRef<Function>(n);
      updates.Set(it.first, GhostOpAdder(mod, all_calls).AddGhostOps(func));
    }
  }

  for (auto pair : updates) {
    mod->Add(pair.first, pair.second, true);
  }

  return mod;
}

namespace transform {
Pass AddGhostOpsInIfElse() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return GhostOps(m); };
  return CreateModulePass(pass_func, 0, "AddGhostOpsInIfElse", {});
}

TVM_REGISTER_GLOBAL("relay._transform.AddGhostOpsInIfElse").set_body_typed(AddGhostOpsInIfElse);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
