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
 * \file simplify.cc
 * \brief Statement simplifier based on analyzer
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

#include "../../arith/const_fold.h"
#include "../../arith/ir_mutator_with_analyzer.h"

namespace tvm {
namespace arith {

using namespace tir;

class VarExtentCollector : public StmtVisitor {
 public:
  void VisitStmt_(const ForNode* op) final {
    HandleExtent(op->loop_var, op->min, op->extent);
    StmtVisitor::VisitStmt(op->body);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      HandleExtent(Downcast<IterVar>(op->node)->var, 0, op->value);
    }
    StmtVisitor::VisitStmt(op->body);
  }

  void HandleExtent(Var v, PrimExpr min, PrimExpr extent) {
    if (range_map_.count(v.get())) {
      Range r = range_map_.at(v.get());
      CHECK(StructuralEqual()(r->min, min) && StructuralEqual()(r->extent, extent))
          << "Reused variable " << v << " with different ranges " << r << " (" << min << ", "
          << extent << ")";
    } else {
      range_map_[v.get()] = Range::FromMinExtent(min, extent);
    }
  }

  std::unordered_map<const Object*, Range> range_map_;
};

class LoadContainer : public ExprVisitor {
 public:
  bool ContainsLoad(const PrimExpr& e) {
    contains_load_ = false;
    VisitExpr(e);
    return contains_load_;
  }

 private:
  void VisitExpr_(const LoadNode* op) {
    contains_load_ = true;
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ScatterLoadNode* op) {
    contains_load_ = true;
    ExprVisitor::VisitExpr_(op);
  }

  bool contains_load_ = false;
};

class StmtSimplifier : public IRMutatorWithAnalyzer {
 public:
  explicit StmtSimplifier(Analyzer* analyzer, bool extra_simplify)
      : IRMutatorWithAnalyzer(analyzer), extra_simplify_(extra_simplify) {}

  using Parent = IRMutatorWithAnalyzer;
  using Parent::VisitStmt;
  using Parent::VisitStmt_;

  PrimExpr VisitExpr(const PrimExpr& expr_) final {
    PrimExpr expr = Parent::VisitExpr(expr_);
    if (LoadContainer().ContainsLoad(expr) || !extra_simplify_) {
      return analyzer_->Simplify(expr);
    }
    bool print = false;  //(expr.as<FloorDivNode>());
    if (print) std::cout << "[SIMPL] Expr " << expr << std::endl;
    std::unordered_map<const VarNode*, arith::IntSet> relaxable;
    for (auto var : CollectVars(expr)) {
      if (print)
        std::cout << "[SIMPL]  Var " << var->name_hint << " "
                  << extent_collector_.range_map_.count(var) << std::endl;
      if (extent_collector_.range_map_.count(var)) {
        Range r = extent_collector_.range_map_.at(var);
        if (analyzer_->CanProveGreaterEqual(r->min, 0)) {
          relaxable[var] = arith::IntSet::FromRange(r);
          if (print) std::cout << "[SIMPL]   Relaxable " << var->name_hint << " " << r << std::endl;
        }
      }
    }

    if (relaxable.size() > 0) {
      PrimExpr simplified = expr;
      for (auto kv : relaxable) {
        auto var = kv.first;
        auto range = kv.second;
        if (print) {
          std::cout << "[SIMPL]  Try relaxing " << var->name_hint << " " << range << std::endl;
        }

        IntSet set = EvalSet(simplified, {{var, range}});
        PrimExpr min_expr = analyzer_->Simplify(set.min());
        PrimExpr max_expr = analyzer_->Simplify(set.max());
        PrimExpr res_expr = analyzer_->Simplify(max_expr - min_expr);
        if (print) {
          std::cout << "[SIMPL]     Expr " << simplified << std::endl;
          std::cout << "[SIMPL]      Min " << min_expr << std::endl;
          std::cout << "[SIMPL]      Max " << max_expr << std::endl;
          std::cout << "[SIMPL]      Res " << res_expr << std::endl;
        }

        if (!(is_pos_inf(min_expr) || is_neg_inf(min_expr) || is_pos_inf(min_expr) ||
              is_neg_inf(min_expr)) &&
            analyzer_->CanProve(res_expr == 0)) {
          simplified = min_expr;
        }
      }
      return analyzer_->Simplify(simplified);
    }
    return analyzer_->Simplify(expr);
  }

  Stmt Simplify(Stmt stmt) {
    extent_collector_(stmt);
    return operator()(std::move(stmt));
  }

  Stmt VisitStmt_(const ForNode* op) final {
    analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    With<ConstraintContext> ctx1(analyzer_, op->loop_var >= op->min);
    With<ConstraintContext> ctx2(analyzer_, op->loop_var < op->min + op->extent);
    return Parent::VisitStmt_(op);
  }

  bool CanInlineLetStmt(const LetStmtNode* op) {
    if (is_const_number(op->value)) return true;
    if (op->value.as<VarNode>()) return true;
    // Won't face the deep expression explosion problem as in Let expression.
    // attempt to inline as much as possible if the value integer type(can be index).
    if (!op->value.dtype().is_int()) return false;
    return SideEffect(op->value) <= CallEffectKind::kPure;
  }

  Stmt VisitStmt_(const LetStmtNode* op) {
    PrimExpr value = this->VisitExpr(op->value);
    if (CanInlineLetStmt(op)) {
      // it is fine to discard the let binding
      // because the call to simplify will always inline the var.
      analyzer_->Bind(op->var, value);
      return this->VisitStmt(op->body);
    }
    Stmt body = this->VisitStmt(op->body);
    if (value.same_as(op->value) && body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = this->CopyOnWrite(op);
      n->value = std::move(value);
      n->body = std::move(body);
      return Stmt(n);
    }
  }

  // eliminate useless stores
  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = Parent::VisitStmt_(op);
    op = stmt.as<StoreNode>();
    if (const LoadNode* load = op->value.as<LoadNode>()) {
      if (load->buffer_var.same_as(op->buffer_var) &&
          tir::ExprDeepEqual()(load->index, op->index)) {
        return Evaluate(0);
      }
    }
    return GetRef<Stmt>(op);
  }

  VarExtentCollector extent_collector_;
  bool extra_simplify_;
};

}  // namespace arith

namespace tir {
namespace transform {

Pass Simplify(bool extra_simplify) {
  auto pass_func = [extra_simplify](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    arith::Analyzer analyzer;
    // std::cout << "SIMPLIFYING \n" << n->body << " " << extra_simplify << std::endl;
    n->body = arith::StmtSimplifier(&analyzer, extra_simplify).Simplify(std::move(n->body));
    // std::cout << "AFTER SIMPLIFICATION \n" << n->body << std::endl;
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.Simplify", {});
}

TVM_REGISTER_GLOBAL("tir.transform.Simplify").set_body_typed(Simplify);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
