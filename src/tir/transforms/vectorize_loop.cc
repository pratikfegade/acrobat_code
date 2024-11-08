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
 * \file vectorize_loop.cc
 */
// Loop vectorizer as in Halide pipeline.
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ir_utils.h"

namespace tvm {
namespace tir {

inline PrimExpr BroadcastTo(PrimExpr e, int lanes) {
  if (e.dtype().lanes() == lanes) return e;
  if (const BroadcastNode* op = e.as<BroadcastNode>()) {
    if (lanes % op->lanes == 0) {
      return Broadcast(op->value, lanes);
    }
  }
  ICHECK_EQ(e.dtype().lanes(), 1) << "Cannot broadcast lane=" << e.dtype().lanes() << " to "
                                  << lanes;
  return Broadcast(e, lanes);
}

// Rewrite vectorized allocation access
// This is necessary for making each vector component containing its own workspace.
// Originates from Halide's loop vectorizer
//
// s[i] = s[i * lanes + var]
//
// The same principle applies when using one thread to simulate multiple context.
//
class VecAllocAccess : public StmtExprMutator {
 public:
  VecAllocAccess(const VarNode* buf, Var var, int var_lanes)
      : buf_(buf), var_(var), var_lanes_(var_lanes) {}
  // Load
  PrimExpr VisitExpr_(const LoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<LoadNode>();
    if (op->buffer_var.get() == buf_) {
      return Load(op->dtype, op->buffer_var, op->index * var_lanes_ + var_, op->predicate);
    } else {
      return expr;
    }
  }
  // Store
  Stmt VisitStmt_(const StoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<StoreNode>();
    if (op->buffer_var.get() == buf_) {
      return Store(op->buffer_var, op->value, op->index * var_lanes_ + var_, op->predicate);
    } else {
      return stmt;
    }
  }

 private:
  // buffer var
  const VarNode* buf_;
  // variable to be replaced
  Var var_;
  // the lanes.
  int var_lanes_;
};

// <DietCode>
//
// The class that performs local padding.
namespace {

/**
 * @brief Check whether can perform local padding on the if-condition. This is
 *        done by testing whether the condition's only dependency is
 *        threadIdx.x. If tested true, then local padding is NOT allowed.
 */
bool CanLocalPad(const PrimExpr& cond) {
  class Visitor : public ExprVisitor {
   public:
    void VisitExpr_(const VarNode* op) final {
      if (IsCUDAThreadIdx(op->name_hint)) {
        cuda_thread_visited_ = true;
      } else if (IsCUDABlockIdx(op->name_hint)) {
        cuda_block_visited_ = false;
      }
    }

    bool cuda_thread_visited_{false};
    bool cuda_block_visited_{false};
  };

  Visitor visitor;
  visitor(cond);
  return !(visitor.cuda_thread_visited_ && !visitor.cuda_block_visited_);
}

class LocalPadder : public StmtExprMutator {
 private:
  // As is commented in the `MakeBoundCheck` function call, we have to *inline*
  // predicates as part of the StoreNode's for local padding, where by *inline*
  // we mean the following:
  //
  //     if (cond) X_shared[...] = X[...]; ⇒ X_shared[...] = cond ? X[...] : 0;
  //
  // However, note that we cannot directly inline all the predicates, for
  // example, if the predicates are in the format of
  //
  //     if (threadIdx.x < ...)
  //
  // then we cannot inline it, as doing so would cause a thread to overwrite the
  // shared memory blocks that belong to others (similarly for other formats
  // that only depend on threadIdx.x). For predicates that do not fall into this
  // category, we put them on a stack and inline them upon seeing a StoreNode.
  Stmt VisitStmt_(const IfThenElseNode* op) final {
    if (!CanLocalPad(op->condition) || !is_no_op(op->else_case)) {
      if (op->else_case.defined()) {
        return IfThenElse(op->condition, VisitStmt(op->then_case), VisitStmt(op->else_case));
      } else {
        return IfThenElse(op->condition, VisitStmt(op->then_case));
      }
    }
    predicate_stack_.push_back(op->condition);
    Stmt body_stmt_with_inlined_predicates = VisitStmt(op->then_case);
    predicate_stack_.pop_back();
    return body_stmt_with_inlined_predicates;
  }
  Stmt VisitStmt_(const StoreNode* op) final {
    // merge all the predicates on stack
    PrimExpr merged_predicate;
    for (PrimExpr pred : predicate_stack_) {
      if (!merged_predicate.defined()) {
        merged_predicate = pred;
      } else {
        merged_predicate = merged_predicate && pred;
      }
    }
    if (!merged_predicate.defined()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    ICHECK(op->value.dtype().is_float())
        << "The current implementation assumes that the data type is FP32";
    return Store(op->buffer_var, Select(merged_predicate, op->value, PrimExpr(0.f)), op->index,
                 op->predicate);
  }
  PrimExpr VisitExpr_(const VarNode* op) final { return StmtExprMutator::VisitExpr_(op); }

 private:
  bool substitute_mode_ = false;
  arith::Analyzer analyzer;
  // predicate stack, constructed by traversing down the if-nest
  std::vector<PrimExpr> predicate_stack_;
};
}  // anonymous namespace

// We use ExprFunctor directly instead of StmtExprMutator
// This is because the transformation can change the dtype of the Expr
// The existing ExprMutator transformation rules may not be well defined.
class Vectorizer : public StmtMutator, public ExprFunctor<PrimExpr(const PrimExpr&)> {
 public:
  using ExprFunctor::VisitExpr;
  using StmtMutator::operator();

  Vectorizer(Var var, int var_lanes) : var_(var), var_lanes_(var_lanes) {
    if (var_lanes == 1) {
      ramp_ = Integer(0);
    } else {
      ramp_ = Ramp(0, 1, var_lanes);
    }
    // ramp_ = Ramp(0, 1, var_lanes);
  }

  Stmt VisitStmt(const Stmt& stmt) final {
    // std::cout << "Vectorizing " << var_->name_hint << std::endl;
    ICHECK(!need_scalarize_);
    Stmt ret = StmtMutator::VisitStmt(stmt);
    if (need_scalarize_) {
      need_scalarize_ = false;
      // std::cout << "  Scalarizing because need scalarizing" << std::endl;
      return Scalarize(stmt);
    } else {
      return ret;
    }
  }

  PrimExpr VisitExpr(const PrimExpr& e) final { return ExprFunctor::VisitExpr(e); }

  PrimExpr VisitExpr_(const AddNode* op) final {
    return AddSubVec(op, [](PrimExpr a, PrimExpr b) { return a + b; });
  }

  PrimExpr VisitExpr_(const SubNode* op) final {
    return AddSubVec(op, [](PrimExpr a, PrimExpr b) { return a - b; });
  }

  PrimExpr VisitExpr_(const MulNode* op) final {
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) && b.same_as(op->b)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(a.dtype().lanes(), b.dtype().lanes());
      if (lanes != 1) {
        const RampNode* b_ramp = b.as<RampNode>();
        const RampNode* a_ramp = a.as<RampNode>();
        if (a_ramp && b.dtype().lanes() == 1 && analyzer_.CanProve(b > 0)) {
          return Ramp(a_ramp->base * b, a_ramp->stride * b, a_ramp->lanes);
        }
        if (b_ramp && a.dtype().lanes() == 1 && analyzer_.CanProve(a > 0)) {
          return Ramp(b_ramp->base * a, b_ramp->stride * a, b_ramp->lanes);
        }
      }
      return Mul(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
    return BinaryVec<Mul>(op);
  }
  PrimExpr VisitExpr_(const DivNode* op) final { return BinaryVec<Div>(op); }
  PrimExpr VisitExpr_(const ModNode* op) final { return BinaryVec<Mod>(op); }
  PrimExpr VisitExpr_(const FloorDivNode* op) final { return BinaryVec<FloorDiv>(op); }
  PrimExpr VisitExpr_(const FloorModNode* op) final { return BinaryVec<FloorMod>(op); }
  PrimExpr VisitExpr_(const MinNode* op) final { return BinaryVec<Min>(op); }
  PrimExpr VisitExpr_(const MaxNode* op) final { return BinaryVec<Max>(op); }
  PrimExpr VisitExpr_(const EQNode* op) final { return BinaryVec<EQ>(op); }
  PrimExpr VisitExpr_(const NENode* op) final { return BinaryVec<NE>(op); }
  PrimExpr VisitExpr_(const LTNode* op) final { return BinaryVec<LT>(op); }
  PrimExpr VisitExpr_(const LENode* op) final { return BinaryVec<LE>(op); }
  PrimExpr VisitExpr_(const GTNode* op) final { return BinaryVec<GT>(op); }
  PrimExpr VisitExpr_(const GENode* op) final { return BinaryVec<GE>(op); }
  PrimExpr VisitExpr_(const AndNode* op) final { return BinaryVec<And>(op); }
  PrimExpr VisitExpr_(const OrNode* op) final { return BinaryVec<Or>(op); }

  PrimExpr VisitExpr_(const NotNode* op) final {
    PrimExpr a = this->VisitExpr(op->a);
    if (a.same_as(op->a)) {
      return GetRef<PrimExpr>(op);
    } else {
      return !(a);
    }
  }

  PrimExpr VisitExpr_(const RampNode* op) final {
    PrimExpr base = this->VisitExpr(op->base);
    PrimExpr stride = this->VisitExpr(op->stride);
    if (base.dtype().lanes() > 1 && stride.dtype().lanes() == 1) {
      const RampNode* base_ramp = base.as<RampNode>();
      if (analyzer_.CanProve(base_ramp->stride == stride * make_const(stride.dtype(), op->lanes))) {
        return Ramp(base_ramp->base, stride, op->lanes * base_ramp->lanes);
      }
    }
    int lanes = std::max(base.dtype().lanes(), stride.dtype().lanes());
    base = BroadcastTo(base, lanes);
    stride = BroadcastTo(stride, lanes);
    Array<PrimExpr> elems;
    for (int i = 0; i < lanes; ++i) {
      elems.push_back(
          Ramp(Shuffle::ExtractElement(base, i), Shuffle::ExtractElement(stride, i), op->lanes));
    }
    return Shuffle::Concat(elems);
  }

  PrimExpr VisitExpr_(const BroadcastNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    if (value.dtype().lanes() != 1) {
      // std::cout << "  Broadcast scalarize" << std::endl;
      need_scalarize_ = true;
      return GetRef<PrimExpr>(op);
    }
    if (value.same_as(op->value)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Broadcast(op->value, op->lanes);
    }
  }

  PrimExpr VisitExpr_(const SelectNode* op) final {
    PrimExpr cond = this->VisitExpr(op->condition);
    PrimExpr t = this->VisitExpr(op->true_value);
    PrimExpr f = this->VisitExpr(op->false_value);
    if (cond.same_as(op->condition) && t.same_as(op->true_value) && f.same_as(op->false_value)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(std::max(cond.dtype().lanes(), t.dtype().lanes()), f.dtype().lanes());
      return Select(cond, BroadcastTo(t, lanes), BroadcastTo(f, lanes));
    }
  }

  PrimExpr VisitExpr_(const CastNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    if (value.same_as(op->value)) {
      return GetRef<PrimExpr>(op);
    } else {
      return Cast(op->dtype.with_lanes(value.dtype().lanes()), value);
    }
  }

  PrimExpr VisitExpr_(const FloatImmNode* op) final { return GetRef<PrimExpr>(op); }

  PrimExpr VisitExpr_(const IntImmNode* op) final { return GetRef<PrimExpr>(op); }

  PrimExpr VisitExpr_(const StringImmNode* op) final { return GetRef<PrimExpr>(op); }

  // Variable
  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);

    if (var.same_as(var_)) {
      return ramp_;
    }
    auto it = let_binding_.find(var);
    if (it != let_binding_.end()) {
      return it->second;
    } else {
      return std::move(var);
    }
  }
  // IfThenElse expr
  PrimExpr MutateIfThenElseExpr_(const CallNode* op) {
    PrimExpr cond = this->VisitExpr(op->args[0]);
    if (cond.dtype().is_vector()) {
      // std::cout << "  ITE scalarize" << std::endl;
      need_scalarize_ = true;
      return GetRef<PrimExpr>(op);
    }
    PrimExpr t = this->VisitExpr(op->args[1]);
    PrimExpr f = this->VisitExpr(op->args[2]);
    if (cond.same_as(op->args[0]) && t.same_as(op->args[1]) && f.same_as(op->args[2])) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(t.dtype().lanes(), f.dtype().lanes());
      t = BroadcastTo(t, lanes);
      f = BroadcastTo(f, lanes);
      return Call(op->dtype.with_lanes(lanes), op->op, {cond, t, f});
    }
  }
  // Call
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::if_then_else())) {
      return MutateIfThenElseExpr_(op);
    } else if (op->op.same_as(builtin::texture2d_load())) {
      int lane = 0;
      Array<PrimExpr> fcd = MutateArray({op->args.back()}, &lane);
      auto new_args = op->args;
      new_args.pop_back();
      new_args.push_back(fcd[0]);
      return Call(op->dtype.with_lanes(4), op->op, new_args);
    } else if (op->op.same_as(builtin::texture2d_store())) {
      int lane = 0;
      // Vectorize the value to store
      Array<PrimExpr> value{op->args.back()};
      Array<PrimExpr> mutated_value = MutateArray(value, &lane);
      Array<PrimExpr> new_args{op->args[0], op->args[1], op->args[2], mutated_value[0]};
      return Call(op->dtype.with_lanes(lane), op->op, new_args);
    }
    auto* op_ptr = op->op.as<OpNode>();
    bool vectorizable = op_ptr && op_vectorizable_.get(GetRef<Op>(op_ptr), false);

    if (!vectorizable) {
      // Cannot vectorize this op
      Array<PrimExpr> new_args;
      for (auto arg : op->args) {
        auto new_arg = this->VisitExpr(arg);
        if (new_arg.dtype().is_vector()) {
          // std::cout << "  Call scalarize" << std::endl;
          need_scalarize_ = true;
          return GetRef<PrimExpr>(op);
        }
        new_args.push_back(new_arg);
      }
      if (op->args.same_as(new_args)) {
        return GetRef<PrimExpr>(op);
      } else {
        return Call(op->dtype, op->op, new_args);
      }
    } else {
      int lane = 0;
      Array<PrimExpr> new_args = MutateArray(op->args, &lane);
      // normal code path.
      if (op->args.same_as(new_args)) {
        return GetRef<PrimExpr>(op);
      } else {
        return Call(op->dtype.with_lanes(lane), op->op, new_args);
      }
    }
  }
  // Load
  PrimExpr VisitExpr_(const LoadNode* op) final {
    PrimExpr index = this->VisitExpr(op->index);
    PrimExpr pred = this->VisitExpr(op->predicate);

    PrimExpr scatter_elem_index;
    PrimExpr scatter_batch_index;
    if (op->scatter_buffer_var.defined()) {
      scatter_elem_index = this->VisitExpr(op->scatter_elem_index);
      scatter_batch_index = this->VisitExpr(op->scatter_batch_index);
    }

    if (index.same_as(op->index) && pred.same_as(op->predicate) &&
        scatter_batch_index.same_as(op->scatter_batch_index) &&
        scatter_elem_index.same_as(op->scatter_elem_index)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(index.dtype().lanes(), pred.dtype().lanes());
      if (op->scatter_buffer_var.defined()) {
        return Load(op->dtype.with_lanes(lanes), op->buffer_var, BroadcastTo(index, lanes),
                    BroadcastTo(pred, lanes), op->scatter_buffer_var, scatter_batch_index,
                    BroadcastTo(scatter_elem_index, lanes));
      } else {
        return Load(op->dtype.with_lanes(lanes), op->buffer_var, BroadcastTo(index, lanes),
                    BroadcastTo(pred, lanes));
      }
    }
  }
  // Let
  PrimExpr VisitExpr_(const LetNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    // Weaker SSA condition
    // A single var can be binded in multiple lets
    // but they have to bind to the same value.
    // This is used to allow cases when we reuse a single let
    // expression to cosntruct a nested expr.
    // (let x = 1 in x + 1) * (let x = 1 in x + 1)
    auto it = let_binding_.find(op->var);
    if (it != let_binding_.end()) {
      ICHECK(deep_equal_(it->second, value))
          << "Let cannot bind the same var to two different values";
    }
    if (value.dtype().lanes() != op->value.dtype().lanes()) {
      Var new_var(op->var->name_hint, value.dtype());
      let_binding_[op->var] = new_var;
      return Let(new_var, value, this->VisitExpr(op->body));
    } else {
      let_binding_[op->var] = op->var;
      PrimExpr body = this->VisitExpr(op->body);
      if (value.same_as(op->value) && body.same_as(op->body)) {
        return GetRef<PrimExpr>(op);
      } else {
        return Let(op->var, value, body);
      }
    }
  }
  // Store
  Stmt VisitStmt_(const StoreNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    PrimExpr index = this->VisitExpr(op->index);
    PrimExpr pred = this->VisitExpr(op->predicate);

    PrimExpr scatter_elem_index;
    PrimExpr scatter_batch_index;
    if (op->scatter_buffer_var.defined()) {
      scatter_elem_index = this->VisitExpr(op->scatter_elem_index);
      scatter_batch_index = this->VisitExpr(op->scatter_batch_index);
    }

    if (value.same_as(op->value) && index.same_as(op->index) &&
        scatter_batch_index.same_as(op->scatter_batch_index) &&
        scatter_elem_index.same_as(op->scatter_elem_index)) {
      return GetRef<Stmt>(op);
    } else {
      int lanes = std::max(value.dtype().lanes(), index.dtype().lanes());
      lanes = std::max(lanes, pred.dtype().lanes());
      if (op->scatter_buffer_var.defined()) {
        return Store(op->buffer_var, BroadcastTo(value, lanes), BroadcastTo(index, lanes),
                     BroadcastTo(pred, lanes), op->scatter_buffer_var, scatter_batch_index,
                     BroadcastTo(scatter_elem_index, lanes));
      } else {
        return Store(op->buffer_var, BroadcastTo(value, lanes), BroadcastTo(index, lanes),
                     BroadcastTo(pred, lanes));
      }
    }
  }
  // For
  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kVectorized) {
      LOG(WARNING) << "Detect vectorize inside vectorized loop, ignoring...";
    }
    ICHECK(is_zero(op->min));
    ICHECK(!op->extent.dtype().is_vector());
    PrimExpr extent = this->VisitExpr(op->extent);
    if (extent.dtype().is_vector()) {
      return Scalarize(GetRef<Stmt>(op));
    }
    Stmt body = this->VisitStmt(op->body);
    if (extent.same_as(op->extent) && body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      return For(op->loop_var, op->min, extent, op->kind, body, op->thread_binding,
                 op->annotations);
    }
  }
  // IfThenElse
  Stmt VisitStmt_(const IfThenElseNode* op) final {
    ICHECK(!op->condition.dtype().is_vector());
    PrimExpr condition = this->VisitExpr(op->condition);

    // <DietCode>
    //
    // Do NOT perform vectorization load in the case of local padding. The
    // reason is two-fold:
    //
    // - Our evaluations show that vectorized loads are NOT able to
    //   significantly improve the performance of the generated CUDA kernels on
    //   modern GPUs.
    // - Vectorized loads, if working together with local padding, can greatly
    //   complicate the generated CUDA kernels. The reason is because a vector
    //   can have some values loaded from the global memory while some that are
    //   padded. Such situation is challenging to handle in the code generation
    //   process.
    //
    // Therefore, upon seeing an IfThenElseNode, we directly scalarize its body
    // (i.e., transform the vectorized loop into the equivalent serial version).

    // if (dmlc::GetEnv("DIETCODE_CODEGEN_OPT", 0) && dmlc::GetEnv("DIETCODE_DO_LOCAL_PADDING", 1)
    // && CanLocalPad(condition)) {
    // LocalPadder local_padder;
    // return local_padder(Scalarize(GetRef<Stmt>(op)));
    // }

    if (condition.dtype().is_vector()) {
      return Scalarize(GetRef<Stmt>(op));
    }
    Stmt then_case = this->VisitStmt(op->then_case);
    Stmt else_case;
    if (op->else_case.defined()) {
      else_case = this->VisitStmt(op->else_case);
    }
    if (condition.same_as(op->condition) && then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return GetRef<Stmt>(op);
    } else {
      return IfThenElse(condition, then_case, else_case);
    }
  }
  // While
  Stmt VisitStmt_(const WhileNode* op) final {
    LOG(FATAL) << "A while loop inside a vectorized loop not supported.";
    return Stmt();
  }
  // LetStmt
  Stmt VisitStmt_(const LetStmtNode* op) final {
    PrimExpr value = this->VisitExpr(op->value);
    ICHECK(!let_binding_.count(op->var)) << "SSA violation, a single var is binded twice";
    let_binding_[op->var] = value;

    if (value.dtype().lanes() != op->value.dtype().lanes()) {
      Var new_var(op->var->name_hint, value.dtype());
      let_binding_[op->var] = new_var;
      return LetStmt(new_var, value, this->VisitStmt(op->body));
    } else {
      let_binding_[op->var] = op->var;
      Stmt body = this->VisitStmt(op->body);
      if (value.same_as(op->value) && body.same_as(op->body)) {
        return GetRef<Stmt>(op);
      } else {
        return LetStmt(op->var, value, body);
      }
    }
  }
  // Allocate
  Stmt VisitStmt_(const AllocateNode* op) final {
    PrimExpr condition = this->VisitExpr(op->condition);
    if (condition.dtype().is_vector()) {
      LOG(WARNING) << "Cannot handle vector extent in alloc ";
      return Scalarize(GetRef<Stmt>(op));
    }
    Array<PrimExpr> extents;
    for (size_t i = 0; i < op->extents.size(); i++) {
      PrimExpr new_ext = this->VisitExpr(op->extents[i]);
      if (new_ext.dtype().is_vector()) {
        LOG(WARNING) << "Cannot handle vector extent in alloc ";
        return Scalarize(GetRef<Stmt>(op));
      }
      extents.push_back(new_ext);
    }
    // place the vector lanes in least significant dimension.
    extents.push_back(var_lanes_);
    // rewrite access to buffer internally.
    Stmt body = VecAllocAccess(op->buffer_var.get(), var_, var_lanes_)(op->body);
    body = this->VisitStmt(body);
    return Allocate(op->buffer_var, op->dtype, extents, condition, body);
  }

  // scalarize the statment
  Stmt Scalarize(Stmt stmt) {
    Var idx(var_->name_hint + ".s", var_->dtype);
    Map<Var, PrimExpr> values{{var_, idx}};
    stmt = Substitute(stmt, values);
    return For(idx, 0, var_lanes_, ForKind::kSerial, stmt);
  }
  // ProducerStore
  Stmt VisitStmt_(const ProducerStoreNode* op) final {
    LOG(FATAL) << "ProducerProvide cannot appear in a TIR PrimFunc";
    return Stmt();
  }

 private:
  // analyzer
  arith::Analyzer analyzer_;
  // deep equal
  ExprDeepEqual deep_equal_;
  // variable to be replaced
  Var var_;
  // the lanes.
  int var_lanes_;
  // ramp representing the var.
  PrimExpr ramp_;
  // flag to mark requirment of scalarization.
  bool need_scalarize_{false};
  // Let binding
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> let_binding_;
  // vectorizable property
  OpAttrMap<TVectorizable> op_vectorizable_ = Op::GetAttrMap<TVectorizable>("TVectorizable");

  // mutate array, with given lane requirement
  // when finished, p_lane updates the lane requirement.
  Array<PrimExpr> MutateArray(Array<PrimExpr> arr, int* p_lanes) {
    if (arr.size() == 0) return arr;
    int& lanes = *p_lanes;
    bool changed = false;
    std::vector<PrimExpr> new_arr(arr.size());
    for (size_t i = 0; i < arr.size(); i++) {
      PrimExpr old_elem = arr[i];
      PrimExpr new_elem = this->VisitExpr(old_elem);
      if (!new_elem.same_as(old_elem)) changed = true;
      new_arr[i] = new_elem;
      lanes = std::max(lanes, new_elem.dtype().lanes());
    }

    for (size_t i = 0; i < arr.size(); ++i) {
      if (new_arr[i].dtype().lanes() != lanes) {
        new_arr[i] = BroadcastTo(new_arr[i], lanes);
        changed = true;
      }
    }
    if (!changed) return arr;
    return Array<PrimExpr>(new_arr);
  }
  template <typename TOp, typename T>
  PrimExpr BinaryVec(const T* op) {
    static_assert(std::is_same<typename TOp::ContainerType, T>::value, "constraint");
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) && b.same_as(op->b)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(a.dtype().lanes(), b.dtype().lanes());
      return TOp(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
  }
  template <typename T, typename FCompute>
  PrimExpr AddSubVec(const T* op, FCompute fcompute) {
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    if (a.same_as(op->a) && b.same_as(op->b)) {
      return GetRef<PrimExpr>(op);
    } else {
      int lanes = std::max(a.dtype().lanes(), b.dtype().lanes());
      if (lanes != 1) {
        const RampNode* b_ramp = b.as<RampNode>();
        const RampNode* a_ramp = a.as<RampNode>();
        if (a.dtype().lanes() == 1 && b_ramp) {
          return Ramp(fcompute(a, b_ramp->base),
                      fcompute(make_zero(b_ramp->stride.dtype()), b_ramp->stride), b_ramp->lanes);
        }
        if (b.dtype().lanes() == 1 && a_ramp) {
          return Ramp(fcompute(a_ramp->base, b), a_ramp->stride, a_ramp->lanes);
        }
      }
      return fcompute(BroadcastTo(a, lanes), BroadcastTo(b, lanes));
    }
  }
};

class LoopVectorizer : public StmtMutator {
 public:
  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kVectorized) {
      ICHECK(is_zero(op->min));
      auto* extent_as_int = op->extent.as<IntImmNode>();
      if (!extent_as_int || extent_as_int->value < 1) {
        LOG(FATAL) << "Failed to vectorize loop with extent " << op->extent;
      }
      return Vectorizer(op->loop_var, static_cast<int>(extent_as_int->value))(op->body);
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }
};

Stmt VectorizeLoop(Stmt stmt) { return LoopVectorizer()(std::move(stmt)); }

class VectorizeSkipper : public StmtMutator {
 public:
  Stmt VisitStmt_(const ForNode* op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<ForNode>();
    if (op->kind == ForKind::kVectorized) {
      return For(op->loop_var, op->min, op->extent, ForKind::kSerial, op->body);
    } else {
      return stmt;
    }
  }
};

Stmt SkipVectorize(Stmt stmt) { return VectorizeSkipper()(std::move(stmt)); }

namespace transform {

// TODO(tvm-team): Make it as a target property.
Pass VectorizeLoop(bool enable_vectorize) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    if (enable_vectorize) {
      n->body = LoopVectorizer()(std::move(n->body));
    } else {
      n->body = VectorizeSkipper()(std::move(n->body));
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.VectorizeLoop", {});
}

TVM_REGISTER_GLOBAL("tir.transform.VectorizeLoop").set_body_typed(VectorizeLoop);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
