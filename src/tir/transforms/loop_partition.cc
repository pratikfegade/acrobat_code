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
 * \file loop_partition.cc
 */
#include <dmlc/parameter.h>
#include <tvm/arith/analyzer.h>
#include <tvm/arith/bound.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <regex>
#include <unordered_map>
#include <unordered_set>

#include "../../arith/interval_set.h"
#include "../../runtime/thread_storage_scope.h"
#include "../../support/utils.h"
#include "ir_utils.h"

namespace tvm {

// <DietCode>
//
// The original implementation only allows for partitions of integral regions,
// which can be limited as it cannot handle certain operators such as %, which
// has striped partitions.
//
// Therefore, to make the implementation more generic, we introduce
// `RelaxedPartition`s, which, instead of deducing integral regions, derives
// predicates of the target variable by relaxing other variables.
namespace arith {

/**
 * @brief Deduce a predicate on `var` by relaxing all the variables other than
 *        `var` in `predicate` (using the hints provided by `hint_map` and
 *        `relax_map`). This is in essence similar to the `arith::DeduceBound`
 *        function call. Both are defined in `arith/bound_deducer.cc`.
 *
 * @param var        the target variable
 * @param predicate  the original predicate from the program
 * @param hint_map   the domain of each variable
 * @param relax_map  ditto
 *
 * @return the relaxed predicate
 */
extern PrimExpr DeduceRelaxedPredicate(Var var, PrimExpr predicate,
                                       const std::unordered_map<const VarNode*, IntSet>& hint_map,
                                       const std::unordered_map<const VarNode*, IntSet>& relax_map);

}  // namespace arith

namespace tir {

struct LoopPartitionConfigNode : public tvm::AttrsNode<LoopPartitionConfigNode> {
  bool partition_const_loop;
  bool no_unroll_loop_with_extent_one;

  TVM_DECLARE_ATTRS(LoopPartitionConfigNode, "tir.transform.LoopPartitionConfig") {
    TVM_ATTR_FIELD(partition_const_loop).describe("Split constant loop").set_default(false);
    TVM_ATTR_FIELD(no_unroll_loop_with_extent_one)
        .describe("Don't unroll loops with extent 1")
        .set_default(false);
  }
};

class LoopPartitionConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LoopPartitionConfig, Attrs, LoopPartitionConfigNode);
};

TVM_REGISTER_NODE_TYPE(LoopPartitionConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("tir.LoopPartition", LoopPartitionConfig);

using arith::DeduceBound;
using arith::Intersect;
using arith::IntSet;

using PartitionKey = std::pair<PrimExpr, bool>;
struct PartitionKeyHash {
  std::size_t operator()(PartitionKey const& k) const noexcept {
    std::size_t h1 = ObjectPtrHash{}(k.first);  // NOLINT(whitespace/braces)
    std::size_t h2 = std::hash<bool>{}(k.second);
    return h1 ^ h2;
  }
};

struct PartitionKeyEqual {
  bool operator()(const PartitionKey& k1, const PartitionKey& k2) const {
    // NOLINTNEXTLINE(whitespace/braces)
    return k1.second == k2.second && ObjectPtrEqual{}(k1.first, k2.first);
  }
};

// Each mapping (cond, cond_value) -> interval represents the fact that
// condition cond is proven to have value cond_value (true or false) in interval.
using Partition = std::unordered_map<PartitionKey, IntSet, PartitionKeyHash, PartitionKeyEqual>;

using ExpressionSet = std::unordered_set<PrimExpr, ObjectPtrHash, ObjectPtrEqual>;

// <DietCode> Begin
using RelaxedPartition =
    std::unordered_map<PartitionKey, PrimExpr, PartitionKeyHash, PartitionKeyEqual>;

struct PrimExprStrHash {
  std::size_t operator()(const PrimExpr& k) const {
    std::ostringstream strout;
    strout << k;
    return std::hash<std::string>()(strout.str());
  }
};

struct PrimExprStrEqual {
  bool operator()(const PrimExpr& lhs, const PrimExpr& rhs) const {
    std::ostringstream lhs_strout, rhs_strout;
    lhs_strout << lhs;
    rhs_strout << rhs;
    return lhs_strout.str() == rhs_strout.str();
  }
};

using PartitionKeySet = std::unordered_set<PartitionKey, PartitionKeyHash, PartitionKeyEqual>;
using RelaxedPartitionReverseMap =
    std::unordered_map<PrimExpr, PartitionKeySet, PrimExprStrHash, PrimExprStrEqual>;
// <DietCode> End

// Select potential candidate IRs that can be partitioned.
// Rule:
//   - the range should not be const
//   - there exist a condition expression in the scope that use the var
class CandidateSelector final : public StmtExprVisitor {
 public:
  using VarIsUsed = bool;
  explicit CandidateSelector(bool partition_const_loop)
      : partition_const_loop_(partition_const_loop) {}

  void VisitStmt_(const ForNode* op) final {
    // partition const loop when sets partition_const_loop_
    if (!is_const_int(op->min) || !is_const_int(op->extent) || partition_const_loop_) {
      // always treat var with hint to be partitioned
      const VarNode* var = op->loop_var.get();
      if (partition_hint_vars.count(var)) {
        candidates.insert(GetRef<Stmt>(op));
        StmtExprVisitor::VisitStmt_(op);
        return;
      }
      record_.insert({var, false});
      StmtExprVisitor::VisitStmt_(op);
      if (record_.at(var) && !no_split_) {
        candidates.insert(GetRef<Stmt>(op));
      }
      record_.erase(var);
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      const IterVarNode* iv = op->node.as<IterVarNode>();
      ICHECK(iv);
      Var var = iv->var;
      runtime::ThreadScope scope = runtime::ThreadScope::Create(iv->thread_tag);
      if ((scope.rank == 0) && (!is_const_int(op->value) || partition_const_loop_)) {
        // always treat var with hint to be partitioned
        if (partition_hint_vars.count(var.get())) {
          candidates.insert(GetRef<Stmt>(op));
          StmtExprVisitor::VisitStmt_(op);
          return;
        }
        record_.insert({var.get(), false});
        StmtExprVisitor::VisitStmt_(op);
        if (record_.at(var.get()) && !no_split_) {
          candidates.insert(GetRef<Stmt>(op));
        }
        record_.erase(var.get());
        return;
      }
    } else if (op->attr_key == attr::pragma_loop_partition_hint) {
      const VarNode* var = nullptr;
      if (op->node->IsInstance<VarNode>()) {
        var = op->node.as<VarNode>();
      } else if (op->node->IsInstance<IterVarNode>()) {
        var = op->node.as<IterVarNode>()->var.get();
      }
      ICHECK(var);
      partition_hint_vars.insert(var);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SeqStmtNode* op) final {
    bool init_no_split = no_split_;
    for (Stmt stmt : op->seq) {
      // erase the no split state of before visiting the next one.
      bool temp = init_no_split;
      std::swap(temp, no_split_);
      this->VisitStmt(stmt);
      // restore the no split flag.
      no_split_ = no_split_ || temp;
    }
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::likely())) {
      in_likely_ = true;
      StmtExprVisitor::VisitExpr_(op);
      in_likely_ = false;
    } else if (op->op.same_as(builtin::tvm_thread_allreduce())) {
      // no split if the body contains allreduce.
      no_split_ = true;
      return;
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  void VisitExpr_(const VarNode* op) final {
    if (in_likely_ && record_.count(op)) {
      record_.at(op) = true;
    }
  }

  std::unordered_set<Stmt, ObjectPtrHash, ObjectPtrEqual> candidates;
  std::unordered_set<const VarNode*> partition_hint_vars;

 private:
  bool in_likely_{false};
  bool no_split_{false};
  bool partition_const_loop_{false};
  std::unordered_map<const VarNode*, VarIsUsed> record_;
};

// Finder try best to find partitions for hinted vars
#define DEFINE_PARTITION_FINDER_VISIT_CMP_OP(OpNodeT) \
  void VisitExpr_(const OpNodeT* op) final {          \
    if (has_partition_hint_) {                        \
      DeduceCondition(GetRef<PrimExpr>(op));          \
      return;                                         \
    }                                                 \
    StmtExprVisitor::VisitExpr_(op);                  \
  }

// Populate partitions data structure, i.e., for a specific variable,
// find an interval in which each condition has fixed true or false value
class PartitionFinder : public StmtExprVisitor {
 public:
  explicit PartitionFinder(Var current_var,
                           const std::unordered_map<const VarNode*, IntSet>& hint_map,
                           const std::unordered_map<const VarNode*, IntSet>& relax_map,
                           bool has_partition_hint)
      : current_var_(current_var),
        has_partition_hint_(has_partition_hint),
        hint_map_(hint_map),
        relax_map_(relax_map) {
    for (const auto& kv : hint_map) {
      out_vars_.insert(kv.first);
    }
    for (const auto& kv : relax_map) {
      out_vars_.insert(kv.first);
    }
  }

  void VisitStmt_(const ForNode* op) final {
    auto f_vset_contains = [this](const VarNode* var) { return out_vars_.count(var); };
    if (UsesVar(op->min, f_vset_contains) || UsesVar(op->extent, f_vset_contains)) return;

    const VarNode* var = op->loop_var.get();
    hint_map_.insert({var, IntSet::Interval(op->min, op->min + op->extent - 1)});
    relax_map_.insert({var, IntSet::Interval(op->min, op->min + op->extent - 1)});
    StmtExprVisitor::VisitStmt_(op);
    relax_map_.erase(var);
    hint_map_.erase(var);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    // handle thread_axis
    if (op->attr_key == attr::thread_extent || op->attr_key == attr::virtual_thread) {
      const IterVarNode* thread_axis = op->node.as<IterVarNode>();
      ICHECK(thread_axis);
      const VarNode* var = thread_axis->var.get();
      IntSet dom = IntSet::FromRange(Range(make_zero(op->value.dtype()), op->value));
      hint_map_.insert({var, dom});
      relax_map_.insert({var, dom});
      StmtExprVisitor::VisitStmt_(op);
      relax_map_.erase(var);
      hint_map_.erase(var);
    } else {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const IfThenElseNode* op) final {
    PrimExpr cond = op->condition;

    if (const CallNode* const call_op = cond.as<CallNode>()) {
      if (call_op->op.same_as(builtin::likely())) {
        cond = call_op->args[0];
      }
    }
    if (const LTNode* const lt_op = cond.as<LTNode>()) {
      if (const VarNode* const var = lt_op->a.as<VarNode>()) {
        std::unordered_map<const VarNode*, IntSet>::iterator relax_map_it = relax_map_.find(var);
        if (relax_map_it != relax_map_.end()) {
          IntSet orig_bound = relax_map_it->second;
          relax_map_it->second = IntSet::Interval(orig_bound.min(), lt_op->b);
          StmtExprVisitor::VisitStmt(op->then_case);
          relax_map_it->second = orig_bound;
          return;
        }
      }
    }
    return StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::likely())) {
      DeduceCondition(op->args[0]);
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(GENode);
  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(GTNode);
  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(LENode);
  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(LTNode);
  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(EQNode);
  DEFINE_PARTITION_FINDER_VISIT_CMP_OP(NENode);

  Partition partitions;
  RelaxedPartition relaxed_partitions;

 private:
  void DeduceCondition(const PrimExpr& cond) {
    // For cond, find out the interval, if exists, in which we can prove that cond is
    // true. Also find the interval, if exists, in which we can prove that cond is
    // false.
    if (UsesVar(cond, [this](const VarNode* var) { return var == current_var_.get(); })) {
      IntSet interval = DeduceBound(current_var_, cond, hint_map_, relax_map_);
      if (!interval.IsNothing()) {
        std::cout << "[LP]  Partition for " << cond << " " << current_var_ << std::endl;
        std::cout << "[LP]     " << interval << std::endl;
        // cond is true within interval
        partitions[{cond, true}] = interval;
      } else {
        PrimExpr pred = DeduceRelaxedPredicate(current_var_, cond, hint_map_, relax_map_);
        if (pred.defined()) {
          relaxed_partitions[{cond, true}] = pred;
        }
      }
      PrimExpr inverse_cond = InverseCond(cond);
      if (inverse_cond.defined()) {
        IntSet interval = DeduceBound(current_var_, inverse_cond, hint_map_, relax_map_);
        if (!interval.IsNothing()) {
          std::cout << "[LP]  Partition for " << inverse_cond << " " << current_var_ << std::endl;
          std::cout << "[LP]     " << interval << std::endl;
          // cond is false within interval
          partitions[{cond, false}] = interval;
        } else {
          PrimExpr pred = DeduceRelaxedPredicate(current_var_, cond, hint_map_, relax_map_);
          if (pred.defined()) {
            relaxed_partitions[{cond, false}] = pred;
          }
        }
      }
    }
  }

  PrimExpr InverseCond(const PrimExpr& cond) {
    PrimExpr inverse_cond;
    if (const LTNode* op = cond.as<LTNode>()) {
      // a < b -> a >= b
      inverse_cond = GE(op->a, op->b);
    } else if (const GTNode* op = cond.as<GTNode>()) {
      // a > b -> a <= b
      inverse_cond = LE(op->a, op->b);
    } else if (const LENode* op = cond.as<LENode>()) {
      // a <= b -> a > b
      inverse_cond = GT(op->a, op->b);
    } else if (const GENode* op = cond.as<GENode>()) {
      // a >= b -> a < b
      inverse_cond = LT(op->a, op->b);
    } else if (const EQNode* op = cond.as<EQNode>()) {
      // a == b -> a != b
      inverse_cond = NE(op->a, op->b);
      // a != b -> a == b
    } else if (const NENode* op = cond.as<NENode>()) {
      inverse_cond = EQ(op->a, op->b);
    }
    return inverse_cond;
  }

  Var current_var_;
  bool has_partition_hint_;
  std::unordered_set<const VarNode*> out_vars_;
  std::unordered_map<const VarNode*, IntSet> hint_map_;
  std::unordered_map<const VarNode*, IntSet> relax_map_;
};

// Replace the set of conditions given by ps with cond_value (true or false)
class ConditionEliminator : public StmtExprMutator {
 public:
  explicit ConditionEliminator(const ExpressionSet& ps, bool cond_value = true)
      : ps_(ps), cond_value_(cond_value) {}

  PrimExpr VisitExpr(const PrimExpr& e) final {
    if (ps_.find(e) != ps_.end()) {
      return VisitExpr(cond_value_ ? const_true() : const_false());
    }
    return StmtExprMutator::VisitExpr(e);
  }

 private:
  ExpressionSet ps_;
  bool cond_value_;
};

// Insert the partition branch at the innermost thread scope
class ThreadPartitionInserter : public StmtMutator {
 public:
  explicit ThreadPartitionInserter(const ExpressionSet& ps, PrimExpr cond)
      : ps_(ps), cond_(cond), innermost_thread_scope_(false) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::thread_extent) {
      innermost_thread_scope_ = true;
      Stmt stmt = StmtMutator::VisitStmt_(op);
      // add branch code inside the innermost thread scope
      if (innermost_thread_scope_) {
        Stmt simplified_body = ConditionEliminator(ps_)(op->body);
        Stmt body = IfThenElse(cond_, simplified_body, op->body);
        PrimExpr value = this->VisitExpr(op->value);
        stmt = AttrStmt(op->node, op->attr_key, value, body);
      }
      innermost_thread_scope_ = false;
      return stmt;
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }

 private:
  const ExpressionSet& ps_;
  PrimExpr cond_;
  bool innermost_thread_scope_;
};

// Try to partition range of iteration variables in order to remove (some)
// likely conditions
class LoopPartitioner : public StmtMutator {
 public:
  explicit LoopPartitioner(bool partition_const_loop, bool no_unroll_loop_with_extent_one)
      : selector(CandidateSelector(partition_const_loop)),
        no_unroll_loop_with_extent_one_(no_unroll_loop_with_extent_one) {}

  Stmt VisitAndMutate(Stmt stmt) {
    selector(stmt);
    // std::cout << "[LP] Selector candidates: " << selector.candidates.size() << std::endl;
    return operator()(std::move(stmt));
  }

  Stmt VisitStmt_(const ForNode* op) final {
    auto fs = GetRef<Stmt>(op);
    if (selector.candidates.count(fs)) {
      Stmt s = TryPartition(fs, op->loop_var, op->min, op->min + op->extent - 1, op->body, false);
      if (s.defined()) return s;
    }

    // normal path when loop partition fails
    // normal loop variable can be put into hint map.
    hint_map_.insert({op->loop_var.get(), IntSet::Interval(op->min, op->min + op->extent - 1)});
    Stmt res = StmtMutator::VisitStmt_(op);
    hint_map_.erase(op->loop_var.get());
    return res;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key != attr::thread_extent) {
      return StmtMutator::VisitStmt_(op);
    }

    const IterVarNode* iv = op->node.as<IterVarNode>();
    ICHECK(iv);
    Var var = iv->var;
    auto as = GetRef<Stmt>(op);
    if (selector.candidates.count(as)) {
      // try optimizing for the inner body first
      auto inner_body = StmtMutator::VisitStmt(op->body);
      Stmt new_as_body = Downcast<AttrStmt>(inner_body);
      AttrStmt new_as = AttrStmt(op->node, op->attr_key, op->value, new_as_body);
      Stmt s = TryPartition(new_as, var, 0, op->value - 1, new_as_body, true);
      if (s.defined()) {
        return s;
      }
    }

    // normal path when loop parittion fails.
    runtime::ThreadScope scope = runtime::ThreadScope::Create(iv->thread_tag);
    Stmt res;
    if (scope.rank == 1) {
      // threadIdx should be put into relax map, in case of divergence.
      relax_map_.insert({var.get(), IntSet::Interval(make_zero(var.dtype()), op->value - 1)});
      res = StmtMutator::VisitStmt_(op);
      relax_map_.erase(var.get());
    } else {
      hint_map_.insert({var.get(), IntSet::Interval(make_zero(var.dtype()), op->value - 1)});
      res = StmtMutator::VisitStmt_(op);
      hint_map_.erase(var.get());
    }
    return res;
  }

 private:
  Stmt TryPartition(const Stmt& stmt, Var var, PrimExpr min, PrimExpr max, Stmt body,
                    bool partition_thread_scope);

  std::pair<IntSet, ExpressionSet> GetIntervalAndCondset(const Partition& partitions,
                                                         const arith::IntervalSet& for_interval,
                                                         bool cond_value);

  inline Stmt MakeFor(const Object* op, PrimExpr extent, Stmt body);

  /* Candidate IRs that may be partitioned potentially */
  std::unordered_map<const VarNode*, IntSet> hint_map_;
  std::unordered_map<const VarNode*, IntSet> relax_map_;
  arith::Analyzer analyzer_;
  CandidateSelector selector;
  bool no_unroll_loop_with_extent_one_;
};

// Returns an interval (in the first component) in which all the conditions
// given in the second component provably have value given by cond_value
std::pair<IntSet, ExpressionSet> LoopPartitioner::GetIntervalAndCondset(
    const Partition& partitions, const arith::IntervalSet& for_interval, bool cond_value) {
  Array<IntSet> sets;
  ExpressionSet cond_set;

  // std::cout << "[LP] Intersecting sets " << cond_value << " " << for_interval << std::endl;

  for (const auto& kv : partitions) {
    if (kv.first.second == cond_value) {
      arith::IntervalSet interval = Downcast<arith::IntervalSet>(kv.second);
      arith::IntervalSet intersection = arith::Intersect(&analyzer_, interval, for_interval);
      // std::cout << "[LP]  Interval " << interval << std::endl;
      // std::cout << "[LP]   Intersection " << intersection << std::endl;
      if (!intersection->IsEmpty()) {
        sets.push_back(kv.second);
        cond_set.insert(kv.first.first);
      }
    }
  }
  IntSet interval = sets.empty() ? IntSet::Nothing() : Intersect(sets);
  return std::make_pair(interval, cond_set);
}

// <DietCode>
//
// Eliminator & Simplifier
//
// Whenever a loop partition happens, the entire loop is divided into two
// regions: one where the predicates are eliminated and the other where the
// predicates cannot but can be further simplified.
namespace {
class PredicateEliminator : public StmtExprMutator {
 public:
  PredicateEliminator(const std::pair<PrimExpr, PartitionKeySet>& relaxed_partition)
      : relaxed_partition_(relaxed_partition) {}

 private:
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::likely())) {
      PrimExpr cond = op->args[0];

      if (relaxed_partition_.second.find(PartitionKey{cond, true}) !=
          relaxed_partition_.second.end()) {
        return Bool(true);
      }
      if (relaxed_partition_.second.find(PartitionKey{cond, false}) !=
          relaxed_partition_.second.end()) {
        return Bool(false);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  std::pair<PrimExpr, PartitionKeySet> relaxed_partition_;
};

class PredicateSimplifier : public StmtExprMutator {
 public:
  PredicateSimplifier(const std::pair<PrimExpr, PartitionKeySet>& relaxed_partition)
      : relaxed_partition_(relaxed_partition) {
    if (const LTNode* const op = relaxed_partition.first.as<LTNode>()) {
      expr_ = op->a;
      result_ = op->b;
    }
    if (const GTNode* const op = relaxed_partition.first.as<GTNode>()) {
      expr_ = op->a;
      result_ = op->b;
    }
  }

 private:
  PrimExpr VisitExpr(const PrimExpr& e) final {
    if (!simplify_predicate_ || !expr_.defined() || !result_.defined()) {
      return StmtExprMutator::VisitExpr(e);
    }
    if (PrimExprStrEqual()(e, expr_)) {
      return result_;
    }
    return StmtExprMutator::VisitExpr(e);
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::likely())) {
      PrimExpr cond = op->args[0];
      if (relaxed_partition_.second.find(PartitionKey{cond, true}) !=
          relaxed_partition_.second.end()) {
        simplify_predicate_ = true;
        PrimExpr simplified_cond = VisitExpr(cond);
        simplify_predicate_ = false;
        return likely(simplified_cond);
      }
      if (relaxed_partition_.second.find(PartitionKey{cond, false}) !=
          relaxed_partition_.second.end()) {
        simplify_predicate_ = true;
        PrimExpr simplified_cond = VisitExpr(cond);
        simplify_predicate_ = false;
        return likely(simplified_cond);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  std::pair<PrimExpr, PartitionKeySet> relaxed_partition_;
  PrimExpr expr_, result_;
  bool simplify_predicate_ = false;
};

// <DietCode>
//
// CUDA Header Stripper & Adder
//
// In the case when the loop to partition is blockIdx.x, strip the CUDA kernel
// header up until to the register allocations (which also includes the
// blockIdx.x, vthread, and threadIdx.x attributes). This is to avoid the header
// being duplicated in each partitioned region.
class CUDAKernelHeaderStripper : public StmtExprVisitor {
 public:
  Stmt kernel_body;

 private:
  void VisitStmt_(const AllocateNode* op) final {
    std::string var_name = op->buffer_var->name_hint;
    if (std::regex_match(std::string(op->buffer_var->name_hint), std::regex("(.*)[.]local"))) {
      ICHECK(support::IsLocal(GetPtrStorageScope(op->buffer_var)));
    }
    if (support::IsLocal(GetPtrStorageScope(op->buffer_var))) {
      kernel_body = op->body;
      return;
    }
    return StmtExprVisitor::VisitStmt_(op);
  }
};

class CUDAKernelHeaderAdder : public StmtExprMutator {
 public:
  CUDAKernelHeaderAdder(const Stmt new_kernel_body) : new_kernel_body_(new_kernel_body) {}

 private:
  Stmt VisitStmt_(const AllocateNode* op) final {
    if (std::regex_match(std::string(op->buffer_var->name_hint), std::regex("(.*)[.]local"))) {
      ICHECK(support::IsLocal(GetPtrStorageScope(op->buffer_var)));
    }
    if (support::IsLocal(GetPtrStorageScope(op->buffer_var))) {
      return Allocate(op->buffer_var, op->dtype, op->extents, op->condition, new_kernel_body_);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt new_kernel_body_;
};
}  // anonymous namespace

/*
 * Tries to recursively partition the range of the variable (given by var) of
 * the for loop (given by node and stmt) into a
 * number of disjoint ranges such that in some ranges one or more predicates
 * in the loopnest are provably true or false in each range. For example, given the
 * following loop to partition:
 * for (i = 0; i < 4; i++)
 *    for (j = 0; j < 10; j++)
 *        if (likely(i*10 + j < 36))
 *            A[10*i+j] = B[10*i+j]
 *
 * We first partition range of i, i.e., [0,3] into subranges [0,2] and [3,3] because the
 * likely condition is always true for the first subrange but not always true for the
 * second subrange. Therefore, we'll have
 * for (i = 0; i < 3; i++)
 *    for (j = 0; j < 10; j++)
 *        if (likely(1))
 *           A[10*i+j] = B[10*i+j]
 * for (i = 0; i < 1; i++)
 *    for (j = 0; j < 10; j++)
 *        if (likely((i+3)*10 + j < 36))
 *            A[10*(i+3)+j] = B[10*(i+3)+j]
 * Which is simplified as:
 * for (i = 0; i < 3; i++)
 *    for (j = 0; j < 10; j++)
 *        A[10*i+j] = B[10*i+j]
 * for (j = 0; j < 10; j++) // loopnest 1
 *    if (likely(j < 6))
 *            A[30+j] = B[30+j]
 * Now, we recursively partition j in loopnest 1 into subranges [0,5] and [6,9] where the
 * condition is true for the first subrange and now always true for the second subrange.
 * for (j = 0; j < 6; j++)
 *    if (likely(1))
 *         A[30+j] = B[30+j]
 * for (j = 0; j < 4; j++) // loop 2
 *    if (likely(j < 0))
 *        A[36+j] = B[36+j]
 * Finally we recursively partition loop 2 above into subrange [0,3] where the
 * condition is false and empty interval where the condition is not false,
 * therefore we generate
 * for (j = 0; j < 4; j++)
 *    if (likely(0))
 *        A[36+j] = B[36+j]
 * which will eventually be simplified to empty code. And because only one loop was generated
 * from loop 2 we stop recursing.
 */
Stmt LoopPartitioner::TryPartition(const Stmt& stmt, Var var, PrimExpr min, PrimExpr max, Stmt body,
                                   bool partition_thread_scope) {
  // <DietCode>
  //
  // Do NOT partition inner loops as doing so would lead to incorrect program
  // behaviors.
  std::string var_name = var->name_hint;
  if (var_name.find(".inner") != std::string::npos ||
      var_name.find(".fused") != std::string::npos) {
    return Stmt();
  }

  // std::cout << "[LP] Try Partition: " << var << std::endl;
  using namespace arith;
  // include hint of var.
  hint_map_.insert({var.get(), IntSet::Interval(min, max)});

  bool has_partition_hint_ = selector.partition_hint_vars.count(var.get());
  PartitionFinder finder(var, hint_map_, relax_map_, has_partition_hint_);
  finder(body);

  // std::cout << "[LP]  Partitions: " << finder.partitions.size() << std::endl;

  hint_map_.erase(var.get());
  if (finder.partitions.empty()) {
    // If the partitions are empty, look into relaxed partitions.
    if (!dmlc::GetEnv("DIETCODE_DO_LOOP_PARTITIONING", 0) || finder.relaxed_partitions.empty()) {
      return Stmt();
    }

    Stmt body_stmt = stmt;
    if (var->name_hint == "blockIdx.x") {
      CUDAKernelHeaderStripper stripper;
      stripper(stmt);
      body_stmt = stripper.kernel_body;
    }

    // Merge predicates that are the same. Here we adopt a simple heuristic: two
    // relaxed predicates are considered the same iff. their are printed the
    // same (as can be seen from the implementation of RelaxedPredicateEqual).
    RelaxedPartitionReverseMap relaxed_partitions_reserved_map;
    for (const std::pair<PartitionKey, PrimExpr>& kv : finder.relaxed_partitions) {
      relaxed_partitions_reserved_map[kv.second].insert(kv.first);
    }

    arith::Analyzer analyzer;
    std::vector<PrimExpr> partitioned_regions;
    // walkthrough the relaxed predicates to obtain the partitioned regions
    std::function<Stmt(RelaxedPartitionReverseMap::iterator,
                       const RelaxedPartitionReverseMap::iterator)>
        relaxed_predicates_walkthru =
            [&relaxed_predicates_walkthru, body_stmt](
                RelaxedPartitionReverseMap::iterator iter,
                const RelaxedPartitionReverseMap::iterator end_iter) -> Stmt {
      RelaxedPartitionReverseMap::iterator curr = iter;
      ++iter;
      PredicateEliminator eliminator(*curr);
      PredicateSimplifier simplifier(*curr);
      if (iter == end_iter) {
        return IfThenElse(curr->first, eliminator(body_stmt), simplifier(body_stmt));
      }
      Stmt child_body_stmt = relaxed_predicates_walkthru(iter, end_iter);
      return IfThenElse(curr->first, eliminator(child_body_stmt), simplifier(child_body_stmt));
    };

    Stmt new_body_stmt = relaxed_predicates_walkthru(relaxed_partitions_reserved_map.begin(),
                                                     relaxed_partitions_reserved_map.end());
    if (var->name_hint == "blockIdx.x") {
      CUDAKernelHeaderAdder adder(new_body_stmt);
      return adder(stmt);
    }
    return new_body_stmt;
  }

  arith::IntervalSet for_interval(min, max);
  bool cond_value;
  IntSet middle_interval;
  ExpressionSet cond_set;

  // find an interval in which all conditions on var are true
  std::tie(middle_interval, cond_set) =
      GetIntervalAndCondset(finder.partitions, for_interval, true);
  if (middle_interval.IsNothing()) {
    // if such interval doesn't exist, find an interval in which all
    // conditions on var are false
    std::tie(middle_interval, cond_set) =
        GetIntervalAndCondset(finder.partitions, for_interval, false);
    if (middle_interval.IsNothing())
      // we couldn't find an interval in which the conditions are provably true or false
      // Therefore, we can't partition the loop based on those conds
      return Stmt();
    cond_value = false;
  } else {
    cond_value = true;
  }

  IntervalSet middle_interval_i = Downcast<IntervalSet>(middle_interval);
  // middle_interval is the subrange of the loop variable range for which a
  // set of conditions are true (or false resp.)
  // The part of the loop variable range that is before (after resp.) that
  // subrange is prefixed with pre- (post- resp.)

  // Calculating pre-subrange and generating code for it.
  // pre-subrange = [min, body_begin)
  PrimExpr body_begin;
  Stmt pre_stmt;
  bool pre_stmt_recurse = true;
  if (middle_interval_i->HasLowerBound()) {
    body_begin = analyzer_.Simplify(middle_interval.min());
    if (!analyzer_.CanProve(body_begin == min)) {
      PrimExpr cond = (body_begin - min >= 0);
      if (!analyzer_.CanProve(cond)) {
        LOG(WARNING) << "Cannot prove: " << cond << ", when generating the pre doubt loop";
        body_begin = Max(body_begin, min);
        // stop recursing on this interval if we can't prove it has non-negative length
        pre_stmt_recurse = false;
      }
      if (!partition_thread_scope) {
        Stmt pre_body = Substitute(body, {{Var{var}, var + min}});
        pre_stmt = MakeFor(stmt.get(), body_begin - min, pre_body);
      }
    }
  } else {
    body_begin = min;
  }

  // Calculating post-subrange and generating code for it.
  // post-subrange = [post_doubt_begin, max+1)
  PrimExpr post_doubt_begin;
  Stmt post_stmt;
  bool post_stmt_recurse = true;
  if (middle_interval_i->HasUpperBound()) {
    post_doubt_begin = analyzer_.Simplify(middle_interval.max() + 1);
    if (!analyzer_.CanProve(middle_interval.max() == max)) {
      // require the extent to be non-negative
      // std::cout << "[LP]    Post doubt extent " << max << " " << post_doubt_begin << std::endl;
      PrimExpr cond = (max - post_doubt_begin + 1 >= 0);
      if (!analyzer_.CanProve(cond)) {
        LOG(WARNING) << "Cannot prove: " << cond << ", when generating the post doubt loop";
        post_doubt_begin = Min(post_doubt_begin, max + 1);
        // stop recursing on this interval if we can't prove it has non-negative length
        post_stmt_recurse = false;
      }
      if (!partition_thread_scope) {
        Stmt post_body = Substitute(body, {{Var{var}, var + post_doubt_begin}});
        post_stmt = MakeFor(stmt.get(), max - post_doubt_begin + 1, post_body);
      }
    }
  } else {
    post_doubt_begin = max + 1;
  }

  Stmt s;

  // Generating code for middle subrange
  if (!partition_thread_scope) {
    Stmt mid_stmt;
    if (!analyzer_.CanProve(body_begin >= post_doubt_begin)) {
      // [body_begin, post_doubt_begin)
      Stmt simplified_body = ConditionEliminator(cond_set, cond_value)(body);
      Stmt new_body = Substitute(simplified_body, {{Var{var}, var + body_begin}});
      mid_stmt = MakeFor(stmt.get(), post_doubt_begin - body_begin, new_body);

      // Recurse for each non-empty subrange only if there are at least
      // two non-empty subranges
      if (pre_stmt.defined() || post_stmt.defined()) {
        mid_stmt = VisitAndMutate(mid_stmt);
        if (pre_stmt.defined() && pre_stmt_recurse) {
          pre_stmt = VisitAndMutate(pre_stmt);
        }
        if (post_stmt.defined() && post_stmt_recurse) {
          post_stmt = VisitAndMutate(post_stmt);
        }
      }
    }
    s = SeqStmt::Flatten(pre_stmt, mid_stmt, post_stmt);
  } else {
    PrimExpr cond = const_true();
    if (!analyzer_.CanProve(body_begin == min)) cond = cond && (var >= body_begin);
    if (!analyzer_.CanProve(post_doubt_begin == (max + 1))) cond = cond && (var < post_doubt_begin);
    s = ThreadPartitionInserter(cond_set, cond)(stmt);
  }
  s = ConvertSSA(s);
  return s;
}

inline Stmt LoopPartitioner::MakeFor(const Object* node, PrimExpr extent, Stmt body) {
  const ForNode* for_node = static_cast<const ForNode*>(node);
  ICHECK(for_node);
  if (analyzer_.CanProve(extent == make_const(DataType::Int(32), 1)) &&
      !no_unroll_loop_with_extent_one_) {
    // If the loop extent is 1, do not create the loop anymore
    return Substitute(body, {{Var{for_node->loop_var}, make_const(DataType::Int(32), 0)}});
  } else {
    ICHECK(for_node->kind != ForKind::kThreadBinding);
    return For(for_node->loop_var, IntImm(for_node->min.dtype(), 0), extent, for_node->kind, body);
  }
}

class RemoveLikelyTagsAndHints : public StmtExprMutator {
 public:
  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::likely())) {
      ICHECK_EQ(op->args.size(), 1);
      return StmtExprMutator::VisitExpr(op->args[0]);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == attr::pragma_loop_partition_hint) {
      return VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

Stmt LoopPartition(Stmt stmt, bool partition_const_loop, bool no_unroll_loop_with_extent_one) {
  stmt = LoopPartitioner(partition_const_loop, no_unroll_loop_with_extent_one)
             .VisitAndMutate(std::move(stmt));
  stmt = RemoveLikelyTagsAndHints()(std::move(stmt));
  return stmt;
}

namespace transform {

Pass LoopPartition() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    auto cfg = ctx->GetConfig<LoopPartitionConfig>("tir.LoopPartition");
    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<LoopPartitionConfig>();
    }
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol).value();

    // <DietCode>
    //
    // Enable constant loop partitioning, which is turned off by default.
    // However, note that simply enabling the optimization is not enough, as the
    // current implementation of the LoopPartitioner give erroneous code that is
    // even not runnable. Please refer to above for the changes that we made.
    n->body = LoopPartition(
        std::move(n->body),
        cfg.value()->partition_const_loop || dmlc::GetEnv("DIETCODE_DO_LOOP_PARTITIONING", 0),
        cfg.value()->no_unroll_loop_with_extent_one);

    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LoopPartition", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LoopPartition").set_body_typed(LoopPartition);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
