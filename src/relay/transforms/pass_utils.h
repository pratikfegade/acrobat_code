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
 * \file tvm/relay/_transforms/pass_utils.h
 * \brief Utilities for writing passes
 */
#ifndef TVM_RELAY_TRANSFORMS_PASS_UTILS_H_
#define TVM_RELAY_TRANSFORMS_PASS_UTILS_H_

#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../../printer/text_printer.h"
#include "../analysis/dependency_graph.h"
#include "../op/annotation/annotation.h"
#include "../op/memory/on_device.h"
#include "./let_list.h"

namespace tvm {
namespace relay {

/*!
 * \brief Check if expr is positive constant.
 * \param expr The expression to be checked.
 * \return Whether all elements of expr is positive constant.
 */
bool IsAllPositiveConstant(const Expr& expr);

/*!
 * \brief Substitute var with subst.
 * \param type The type to be substituted.
 * \param tvar The type variable to be substituted.
 * \param subst The target of substitution.
 * \return The substituted result.
 */
Type TypeSubst(const Type& type, const TypeVar& tvar, const Type& subst);

/*!
 * \brief Substitute var with subst.
 * \param expr The expr to be substituted.
 * \param tvar The type variable to be substituted.
 * \param subst The target of substitution.
 * \return The substituted result.
 */
Expr TypeSubst(const Expr& expr, const TypeVar& tvar, const Type& subst);

/*!
 * \brief Substitute type vars in type.
 * \param type The type to be substituted.
 * \param subst_map The map of substitution.
 * \return The substituted result.
 */
Type TypeSubst(const Type& type, const tvm::Map<TypeVar, Type>& subst_map);

/*!
 * \brief Substitute type vars in type.
 * \param expr The expr to be substituted.
 * \param subst_map The map of substitution.
 * \return The substituted result.
 */
Expr TypeSubst(const Expr& expr, const tvm::Map<TypeVar, Type>& subst_map);

/*!
 * \brief Check if type is dynamic.
 * \param ty The type to be checked.
 * \return Whether the type is dynamic.
 */
bool IsDynamic(const Type& ty);

/*!
 * \brief Check if call is data dependent.
 * \param call The call to be checked.
 * \return Whether the call is data dependent.
 */
bool IsDataDependent(const CallNode* call);

/*!
 * \brief Make arbitrary transformation preserve the out most function.
 * \param func The transformation.
 * \param e The expression
 * \return the transformed expression. If e is a function the return is also a function.
 */
inline Expr TransformF(const std::function<Expr(const Expr&)>& func, const Expr& e) {
  if (const FunctionNode* f = e.as<FunctionNode>()) {
    return Function(f->params, func(f->body), f->ret_type, f->type_params, f->attrs);
  } else {
    return func(e);
  }
}

/*!
 * \brief Decide whether the expression atomic or not?
 * \param e the expression
 * \return
 *   is it atomic?
 *   if so, the compute cost of the expression is bounded so it can be copy without graph mode.
 */
inline bool IsAtomic(const Expr& expr) {
  Expr true_expr = IgnoreOnDevice(expr);
  return true_expr.as<VarNode>() || true_expr.as<OpNode>() || true_expr.as<ConstructorNode>() ||
         true_expr.as<GlobalVarNode>() ||
         true_expr.as<ConstantNode>();  // Constant is always by reference.
}

/*!
 * \brief Cache the compiler_begin annotation op to reduce registry lookup overhead
 * \param void
 * \return compiler_begin op
 */
inline const Op& CompilerBeginOp() {
  static auto op = Op::Get("annotation.compiler_begin");
  return op;
}

/*!
 * \brief Cache the compiler_end annotation op to reduce registry lookup overhead
 * \param void
 * \return compiler_end op
 */
inline const Op& CompilerEndOp() {
  static auto op = Op::Get("annotation.compiler_end");
  return op;
}

/*!
 * \brief Cache the invoke_tvm_op annotation op to reduce registry lookup overhead
 * \param void
 * \return invoke_tvm op
 */
inline const Op& GetInvokeTVMOp() {
  static const Op& op = Op::Get("vm.invoke_tvm_op");
  return op;
}

/*!
 * \brief Cache the add annotation op to reduce registry lookup overhead
 * \param void
 * \return add op
 */
inline const Op& GetAddOp() {
  static const Op& op = Op::Get("add");
  return op;
}

inline bool IsScalarTensorType(const Type& type) {
  if (auto tn = type.as<TensorTypeNode>()) {
    return (tn->shape.size() == 0);
  }
  return false;
}

/*!
 * \brief Check if all the inputs and outputs to a call are scalars
 * \param op Call to check
 * \return bool If all inputs and outputs are scalars
 */
inline bool IsOpOnScalars(const CallNode* op) {
  if (op->op == GetInvokeTVMOp()) {
    auto check_tuple = [&](const Expr& e) {
      for (auto f : e.as<TupleNode>()->fields) {
        if (!IsScalarTensorType(f->checked_type())) {
          return false;
        }
      }
      return true;
    };
    return check_tuple(op->args[1]) && check_tuple(op->args[2]);
  } else {
    if (!IsScalarTensorType(op->checked_type())) {
      return false;
    }
    size_t start = 0;
    for (size_t i = start; i < op->args.size(); ++i) {
      if (!IsScalarTensorType(op->args[i]->checked_type())) {
        return false;
      }
    }
  }
  return true;
}

inline bool IsMarkedScalarOp(const CallNode* op) {
  if (!op->attrs.defined()) {
    return false;
  }
  if (!op->attrs->IsInstance<DictAttrsNode>()) {
    return false;
  }
  auto opt_op = Downcast<DictAttrs>(op->attrs).GetAttr(tir::attr::kDBScalarCall, NullValue<Expr>());
  if (opt_op) {
    return opt_op.value().as<CallNode>() != nullptr;
  }
  return false;
}

template <typename ConditionObjectPtr>
struct TreeNode {
  typedef std::shared_ptr<TreeNode<ConditionObjectPtr>> pointer;
  virtual ~TreeNode() {}
};

template <typename ConditionObjectPtr>
struct TreeLeafNode : TreeNode<ConditionObjectPtr> {
  using TreeObjectPtr = typename TreeNode<ConditionObjectPtr>::pointer;

  Expr body;

  explicit TreeLeafNode(Expr body) : body(body) {}

  static TreeObjectPtr Make(Expr body) { return std::make_shared<TreeLeafNode>(body); }

  ~TreeLeafNode() {}
};

template <typename ConditionObjectPtr>
struct TreeLeafFatalNode : TreeNode<ConditionObjectPtr> {
  using TreeObjectPtr = typename TreeNode<ConditionObjectPtr>::pointer;

  TreeLeafFatalNode() = default;

  static TreeObjectPtr Make() { return std::make_shared<TreeLeafFatalNode>(); }

  ~TreeLeafFatalNode() {}
};

template <typename ConditionObjectPtr>
struct TreeBranchNode : TreeNode<ConditionObjectPtr> {
  using TreeObjectPtr = typename TreeNode<ConditionObjectPtr>::pointer;

  ConditionObjectPtr cond;
  TreeObjectPtr then_branch;
  TreeObjectPtr else_branch;

  TreeBranchNode(ConditionObjectPtr cond, TreeObjectPtr then_branch, TreeObjectPtr else_branch)
      : cond(cond), then_branch(then_branch), else_branch(else_branch) {}

  static TreeObjectPtr Make(ConditionObjectPtr cond, TreeObjectPtr then_branch,
                            TreeObjectPtr else_branch) {
    return std::make_shared<TreeBranchNode>(cond, then_branch, else_branch);
  }

  ~TreeBranchNode() {}
};

struct ScopeNode;
using Scope = std::shared_ptr<ScopeNode>;
using NodeScopeMap = std::unordered_map<DependencyGraph::Node*, Scope>;
using ExprSet = std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>;

/* Invariant: when parent is null level is 0
 * Invariant: when parent is not null level is 1 + parent->level
 */
struct ScopeNode {
  // the level of the scope
  size_t level;
  // the parent scope
  Scope parent;
  // the corresponding let list which holds all let bindings in the scope
  std::shared_ptr<LetList> let_list = std::make_shared<LetList>();
  explicit ScopeNode(const Scope& parent) : level(1 + parent->level), parent(parent) {}
  ScopeNode() : level(0) {}
};

/*! \brief Calculate the scope of nodes in the dependency graph by least common ancestor.
 *
 *  \param dg the input dependency graph
 *  \param expr_scope the output node -> scope mapping for all nodes.
 *  \param lifted_exprs the output set of expressions whose scope is lifted due to dependency
 */
std::pair<NodeScopeMap, ExprSet> CalcScope(const DependencyGraph& dg);

/*! \brief find the least common ancestor of lhs scope and rhs scope.
 */
Scope LCA(Scope lhs, Scope rhs);

// For basic block normal form.
Expr ToBasicBlockNormalFormAux(const Expr& e);

// Remove on_device calls for easier printing and analysis.
Expr RemoveOnDeviceCalls(const Expr& e);

// Lift lets out of values to make a program with straight line
// control flow
Expr LiftLetsOutOfValues(const Expr& expr);

// Get the type of a relay var, either from the type annotation or the
// checked_type.
Type GetVarType(relay::Var var);

// Pretty print individual relay expressions.
inline std::string DebugPrint(const ObjectRef& obj) {
  return tvm::TextPrinter(false, nullptr, true).PrintFinal(obj).str();
}

template <typename T>
inline IRModule AddFunctionTaints(Map<Function, T> taints, IRModule& mod,
                                  const std::string& attr_name) {
  class TaintAdder : public ExprMutator {
   public:
    TaintAdder(Map<Function, T> taints_map, const std::string& attr_name)
        : taints_map_(taints_map), attr_name_(attr_name) {}

   private:
    Expr VisitExpr_(const FunctionNode* op) final {
      auto mutated = Downcast<Function>(ExprMutator::VisitExpr_(op));
      auto it = taints_map_.find(GetRef<Function>(op));
      if (it != taints_map_.end()) {
        auto taints = (*it).second;
        mutated = WithAttr(mutated, attr_name_, taints);
      }
      return mutated;
    }

    Map<Function, T> taints_map_;
    const std::string& attr_name_;
  };

  Map<GlobalVar, Function> updated_functions;
  TaintAdder adder(taints, attr_name);
  for (auto kv : mod->functions) {
    if (kv.second.as<FunctionNode>()) {
      updated_functions.Set(kv.first, Downcast<Function>(adder(Downcast<Function>(kv.second))));
    }
  }
  for (auto kv : updated_functions) {
    mod->Add(kv.first, kv.second, true);
  }
  return mod;
}

// ToANormalForm for expressions and as a Pass are declared in transform.h

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_TRANSFORMS_PASS_UTILS_H_
