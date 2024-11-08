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
 * \file auto_scheduler_layout_rewrite.h
 * \brief Rewrite the layout of "layout free" tensors (e.g., the weight tensors in
 * conv2d and dense layers) according to the tile structure generated by the auto-scheduler.
 */

#include "auto_scheduler_layout_rewrite.h"

#include <tvm/ir/transform.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <deque>
#include <functional>
#include <vector>

#include "../backend/model_parameter_taint_analysis.h"
#include "../backend/te_compiler.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

// Two global variables for receiving layout information from python
std::deque<std::string> AutoSchedulerLayoutRewriter::global_ori_layouts_queue;
std::deque<std::string> AutoSchedulerLayoutRewriter::global_new_layouts_queue;

// Copy an Attrs but with a new auto_scheduler_rewritten_layout filed.
template <typename T>
Attrs CopyAttrsWithNewLayout(const T* ptr, const std::string& layout) {
  auto n = make_object<T>(*ptr);
  n->auto_scheduler_rewritten_layout = layout;
  return Attrs(n);
}

// Mutate ops in a function
class FuncMutator : public ExprMutator {
 public:
  FuncMutator(const std::deque<std::string>& ori_layouts_queue,
              const std::deque<std::string>& new_layouts_queue)
      : ExprMutator(),
        ori_layouts_queue_(ori_layouts_queue),
        new_layouts_queue_(new_layouts_queue) {}

  Expr VisitExpr_(const CallNode* n) {
    auto new_n = ExprMutator::VisitExpr_(n);

    const auto* call = new_n.as<CallNode>();
    if (call && call->op.as<OpNode>() &&
        (std::find(target_ops_.begin(), target_ops_.end(), n->op.as<OpNode>()->name) !=
         target_ops_.end()) &&
        !ori_layouts_queue_.empty() && !new_layouts_queue_.empty()) {
      // Pop a new layout from the queue
      const std::string ori_layout = ori_layouts_queue_.front();
      const std::string new_layout = new_layouts_queue_.front();
      ori_layouts_queue_.pop_front();
      new_layouts_queue_.pop_front();

      // Insert a new op to do layout transform. (This will be simplified by FoldConstant later).
      Expr updated_kernel = MakeAutoSchedulerLayoutTransform(call->args[1], ori_layout, new_layout);
      Array<Expr> updated_args = {call->args[0], updated_kernel};

      // Update the attrs
      Attrs updated_attrs;
      if (auto pattr = call->attrs.as<Conv2DAttrs>()) {
        updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
      } else if (auto pattr = call->attrs.as<Conv2DWinogradAttrs>()) {
        updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
      } else if (auto pattr = call->attrs.as<Conv3DAttrs>()) {
        updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
      } else if (auto pattr = call->attrs.as<MatmulAttrs>()) {
        updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
      } else if (auto pattr = call->attrs.as<DenseAttrs>()) {
        updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
      } else if (auto pattr = call->attrs.as<BatchMatmulAttrs>()) {
        updated_attrs = CopyAttrsWithNewLayout(pattr, new_layout);
      } else {
        LOG(FATAL) << "Unhandled attribute: " << call->attrs;
      }
      new_n = Call(call->op, updated_args, updated_attrs);
    }
    return new_n;
  }

 private:
  std::deque<std::string> ori_layouts_queue_;
  std::deque<std::string> new_layouts_queue_;

  std::vector<std::string> target_ops_{
      "nn.conv2d", "nn.conv3d", "nn.contrib_conv2d_winograd_without_weight_transform",
      "nn.matmul", "nn.dense",  "nn.batch_matmul"};
};

class PrimitiveFunctionBodyChecker : public ExprVisitor {
 public:
  bool Check(const Expr expr) {
    this->VisitExpr(expr);
    return is_primitive_;
  }

 private:
  void VisitExpr_(const CallNode* op) override {
    if (const auto* func = op->op.as<FunctionNode>()) {
      is_primitive_ = is_primitive_ && false;
    }
  }

  bool is_primitive_ = true;
};

Expr AutoSchedulerLayoutRewriter::VisitExpr_(const CallNode* n) {
  auto new_n = ExprMutator::VisitExpr_(n);

  if (const auto* call = new_n.as<CallNode>()) {
    if (const auto* func = call->op.as<FunctionNode>()) {
      auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
      if (PrimitiveFunctionBodyChecker().Check(func->body)) {
        global_ori_layouts_queue.clear();
        global_new_layouts_queue.clear();

        // Use ScheduleGetter to call python lower functions.
        // This is used to get the layout transform information.
        // The layout transformation will be recorded to global_ori_layout_queue
        // and global_new_layouts_queue in ComputeDAG::RewriteLayout.
        auto f = runtime::Registry::Get("auto_scheduler.enter_layout_rewrite");
        CHECK(f) << "Could not find auto_scheduler.enter_layout_rewrite function.";
        (*f)();

        // std::cout << "[ASLR] Entering scope" << std::endl;

        auto callee_func = Downcast<Function>(call->op);
        auto opt_func_model_parameter_taints =
            callee_func->GetAttr<Array<Bool>>(tir::attr::kDBModelParamterTaints);
        ICHECK(opt_func_model_parameter_taints)
            << callee_func->GetAttr<String>(tir::attr::kDBFunctionName) << " " << callee_func.get();
        auto func_model_parameter_taints = opt_func_model_parameter_taints.value();

        tec::PrimFuncFor(
            GetRef<Function>(func), Target::Current(), [](std::string name) { return name; },
            func_model_parameter_taints, {}, batched_execution_, scattered_execution_);
        // std::cout << "[ASLR] Exiting scope\n" << std::endl;

        f = runtime::Registry::Get("auto_scheduler.exit_layout_rewrite");
        CHECK(f) << "Could not find ansor.exit_layout_rewrite function.";
        (*f)();

        // Mutate the called function
        if (!global_ori_layouts_queue.empty() && !global_new_layouts_queue.empty()) {
          auto ret =
              FuncMutator(global_ori_layouts_queue, global_new_layouts_queue).VisitExpr(new_n);
          return ret;
        }
      }
    }
  }
  return new_n;
}

Expr AutoSchedulerLayoutRewrite(const Expr& expr, bool batched_execution,
                                bool scattered_execution) {
  return AutoSchedulerLayoutRewriter(batched_execution, scattered_execution).Mutate(expr);
}

namespace transform {

Pass AutoSchedulerLayoutRewrite(bool batched_execution, bool scattered_execution) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(
            relay::AutoSchedulerLayoutRewrite(f, batched_execution, scattered_execution));
      };
  return CreateFunctionPass(pass_func, 3, "AutoSchedulerLayoutRewrite", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.AutoSchedulerLayoutRewrite")
    .set_body_typed(AutoSchedulerLayoutRewrite);

TVM_REGISTER_GLOBAL("relay.attrs.get_auto_scheduler_rewritten_layout")
    .set_body_typed([](const Attrs& attrs) {
      if (attrs->IsInstance<Conv2DAttrs>()) {
        return attrs.as<Conv2DAttrs>()->auto_scheduler_rewritten_layout;
      } else if (attrs->IsInstance<Conv2DWinogradAttrs>()) {
        return attrs.as<Conv2DWinogradAttrs>()->auto_scheduler_rewritten_layout;
      } else if (attrs->IsInstance<Conv3DAttrs>()) {
        return attrs.as<Conv3DAttrs>()->auto_scheduler_rewritten_layout;
      } else if (attrs->IsInstance<MatmulAttrs>()) {
        return attrs.as<MatmulAttrs>()->auto_scheduler_rewritten_layout;
      } else if (attrs->IsInstance<DenseAttrs>()) {
        return attrs.as<DenseAttrs>()->auto_scheduler_rewritten_layout;
      } else if (attrs->IsInstance<BatchMatmulAttrs>()) {
        return attrs.as<BatchMatmulAttrs>()->auto_scheduler_rewritten_layout;
      } else {
        LOG(FATAL) << "Unhandled attribute: " << attrs;
      }
      return tvm::String();
    });

}  // namespace transform

}  // namespace relay
}  // namespace tvm
