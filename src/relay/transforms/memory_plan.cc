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
 * \file src/relay/transforms/memory_plan.cc
 *
 * \brief Memory planning pass that currently coalesces allocations.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/op.h>

#include "../../support/arena.h"
#include "../op/annotation/annotation.h"
#include "../op/memory/memory.h"
#include "./expr_subst.h"
#include "pass_utils.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

/*!
 * Represents a control-free allocation region.
 *
 * The below pass groups sets of allocations into regions, then
 * replaces the region with a single allocation.
 */
class Region {
 public:
  static std::shared_ptr<Region> empty(size_t region_no) {
    auto zero = MakeConstantScalar(DataType::Int(64), 0);
    auto region_var = Var("region" + std::to_string(region_no), Type(nullptr));
    Region* raw_ptr =
        new Region(region_var, zero, NullValue<Expr>(), {}, SEScope::FullyUnconstrained());
    // std::cout << "Creating region " << raw_ptr << raw_ptr->var_ << std::endl;
    return std::shared_ptr<Region>(raw_ptr);
  }

  Region(const Var& var, const Expr& size, const Expr& alignment, DataType dtype,
         const SEScope& device)
      : var_(var), size_(size), alignment_(alignment), dtype_(dtype), device_(device) {}

  void grow(Var old_storage, Expr size, Expr alignment, SEScope dev, DataType dtype) {
    // Grow the region by a given allocation as well as track the old storage
    // for later rewriting the program to use the allocated region.

    // std::cout << "Growing region by " << this << " " << size << std::endl;

    if (initialized_) {
      ICHECK_EQ(dtype_, dtype) << "must have matching dtypes in a region";
    } else {
      dtype_ = dtype;
      initialized_ = true;
    }

    if (alignment_.defined()) {
      ICHECK(tvm::StructuralEqual()(alignment_, alignment))
          << "must have matching alignments in a region";
    } else {
      alignment_ = alignment;
    }

    if (device_initialized_) {
      // std::cout << "EQ1 " << device_ << " " << dev << std::endl;
      // std::cout << "EQ1 " << device_->target << " " << dev->target << std::endl;
      // std::cout << "EQ1 " << device_->target->kind << " " << dev->target->kind << std::endl;
      // std::cout << "EQ2 " << device_->virtual_device_id << " " << dev->virtual_device_id
      // << std::endl;
      ICHECK_EQ(device_->target->kind, dev->target->kind) << "must have matching device";
      ICHECK_EQ(device_->virtual_device_id, dev->virtual_device_id) << "must have matching device";
    } else {
      ICHECK(dev.defined());
      device_ = dev;
      device_initialized_ = true;
    }
    Expr new_size =
        Multiply(Divide(Subtract(Add(size, alignment_), MakeConstantScalar(DataType::Int(64), 1)),
                        alignment_),
                 alignment_);

    // Record the offset at which we allocate the storage.
    auto offset_var =
        Var("offset" + std::to_string(offsets_.size()), TensorType({}, DataType::Int(64)));
    offsets_[old_storage] = std::make_pair(offset_var, size_);

    size_ = Add(size_, new_size);
  }

  Expr offset_for(const Expr& alloc) {
    // for (auto it : offsets_) {
    // std::cout << " Looking at " << it.first << " for " << alloc << std::endl;
    // }
    auto it = offsets_.find(alloc);
    if (it != offsets_.end()) {
      return it->second.first;
    } else {
      return NullValue<Expr>();
    }
  }

  Expr to_expr(Expr body) {
    // Generate the prelude code for a region, wrapping the body in it.
    // The prelude contains the single allocation for a region, and
    // all offset computations.

    if (!device_.defined()) {
      device_ = SEScope(kDLCPU, 0, Target("llvm"), "global");
    }

    // Generate bindings for each and every size computation
    // we must do this to maintain ANF.
    std::vector<std::pair<Var, Expr>> bindings;

    // First compute the total size.
    Var total_size = Var("total_size" + std::to_string(tvm::StructuralHash()(body)),
                         TensorType({}, DataType::Int(64)));
    bindings.push_back(std::make_pair(total_size, size_));

    // Allocate the entire region with a single call.
    Expr alloc = AllocStorage(total_size, alignment_, device_, dtype_);
    // std::cout << "[MP] New storage " << alloc << std::endl;
    bindings.push_back(std::make_pair(var_, alloc));

    // Generate variables which contain all of the offset math.
    // Ensure we constant evaluate away all the math here.

    // In theory we can support dynamic offsets but this
    // requires another round of memory planning and
    // potentially colaescing.
    for (auto it : offsets_) {
      auto alloc = it.first;
      auto var = it.second.first;
      auto offset = it.second.second;

      bindings.push_back(std::make_pair(var, offset));
    }

    for (auto it = bindings.rbegin(); it != bindings.rend(); ++it) {
      auto var = it->first;
      auto value = it->second;
      body = Let(var, value, body);
    }
    return body;
  }

  Var var_;
  Expr size_;
  Expr alignment_;
  DataType dtype_;
  SEScope device_;
  bool initialized_{false};
  bool device_initialized_{false};
  std::unordered_map<Expr, std::pair<Var, Expr>, StructuralHash, StructuralEqual> offsets_;
};

class MemoryPlanner : public ExprMutator {
 public:
  Expr PlanMemory(const Expr& expr) { return this->VisitExpr(expr); }

  Expr VisitExpr_(const LetNode* let_node) {
    // std::cout << "[MP] Visiting let" << std::endl;
    std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual> dynamic_regions;
    std::vector<std::pair<Var, Expr>> bindings;

    Expr let = GetRef<Expr>(let_node);
    while ((let_node = let.as<LetNode>())) {
      auto lhs = let_node->var;
      auto rhs = let_node->value;

      // std::cout << "[MP]  Let var " << lhs->vid->name_hint << std::endl;
      {
        enter_scope("Let value of " + lhs->vid->name_hint);
        rhs = this->VisitExpr(rhs);
        rhs = exit_scope(rhs);
      }

      auto on_device_props = GetOnDeviceProps(rhs);
      if (on_device_props.body.defined()) {
        rhs = on_device_props.body;
      }

      const CallNode* rhs_call_op = rhs.as<CallNode>();
      if (rhs_call_op && rhs_call_op->op == Op::Get("memory.alloc_storage")) {
        ICHECK(on_device_props.se_scope->target.defined()) << lhs;
        auto binding =
            process_alloc_storage(&dynamic_regions, lhs, rhs_call_op, on_device_props.se_scope);
        lhs = binding.first;
        rhs = binding.second;
      } else if (rhs_call_op && rhs_call_op->op == Op::Get("memory.alloc_tensor")) {
        auto binding = process_alloc_tensor(lhs, rhs_call_op);
        lhs = binding.first;
        rhs = binding.second;
      }

      if (on_device_props.body.defined()) {
        // attrs = on_device_props[1];
        // rhs = rhs;
      }

      bindings.push_back(std::make_pair(lhs, rhs));
      let = let_node->body;
    }

    let = VisitExpr(let);
    for (auto it = bindings.rbegin(); it != bindings.rend(); ++it) {
      auto var = it->first;
      auto value = it->second;
      ICHECK(var.defined());
      ICHECK(value.defined());

      let = Let(var, value, let);
      if (dynamic_regions.count(var)) {
        let = exit_scope(let);
      }
    }
    return let;
  }

  Expr VisitExpr_(const IfNode* ite) {
    enter_scope("If true");
    auto true_branch = VisitExpr(ite->true_branch);
    true_branch = exit_scope(true_branch);

    enter_scope("If false");
    auto false_branch = VisitExpr(ite->false_branch);
    false_branch = exit_scope(false_branch);

    return If(ite->cond, true_branch, false_branch);
  }

  Expr VisitExpr_(const FunctionNode* func_node) {
    // Transform the function body to use region allocation scheme.
    if (func_node->HasNonzeroAttr(attr::kPrimitive)) {
      return ExprMutator::VisitExpr_(func_node);
    } else {
      enter_scope("Function start");
      auto body = VisitExpr(func_node->body);
      body = exit_scope(body);
      return Function(func_node->params, body, func_node->ret_type, func_node->type_params,
                      func_node->attrs);
    }
  }

 private:
  std::pair<Var, Expr> process_alloc_storage(
      std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>* p_dynamic_regions, const Var& lhs,
      const CallNode* call, const SEScope& se_scope) {
    // std::cout << "Processing storage " << lhs << " " << GetRef<Expr>(call) << std::endl;
    std::unordered_set<Expr, ObjectPtrHash, ObjectPtrEqual>& dynamic_regions = *p_dynamic_regions;
    // Process alloc_storage
    auto size = call->args[0];
    auto alignment = call->args[1];
    auto dtype = call->attrs.as<AllocStorageAttrs>()->dtype;

    auto size_props = GetOnDeviceProps(size);
    if (size_props.body.defined()) {
      // ICHECK_EQ(se_scope, size_props[1].se_scope);
      size = size_props.body;
    }

    if (!size.as<ConstantNode>()) {
      enter_scope("Non constant size");
      dynamic_regions.insert(lhs);
    } else {
      // A new scope is created when entering a new region with
      // different device device.
      std::shared_ptr<Region> region = current_region(dtype);
      if (region->device_.defined() && !region->device_->IsFullyUnconstrained() &&
          region->device_->target->kind != se_scope->target->kind) {
        enter_scope("Constant size but different device");
        dynamic_regions.insert(lhs);
      }
    }
    std::shared_ptr<Region> region = current_region(dtype);
    region->grow(lhs, size, alignment, se_scope, dtype);
    ICHECK(region->var_.defined()) << region.get();
    // std::cout << "Region var " << region->var_ << std::endl;
    return std::make_pair(lhs, region->var_);
    // return std::make_pair(lhs, GetRef<Expr>(call));
  }

  std::pair<Var, Expr> process_alloc_tensor(const Var& lhs, const CallNode* call) {
    // Process alloc tensor. Region and offset are computed
    // std::cout << "Processing tensor " << lhs << " " << GetRef<Expr>(call) << std::endl;
    auto storage = call->args[0];
    auto old_offset = call->args[1];
    auto shape = call->args[2];

    auto old_offset_props = GetOnDeviceProps(old_offset);
    if (old_offset_props.body.defined()) {
      old_offset = old_offset_props.body;
    }
    auto region_and_offset = new_region_and_offset(storage);
    auto region = region_and_offset.first;
    auto offset = region_and_offset.second;

    auto& tensor_allocations_in_current_region = tensor_allocations_.back();
    Var intermediate_var =
        Var(Id(lhs->vid->name_hint + std::to_string(tensor_allocations_in_current_region.size())),
            lhs->type_annotation);

    tensor_allocations_in_current_region.push_back(std::make_pair(
        intermediate_var, Call(call->op, Array<Expr>({region->var_, offset, shape}), call->attrs)));

    // std::cout << "Added " << tensor_allocations_.back().size() << " "
    // << tensor_allocations_in_current_region.size() << std::endl;

    return std::make_pair(lhs, intermediate_var);
  }

  std::pair<std::shared_ptr<Region>, Expr> new_region_and_offset(const Expr& old_storage) {
    // std::cout << "Looking at regions " << regions_.size() << std::endl;
    for (auto it = this->regions_.rbegin(); it != this->regions_.rend(); ++it) {
      auto dtype_region = *it;
      // std::cout << "  Looking at region_maps " << dtype_region.size() << std::endl;
      for (auto iit : dtype_region) {
        auto region = iit.second;
        Expr offset = region->offset_for(old_storage);
        if (offset.defined()) {
          return std::make_pair(region, offset);
        }
      }
    }
    ICHECK(false) << "No region/offset found for old_storage " << old_storage;
  }

  std::shared_ptr<Region> current_region(DataType dtype) {
    auto& current_scope = regions_.back();

    auto it = current_scope.find(dtype);
    if (it == current_scope.end()) {
      std::shared_ptr<Region> region = Region::empty(regions_.size());
      current_scope[dtype] = region;
      return region;
    } else {
      return it->second;
    }
  }

  void enter_scope(std::string reason) {
    // std::cout << "[MP]   Enter Scope: " << reason << std::endl;

    std::vector<std::pair<Var, Expr>> new_tensor_allocations;
    this->tensor_allocations_.push_back(new_tensor_allocations);

    std::unordered_map<DataType, std::shared_ptr<Region>> new_region_map;
    this->regions_.push_back(new_region_map);
  }

  // When leaving a scope build a region allocation for the scope.
  Expr exit_scope(Expr body) {
    // std::cout << "[MP]   Exit Scope" << std::endl;

    auto& tensors_in_region = this->tensor_allocations_.back();
    for (auto it : tensors_in_region) {
      auto var = it.first;
      auto value = it.second;
      // std::cout << "Allocation " << var << " " << value << std::endl;
      body = Let(var, value, body);
    }
    this->tensor_allocations_.pop_back();

    auto& dtype_region = this->regions_.back();
    for (auto it : dtype_region) {
      auto region = it.second;
      if (region->offsets_.size() > 0) {
        body = region->to_expr(body);
      }
    }
    this->regions_.pop_back();
    return body;
  }

  std::vector<std::unordered_map<DataType, std::shared_ptr<Region>>> regions_;
  std::vector<std::vector<std::pair<Var, Expr>>> tensor_allocations_;
};

Expr CPPMemoryPlan(const Expr& expr) { return MemoryPlanner().PlanMemory(expr); }

namespace transform {

Pass CPPMemoryPlan() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        // std::cout << "Naznin " << f << std::endl;
        f = Downcast<Function>(CPPMemoryPlan(f));
        // std::cout << "Shabnam " << f << std::endl;
        return f;
      };
  return CreateFunctionPass(pass_func, 0, "CPPMemoryPlan", {});
}

TVM_REGISTER_GLOBAL("relay._transform.CPPMemoryPlan").set_body_typed(CPPMemoryPlan);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
