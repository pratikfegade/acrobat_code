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

#include "../../support/utils.h"
#include "batch_te_graph.h"

namespace tvm {
namespace relay {
namespace tec {

class BatchifyRewriter : public te::ExprMutator {
 public:
  BatchifyRewriter(const Map<te::Operation, te::Operation>& rmap, const te::IterVar& batch_iv)
      : rmap_(rmap), batch_iv_(batch_iv) {}

  PrimExpr Rewrite(const PrimExpr& expr) {
    PrimExpr rewritten = this->VisitExpr(expr);
    std::cout << "[BR] Rewritten " << expr << std::endl;
    std::cout << "[BR]    " << rewritten << std::endl;
    return rewritten;
  }

 private:
  PrimExpr VisitExpr_(const tir::ProducerLoadNode* op) override {
    if (auto tensor = op->producer.as<te::TensorNode>()) {
      auto producer_op = tensor->op;
      if (!rmap_.count(producer_op)) {
        return tir::ExprMutator::VisitExpr_(op);
      }
      auto replaced_producer_op = rmap_.at(producer_op);

      Array<PrimExpr> indices;
      indices.push_back(batch_iv_);
      indices.push_back_all(op->indices);
      auto expr = tir::ProducerLoad(GetTensor(replaced_producer_op, tensor->value_index), indices);
      return expr;
    } else {
      return tir::ExprMutator::VisitExpr_(op);
    }
  }

  te::Tensor GetTensor(const te::Operation& op, size_t index) {
    auto key = std::make_pair(op.get(), index);
    auto it = tensor_cache_.find(key);
    if (it == tensor_cache_.end()) {
      tensor_cache_.insert({key, op.output(index)});
    }
    return tensor_cache_.at(key);
  }

  const Map<te::Operation, te::Operation>& rmap_;
  std::unordered_map<std::pair<const Object*, size_t>, te::Tensor, support::PairHash,
                     support::PairEquals>
      tensor_cache_;
  const te::IterVar& batch_iv_;
};

std::pair<Map<te::Operation, te::Operation>, tir::Var> BatchifyTEGraph(
    const Array<te::Tensor>& inputs, const Array<te::Tensor>& outputs,
    const std::vector<bool>& reuse_taints, const std::string& unbatched_name) {
  bool print = false;
  // bool print = (unbatched_name == "vm_mod_fused_zeros");
  if (print) {
    std::cout << "[BR] Batchifying " << unbatched_name << std::endl;
  }
  Array<te::Operation> graph_ops = GetSubGraph(outputs, inputs, true);
  if (inputs.size() == 0) {
    for (auto tensor : outputs) {
      graph_ops.push_back(tensor->op);
    }
  }

  std::unordered_set<const Object*> no_batchify;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (print) {
      std::cout << "[BR]  Input " << inputs[i] << " " << reuse_taints[i] << std::endl;
    }
    if (reuse_taints[i]) {
      no_batchify.insert(inputs[i]->op.get());
    }
  }

  for (auto op : graph_ops) {
    if (auto cop = op.as<te::ComputeOpNode>()) {
      bool reuse = true;
      for (auto tensor : cop->InputTensors()) {
        if (!no_batchify.count(tensor->op.get())) {
          reuse = false;
          break;
        }
      }
      if (reuse) {
        no_batchify.insert(cop);
        if (print) {
          std::cout << "[BR]    NoBatchD " << op << std::endl;
        }
      }
    }
  }
  tir::Var batch_size = tir::Var("batch_size", DataType::Int(32));
  Map<te::Operation, te::Operation> rewritten;
  Map<te::Operation, te::Operation> ret;
  for (auto op : graph_ops) {
    if (print) {
      std::cout << "[BR]  Op " << op << std::endl;
    }

    auto batchified_op = op;
    if (!no_batchify.count(op.get())) {
      if (print) {
        std::cout << "[BR]   Batching" << std::endl;
      }
      if (auto pop = op.as<te::PlaceholderOpNode>()) {
        Array<PrimExpr> new_shape;
        new_shape.push_back(batch_size);
        new_shape.push_back_all(pop->shape);
        batchified_op = te::PlaceholderOp(pop->name, new_shape, pop->dtype);
      } else if (auto cop = op.as<te::ComputeOpNode>()) {
        te::IterVar batch_iv =
            te::IterVar(Range(0, batch_size), tir::Var("b_iv", DataType::Int(32)), tir::kDataPar);
        Array<te::IterVar> new_axis;
        new_axis.push_back(batch_iv);
        new_axis.push_back_all(cop->axis);

        BatchifyRewriter rewriter(rewritten, batch_iv);
        Array<PrimExpr> new_body;
        for (auto e : cop->body) {
          new_body.push_back(rewriter(e));
        }
        batchified_op = te::ComputeOp(cop->name, cop->tag, cop->attrs, new_axis, new_body);
      } else {
        ICHECK(false) << "Op " << op << " not supported for batchifying";
      }
    }
    if (!op.same_as(batchified_op)) {
      if (print) {
        std::cout << "[BR]    Batched " << batchified_op << std::endl;
      }
      rewritten.Set(op, batchified_op);
    }
    ret.Set(op, batchified_op);
  }
  return std::make_pair(ret, batch_size);
}

}  // namespace tec
}  // namespace relay
}  // namespace tvm

#undef COUT
