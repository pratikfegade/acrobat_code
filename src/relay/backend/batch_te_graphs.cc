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
    // std::cout << "[BR] Rewritten " << expr << std::endl;
    // std::cout << "[BR]    " << rewritten << std::endl;
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
      auto expr = tir::ProducerLoad(replaced_producer_op.output(tensor->value_index), indices);
      return expr;
    } else {
      return tir::ExprMutator::VisitExpr_(op);
    }
  }

  const Map<te::Operation, te::Operation>& rmap_;
  const te::IterVar& batch_iv_;
};

std::pair<Map<te::Operation, te::Operation>, te::Tensor> BatchifyTEGraph(
    const Array<te::Tensor>& inputs, const Array<te::Tensor>& outputs) {
  Array<te::Operation> graph_ops = GetSubGraph(outputs, inputs, true);
  if (inputs.size() == 0) {
    for (auto tensor : outputs) {
      graph_ops.push_back(tensor->op);
    }
  }
  // std::cout << "[BR] Batchifying " << graph_ops.size() << " " << inputs.size() << " "
  //           << outputs.size() << std::endl;

  // for (auto tensor : inputs) {
  //   std::cout << "[BR]   Input " << tensor << " " << tensor->op.get() << std::endl;
  // }

  // for (auto tensor : outputs) {
  //   std::cout << "[BR]   Output " << tensor << " " << tensor->op.get() << std::endl;
  // }

  // for (auto tensor : outputs) {
  //   std::cout << "[BR]   OutputOps " << tensor->op << std::endl;
  // }

  tvm::te::Tensor batch_size_tensor = tvm::te::placeholder(Array<PrimExpr>(), DataType::Int(32));

  PrimExpr batch_size = batch_size_tensor(Array<PrimExpr>());
  Map<te::Operation, te::Operation> rewritten;
  Map<te::Operation, te::Operation> ret;
  for (auto op : graph_ops) {
    // std::cout << "[BR]  Op " << op << std::endl;
    auto batchified_op = op;
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
    }
    // std::cout << "[BR]   Rewritten " << batchified_op << std::endl;
    if (!op.same_as(batchified_op)) {
      rewritten.Set(op, batchified_op);
    }
    ret.Set(op, batchified_op);
  }
  return std::make_pair(ret, batch_size_tensor);
}

}  // namespace tec
}  // namespace relay
}  // namespace tvm

#undef COUT
