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
 * \file bf16_legalize.cc
 * \brief legalize bf16 type by adding cast_to_fp32
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class ScatterLowerer : public StmtExprMutator {
 public:
  ScatterLowerer(bool print) : print_(print) {}

  Stmt VisitStmt_(const StoreNode* op) final {
    if (op->scatter_buffer_var.defined()) {
      return ScatterStore(Downcast<Var>(VisitExpr(op->scatter_buffer_var)), VisitExpr(op->value),
                          VisitExpr(op->scatter_batch_index), VisitExpr(op->scatter_elem_index),
                          VisitExpr(op->predicate), op->span);
    } else {
      return StmtExprMutator::VisitStmt_(op);
    }
  }

  PrimExpr VisitExpr_(const LoadNode* op) final {
    if (print_ && op->buffer_var->name_hint == "placeholder1") {
      std::cout << "[LSM]  Visitin load " << GetRef<PrimExpr>(op) << " " << op->scatter_buffer_var
                << std::endl;
    }
    if (op->scatter_buffer_var.defined()) {
      return ScatterLoad(op->dtype, Downcast<Var>(VisitExpr(op->scatter_buffer_var)),
                         VisitExpr(op->scatter_batch_index), VisitExpr(op->scatter_elem_index),
                         VisitExpr(op->predicate), op->span);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

  bool print_;
};

namespace transform {

Pass LowerScatterLoadsAndStores() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    bool print =
        false;  //(global_symbol.value() == "vm_mod_fused_nn_dense_expand_dims_add_batched");
    if (print) {
      std::cout << "[LSM]  Staring lowering " << global_symbol << std::endl;
    }
    auto* n = f.CopyOnWrite();
    ScatterLowerer lowerer(print);
    n->body = lowerer(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerScatterLoadsAndStores", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerScatterLoadsAndStores")
    .set_body_typed(LowerScatterLoadsAndStores);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
