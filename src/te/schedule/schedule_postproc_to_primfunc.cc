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
 * \file schedule_postproc_to_primfunc.cc
 *
 * \brief Translate the function body generated by ScheduleOps
 *  with te related dialects that incorporates Tensor
 *  into the Stmts to a PrimFunc.
 *
 *  Perform this translation before running any TIR optimizations.
 *
 *  Rationale: The body generated by ScheduleOps is not
 *  a formal PrimFunc and cannot be used for further optimization.
 *  This function canonicalize that body and creates a formal PrimFunc.
 *
 *  List of actions taken by the function:
 *  - Remove occurences of te::Tensor, te::Operation in the IR
 *    and replace them by corresponding IR nodes via tir::Buffer.
 *  - Add annotation of extern buffers using the buffer_map field
 *    in the PrimFunc type.
 */
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <unordered_map>
#include <utility>

namespace tvm {
namespace te {

// create a buffer for tensor.
Buffer CreateBufferFor(const Tensor& tensor, String storage_scope = "") {
  std::string name = tensor->op->name;
  if (tensor->op->num_outputs() != 1) {
    name += ".v" + std::to_string(tensor->value_index);
  }
  Buffer buffer = decl_buffer(tensor->shape, tensor->dtype, name, storage_scope);
  return buffer;
}

// A remapper that maps tensor to buffer
class TensorToBufferMapper : public StmtExprMutator {
 public:
  explicit TensorToBufferMapper(std::unordered_map<Tensor, Buffer> buffer_map)
      : buffer_map_(buffer_map) {}

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    auto ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<AttrStmtNode>();
    if (op->attr_key == tir::attr::double_buffer_scope ||
        op->attr_key == tir::attr::rolling_buffer_scope) {
      Stmt body = op->body;
      Operation operation = Downcast<Operation>(op->node);
      for (int i = operation->num_outputs(); i != 0; --i) {
        Buffer buffer = GetOrAllocBuffer(operation.output(i - 1));
        // std::cout << "[SCHED] Alloced1 " << buffer->data << std::endl;
        body = AttrStmt(buffer, op->attr_key, op->value, body);
      }
      return body;
    } else if (op->attr_key == tir::attr::buffer_bind_scope) {
      Array<ObjectRef> tuple = Downcast<Array<ObjectRef>>(op->node);
      Tensor tensor = Downcast<Tensor>(tuple[1]);
      // std::cout << "[SCHED] Alloced2 " << tensor << std::endl;
      return AttrStmt(Array<ObjectRef>{tuple[0], GetOrAllocBuffer(tensor)}, op->attr_key, op->value,
                      op->body);
    } else if (op->attr_key == tir::attr::buffer_dim_align ||
               op->attr_key == tir::attr::prefetch_scope) {
      Tensor tensor = Downcast<Tensor>(op->node);
      Buffer buffer = GetOrAllocBuffer(tensor);
      // std::cout << "[SCHED] Alloced3 " << buffer << std::endl;
      return AttrStmt(buffer, op->attr_key, op->value, op->body);
    } else {
      return ret;
    }
  }

  Stmt VisitStmt_(const ProducerRealizeNode* op) final {
    Tensor tensor = Downcast<Tensor>(op->producer);
    Buffer buffer = GetOrAllocBuffer(tensor, op->storage_scope);
    // std::cout << "[SCHED] Alloced4 " << buffer << std::endl;

    auto ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<ProducerRealizeNode>();

    return BufferRealize(buffer, op->bounds, op->condition, op->body);
  }

  Stmt VisitStmt_(const ProducerStoreNode* op) final {
    Tensor tensor = Downcast<Tensor>(op->producer);
    Buffer buffer = GetBuffer(tensor);

    auto ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<ProducerStoreNode>();

    return BufferStore(buffer, op->value, op->indices);
  }

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    auto ret = StmtExprMutator::VisitExpr_(op);
    op = ret.as<ProducerLoadNode>();
    Tensor tensor = Downcast<Tensor>(op->producer);
    Buffer buffer = GetBuffer(tensor);
    return tir::BufferLoad(buffer, op->indices);
  }

 private:
  Buffer GetOrAllocBuffer(const Tensor& tensor, String storage_scope = "") {
    return GetBuffer(tensor, storage_scope, true);
  }

  Buffer GetBuffer(const Tensor& tensor, String storage_scope = "", bool allow_alloc = false) {
    auto it = buffer_map_.find(tensor);
    if (it != buffer_map_.end()) return it->second;
    ICHECK(allow_alloc) << "Cannot find the Realization point of tensor " << tensor << " "
                        << tensor->op;

    auto buffer = CreateBufferFor(tensor, storage_scope);
    buffer_map_[tensor] = buffer;
    return buffer;
  }

  // maps tensor to buffer.
  std::unordered_map<Tensor, Buffer> buffer_map_;
};

PrimFunc SchedulePostProcToPrimFunc(Array<ObjectRef> arg_list, Stmt body,
                                    Optional<Map<Tensor, Buffer>> extern_buffer_opt,
                                    Optional<Map<Var, Buffer>> scatter_buffer_opt) {
  std::unordered_map<Tensor, Buffer> extern_buffer;

  if (extern_buffer_opt.defined()) {
    auto v = extern_buffer_opt.value();
    extern_buffer = std::unordered_map<Tensor, Buffer>(v.begin(), v.end());
  }

  // for (auto it : extern_buffer) {
  //   std::cout << "[ABC] " << it.first << " " << it.first->op.get() << " " << it.second <<
  //   std::endl;
  // }

  // for (auto arg : arg_list) {
  //   std::cout << "[ABC] ARG " << arg << std::endl;
  // }

  Array<tir::Var> params;
  Map<tir::Var, tir::Buffer> buffer_map;

  for (auto var : arg_list) {
    if (auto* n = var.as<tir::VarNode>()) {
      params.push_back(GetRef<tir::Var>(n));
    } else if (auto* n = var.as<te::TensorNode>()) {
      te::Tensor tensor = GetRef<te::Tensor>(n);
      ICHECK(!extern_buffer.count(tensor));

      tir::Buffer buffer = CreateBufferFor(tensor);
      tir::Var bptr(buffer->name, PrimType(DataType::Handle()));
      params.push_back(bptr);
      buffer_map.Set(bptr, buffer);
      extern_buffer[tensor] = buffer;
    } else {
      tir::Buffer buffer = Downcast<tir::Buffer>(var);
      tir::Var bptr(buffer->name, buffer->data->type_annotation);
      // std::cout << "[BUF]   ParamBuffer " << buffer->name << " " << bptr.get() << " "
      // << buffer->data.get() << std::endl;
      params.push_back(bptr);
      buffer_map.Set(bptr, buffer);
    }
  }

  // for (auto it : buffer_map) {
  //   std::cout << "[ABC] " << it.second->data->name_hint << " " << it.second->data.get() << " "
  //             << scatter_buffer_opt.value().count(it.second->data) << std::endl;
  // }
  // for (auto var : params) {
  //   std::cout << "[XYZ] " << var->name_hint << " " << var.get() << " "
  //             << scatter_buffer_opt.value().count(var) << std::endl;
  // }
  // for (auto it : scatter_buffer_opt.value()) {
  //   std::cout << "[MNO] " << it.first->name_hint << " " << it.first.get() << std::endl;
  // }

  std::cout << "[SCHED] Body " << body << std::endl;

  body = TensorToBufferMapper(std::move(extern_buffer))(std::move(body));
  // We mark this PrimFunc as coming from a TE schedule
  return WithAttr(tir::PrimFunc(params, body, VoidType(), buffer_map, scatter_buffer_opt.value()),
                  "from_legacy_te_schedule", Bool(true));
}

TVM_REGISTER_GLOBAL("schedule.SchedulePostProcToPrimFunc")
    .set_body_typed(SchedulePostProcToPrimFunc);

}  // namespace te
}  // namespace tvm
