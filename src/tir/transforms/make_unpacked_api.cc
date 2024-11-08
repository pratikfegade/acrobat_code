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
 * \file make_unpacked_api.cc Lower PrimFunc to a standard C function API.
 */
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>
#include <utility>
#include <vector>

#include "arg_binder.h"
#include "ir_utils.h"

namespace tvm {
namespace tir {

//////////////////////////////////////////////////////////////////////////////////////////////////////////
class ParamDebugger : public tir::StmtExprVisitor {
 public:
  ParamDebugger(const tir::PrimFunc& func) : func_(func) {}

  void Compute() {
    std::cout << "[DA_PD] Func " << func_ << std::endl;
    for (auto p : func_->params) {
      std::cout << "[DA_PD]  Param " << p << " " << p.get() << std::endl;
    }
    VisitStmt(func_->body);
  }

 private:
  void MergeAndSet(const tir::Var& var) {
    std::cout << "[DA_PD]  var " << var << " " << var.get() << std::endl;
  }

  void VisitExpr_(const tir::LoadNode* op) {
    MergeAndSet(op->buffer_var);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tir::StoreNode* op) {
    MergeAndSet(op->buffer_var);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const tir::ScatterLoadNode* op) {
    MergeAndSet(op->buffer_var);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tir::ScatterStoreNode* op) {
    MergeAndSet(op->buffer_var);
    StmtExprVisitor::VisitStmt_(op);
  }

  const tir::PrimFunc func_;
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////

PrimFunc MakeUnpackedAPI(PrimFunc&& func) {
  auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol) << "MakeUnpackedAPI: Expect PrimFunc to have the global_symbol attribute";

  // if (global_symbol == "vm_mod_fused_sigmoid_tanh_multiply") {
  // ParamDebugger(func).Compute();
  // }

  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target.defined()) << "MakeUnpackedAPI: Require the target attribute";

  auto* func_ptr = func.CopyOnWrite();

  // Setup device context
  int target_device_type = target.value()->kind->device_type;
  Integer device_type(target_device_type);
  Integer device_id(0);
  PrimExpr node = StringImm("default");
  const Stmt nop = Evaluate(0);
  std::vector<Stmt> device_init;

  // Create arg to buffer binder
  std::unordered_map<const VarNode*, PrimExpr> vmap;
  ArgBinder binder(&vmap);

  // Collect variables and buffers to map between
  Array<Var> args;
  std::vector<std::pair<Var, Var>> var_def;
  bool buffer_map_found = false;

  for (int i = 0; i < static_cast<int>(func_ptr->params.size()); ++i) {
    Var param = func_ptr->params[i];

    auto it = func_ptr->buffer_map.find(param);
    if (it != func_ptr->buffer_map.end()) {
      args.push_back((*it).second->data);
      buffer_map_found = true;
    } else {
      args.push_back(param);
    }
  }

  if (buffer_map_found) {
    device_init.push_back(AttrStmt(node, attr::device_id, device_id, nop));
    device_init.push_back(AttrStmt(node, attr::device_type, device_type, nop));
  }

  func_ptr->body = MergeNest({device_init, binder.init_nest(), binder.asserts()}, func_ptr->body);
  func_ptr->params = args;
  func_ptr->ret_type = PrimType(DataType::Int(32));

  // if (global_symbol == "vm_mod_fused_sigmoid_tanh_multiply") {
  // ParamDebugger(func).Compute();
  // }

  // return the function.
  return std::move(func);
}

namespace transform {

Pass MakeUnpackedAPI() {
  auto pass_func = [](IRModule m, PassContext ctx) {
    // std::cout << "[OLA1]" << std::endl;
    IRModuleNode* mptr = m.CopyOnWrite();
    std::vector<std::pair<GlobalVar, PrimFunc>> updates;

    for (const auto& kv : mptr->functions) {
      if (auto* n = kv.second.as<PrimFuncNode>()) {
        PrimFunc func = GetRef<PrimFunc>(n);
        if (func->GetAttr<Integer>(tvm::attr::kCallingConv, Integer(CallingConv::kDefault)) ==
            CallingConv::kDefault) {
          auto updated_func = MakeUnpackedAPI(std::move(func));
          updates.push_back({kv.first, updated_func});
        }
      }
    }

    for (const auto& pair : updates) {
      mptr->AddUnchecked(pair.first, pair.second);
    }
    return m;
  };

  return tvm::transform::CreateModulePass(pass_func, 0, "tir.MakeUnpackedAPI", {});
}

TVM_REGISTER_GLOBAL("tir.transform.MakeUnpackedAPI").set_body_typed(MakeUnpackedAPI);
}  // namespace transform
}  // namespace tir
}  // namespace tvm
