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

#include "db_ops.h"

#include <tvm/relay/attrs/random.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

namespace tvm {
namespace relay {

////////////////////////////// Dynamic batching set phase
bool DBSetPhaseRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  reporter->Assign(types[0], TensorType({}, DataType::Int(32), true));
  reporter->Assign(types[1], VoidType());
  return true;
}

Expr MakeDBSetPhase(const Expr& phase) {
  static const Op& op = Op::Get("db.set_phase");
  return Call(op, {phase}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op.db._make.set_phase").set_body_typed(MakeDBSetPhase);

RELAY_REGISTER_OP("db.set_phase")
    .describe(
        R"doc(Inform compiler/runtime of a program phase change for dynamic batching.)doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("phase", "Tensor", "Program phase number to set")
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", true)
    .add_type_rel("DBPhaseChange", DBSetPhaseRel);

const Op& GetDBSetPhaseOp() {
  static auto op = Op::Get("db.set_phase");
  return op;
}

////////////////////////////// Dynamic batching ghost op
bool DBGhostOpRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  reporter->Assign(types[0], VoidType());
  return true;
}

Expr MakeDBGhostOp() {
  static const Op& op = Op::Get("db.ghost_op");
  auto res = Call(op, {}, Attrs(), {});
  res->checked_type_ = VoidType();
  return res;
}

TVM_REGISTER_GLOBAL("relay.op.db._make.ghost_op").set_body_typed(MakeDBGhostOp);

RELAY_REGISTER_OP("db.ghost_op")
    .describe(
        R"doc(Inform compiler/runtime of a ghost op for dynamic batching.)doc" TVM_ADD_FILELINE)
    .set_num_inputs(0)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", true)
    .add_type_rel("DBGhostOp", DBGhostOpRel);

const Op& GetDBGhostOpOp() {
  static auto op = Op::Get("db.ghost_op");
  return op;
}

}  // namespace relay
}  // namespace tvm
