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

////////////////////////////// Dynamic batching
bool DBPhaseChangeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  reporter->Assign(types[0], VoidType());
  return true;
}

Expr MakeDBPhaseChange() {
  static const Op& op = Op::Get("db.phase_change");
  return Call(op, {}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op.db._make.phase_change").set_body_typed(MakeDBPhaseChange);

RELAY_REGISTER_OP("db.phase_change")
    .describe(
        R"doc(Inform compiler/runtime of a program phase change for dynamic batching.)doc" TVM_ADD_FILELINE)
    .set_num_inputs(0)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", true)
    .add_type_rel("DBPhaseChange", DBPhaseChangeRel);

const Op& GetDBPhaseChangeOp() {
  static auto op = Op::Get("db.phase_change");
  return op;
}

}  // namespace relay
}  // namespace tvm
