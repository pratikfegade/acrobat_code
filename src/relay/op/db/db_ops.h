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
 * \file relay/op/memory/device_copy.h
 * \brief Helpers for working with "device_copy" attributes.
 */

#ifndef TVM_RELAY_OP_DB_DB_OPS_H_
#define TVM_RELAY_OP_DB_DB_OPS_H_

#include <tvm/relay/expr.h>

#include <utility>

#include "../call/call.h"

namespace tvm {
namespace relay {

/*! \brief Returns the "phase change" operator. */
const Op& GetDBSetPhaseOp();

/*! \brief Create a call to the phase change op. */
Expr MakeDBSetPhase(const Expr& phase);

/*! \brief Returns the "phase change" operator. */
const Op& GetDBGhostOpOp();

/*! \brief Create a call to the phase change op. */
Expr MakeDBGhostOp();

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_DB_DB_OPS_H_
