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
 * \file src/relay/transforms/fuse_ops.cc
 *
 * \brief This is a backend-aware optimization pass.
 *   Fuse necessary ops into a single one.
 */
#ifndef TVM_RELAY_TRANSFORMS_MARK_SCALAR_CALLS_H_
#define TVM_RELAY_TRANSFORMS_MARK_SCALAR_CALLS_H_

#include <tvm/ir/module.h>
#include <tvm/relay/expr.h>

namespace tvm {
namespace relay {
namespace transform {

std::unordered_map<const ExprNode*, std::vector<bool>> GetScalarificationTaints(const IRModule& m);

}
}  // namespace relay
}  // namespace tvm

#endif
