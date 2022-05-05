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

#ifndef TVM_RELAY_BACKEND_BATCH_TE_GRAPH_H_
#define TVM_RELAY_BACKEND_BATCH_TE_GRAPH_H_

#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>

#include "../../te/schedule/graph.h"

namespace tvm {
namespace relay {
namespace tec {

std::pair<Map<te::Operation, te::Operation>, tir::Var> BatchifyTEGraph(
    const Array<te::Tensor>& inputs, const Array<te::Tensor>& outputs,
    const std::vector<bool>& reuse_taints, const std::string& unbatched_name);

std::pair<Array<te::Tensor>, Array<te::Tensor>> StaticBatchifyTEGraph(
    const Array<te::Tensor>& inputs, const Array<te::Tensor>& tensor_outs,
    const Array<Bool>& static_reuse_flags, const int static_batch_size);

}  // namespace tec
}  // namespace relay
}  // namespace tvm

#undef COUT
#endif  // TVM_RELAY_BACKEND_BATCH_TE_GRAPH_H_
