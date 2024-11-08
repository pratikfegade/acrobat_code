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

#ifndef TVM_RELAY_OP_RANDOM_DB_RANDOM_H_
#define TVM_RELAY_OP_RANDOM_DB_RANDOM_H_

#include <tvm/relay/expr.h>

#include <utility>

#include "../call/call.h"

namespace tvm {
namespace relay {

/*! \brief Returns the "device_copy" operator. */
const Op& GetDBRandomUniformOp();

/*!
 * \brief Wraps \p expr in a "device_copy" CallNode indicating it should be evaluated and
 * stored at \p src_se_scope but then copied to \p dst_se_scope.
 */
Expr MakeDBRandomUniform(Expr low, Expr high, Expr dummy, Array<Integer> out_shape,
                         DataType out_dtype);

/*! \brief Result of \p GetDBRandomPropsProps. */
struct DBRandomUniformProps {
  Expr low;
  Expr high;
  Array<Integer> out_shape;
  DataType out_dtype;

  DBRandomUniformProps() = default;

  DBRandomUniformProps(Expr low, Expr high, Array<Integer> out_shape, DataType out_dtype)
      : low(std::move(low)),
        high(std::move(high)),
        out_shape(std::move(out_shape)),
        out_dtype(std::move(out_dtype)) {}
};

/*!
 * \brief Returns the body expression, source, and destination \p SEScopes for \p call_node
 * if it is a "device_copy" CallNode. Otherwise returns the null expression and unconstrained
 * device and scopes.
 */
DBRandomUniformProps GetDBRandomUniformProps(const CallNode* call_node);

/*!
 * \brief Returns the body expression, source, and destination \p SEScopes for \p expr if it
 * is a "device_copy" Call. Otherwise returns the null expression and unconstrained device and
 * scopes.
 */
DBRandomUniformProps GetDBRandomUniformProps(const Expr& expr);

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_RANDOM_DB_RANDOM_H_
