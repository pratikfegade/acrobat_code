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
#ifndef TVM_RELAY_TRANSFORMS_FUNCTION_POINTER_ANALYSIS_H_
#define TVM_RELAY_TRANSFORMS_FUNCTION_POINTER_ANALYSIS_H_

#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

#include "../../support/utils.h"

namespace tvm {
namespace relay {

using FPAVarKey = std::pair<const FunctionNode*, const VarNode*>;
using FPAFunctionKey = std::pair<const FunctionNode*, const FunctionNode*>;
using FPAOpKey = std::pair<const FunctionNode*, const CallNode*>;

using PairHash = support::PairHash;
using PairEquals = support::PairEquals;

using FunctionSet = std::set<const FunctionNode*>;
using FPAVarStateMap = std::unordered_map<FPAVarKey, FunctionSet, PairHash, PairEquals>;
using FPAFunctionStateMap = std::unordered_map<FPAFunctionKey, FunctionSet, PairHash, PairEquals>;
using FPABaseExprFunctor = ExprFunctor<FunctionSet(const Expr& n)>;
using PreciseCallGraph = std::unordered_map<const FunctionNode*, FunctionSet>;
using CalleesMap = std::unordered_map<FPAOpKey, FunctionSet, PairHash, PairEquals>;
using CallDepthMap = std::unordered_map<const CallNode*, int>;

std::pair<FunctionSet, CalleesMap> GetRecursiveFunctions(const IRModule& mod);

CalleesMap GetCalleesMap(const IRModule& mod);

std::vector<Var> CollectPatternVars(const Pattern& p);

}  // namespace relay
}  // namespace tvm

#endif