// /*
//  * Licensed to the Apache Software Foundation (ASF) under one
//  * or more contributor license agreements.  See the NOTICE file
//  * distributed with this work for additional information
//  * regarding copyright ownership.  The ASF licenses this file
//  * to you under the Apache License, Version 2.0 (the
//  * "License"); you may not use this file except in compliance
//  * with the License.  You may obtain a copy of the License at
//  *
//  *   http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing,
//  * software distributed under the License is distributed on an
//  * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//  * KIND, either express or implied.  See the License for the
//  * specific language governing permissions and limitations
//  * under the License.
//  */

// /*!
//  * \file well_formed.cc
//  * \brief check that expression is well formed.
//  */
// #include <tvm/relay/analysis.h>
// #include <tvm/relay/expr_functor.h>
// #include <tvm/relay/pattern_functor.h>
// #include <tvm/runtime/logging.h>

// #include <unordered_set>

// namespace tvm {
// namespace relay {

// template <typename TaintType>
// class TaintAnalysis : public ExprVisitor {
//  public:
//   virtual TaintType GetTaint(const Var& var);

//  private:
//   std::unordered_map<Expr, TaintType, ObjectPtrHash, ObjectPtrEqual> taint_map;
// };

// bool ParameterTaintAnalysis(const Expr& e, Optional<DiagnosticContext> diag_ctx) {
//   return WellFormedChecker(diag_ctx).CheckWellFormed(e);
// }

// TVM_REGISTER_GLOBAL("relay.analysis.ParameterTaintAnalysis").set_body_typed([](Expr e) {
//   return ParameterTaintAnalysis(e);
// });

// }  // namespace relay
// }  // namespace tvm
