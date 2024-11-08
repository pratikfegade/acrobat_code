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

#include <tvm/relay/attrs/random.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

#include "db_random.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(ThreefryGenerateAttrs);

static TensorType ThreefryKeyType() { return TensorType({10}, tvm::DataType::UInt(64)); }

bool ThreefryGenerateRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                         const TypeReporter& reporter) {
  const ThreefryGenerateAttrs* param = attrs.as<ThreefryGenerateAttrs>();
  ICHECK_EQ(types.size(), 2) << "ThreefryGenerate should have one input and one output";

  reporter->Assign(types[0], ThreefryKeyType());

  std::vector<IndexExpr> oshape;
  for (auto& x : param->out_shape) {
    oshape.push_back(x);
  }
  // generate returns the next key and an array of random values
  // TODO(@tkonolige, @altanh): support other output dtypes?
  reporter->Assign(types[1],
                   TupleType({ThreefryKeyType(), TensorType(oshape, tvm::DataType::UInt(64))}));
  return true;
}

Expr MakeThreefryGenerate(Expr key, Array<Integer> out_shape) {
  auto attrs = make_object<ThreefryGenerateAttrs>();
  attrs->out_shape = out_shape;
  static const Op& op = Op::Get("random.threefry_generate");
  return Call(op, {key}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.random._make.threefry_generate").set_body_typed(MakeThreefryGenerate);

RELAY_REGISTER_OP("random.threefry_generate")
    .describe(
        R"doc(Generate an array of random numbers using the Threefry algorithm.)doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_attrs_type<ThreefryGenerateAttrs>()
    .add_argument("key", "Tensor", "Input Threefry key")
    .add_type_rel("ThreefryGenerate", ThreefryGenerateRel);

bool ThreefrySplitRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2) << "ThreefrySplit should have one input and one output";

  reporter->Assign(types[0], ThreefryKeyType());
  reporter->Assign(types[1], TupleType({ThreefryKeyType(), ThreefryKeyType()}));

  return true;
}

Expr MakeThreefrySplit(Expr key) {
  static const Op& op = Op::Get("random.threefry_split");
  return Call(op, {key}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op.random._make.threefry_split").set_body_typed(MakeThreefrySplit);

RELAY_REGISTER_OP("random.threefry_split")
    .describe(R"doc(Split the input Threefry key into two new ones.)doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("key", "Tensor", "Input Threefry key")
    .add_type_rel("ThreefrySplit", ThreefrySplitRel);

TVM_REGISTER_NODE_TYPE(UniformAttrs);

bool UniformRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  const UniformAttrs* param = attrs.as<UniformAttrs>();
  ICHECK_EQ(types.size(), 4) << "Uniform should have three inputs and one output";

  std::vector<IndexExpr> oshape;
  for (auto& x : param->out_shape) {
    oshape.push_back(x);
  }
  DataType out_dtype = param->out_dtype;
  // we are supporting float32 and float64 at the moment.
  if (!(out_dtype.is_float() && (out_dtype.bits() == 32 || out_dtype.bits() == 64))) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "We only support generating uniform random value of "
                                     << "type float32 or float64, got " << out_dtype << ".");
    return false;
  }
  reporter->Assign(types[0], ThreefryKeyType());
  reporter->Assign(types[1], TensorType({}, out_dtype));
  reporter->Assign(types[2], TensorType({}, out_dtype));
  // generate returns the next key and an array of random values
  reporter->Assign(types[3], TupleType({ThreefryKeyType(), TensorType(oshape, out_dtype)}));
  return true;
}

Expr MakeUniform(Expr key, Expr low, Expr high, Array<Integer> out_shape, DataType out_dtype) {
  auto attrs = make_object<UniformAttrs>();
  attrs->out_shape = out_shape;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("random.uniform");
  return Call(op, {key, low, high}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.random._make.uniform").set_body_typed(MakeUniform);

RELAY_REGISTER_OP("random.uniform")
    .describe(
        R"doc(Generate an array of random numbers under uniform distribution.)doc" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .set_attrs_type<UniformAttrs>()
    .add_argument("key", "Tensor", "Input Threefry key")
    .add_argument("low", "Tensor", "Lower bound of the distribution")
    .add_argument("high", "Tensor", "Higher bound of the distribution")
    .add_type_rel("Uniform", UniformRel);

TVM_REGISTER_NODE_TYPE(NormalAttrs);

bool NormalRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  const NormalAttrs* param = attrs.as<NormalAttrs>();
  ICHECK_EQ(types.size(), 4) << "Normal should have three inputs and one output";

  std::vector<IndexExpr> oshape;
  for (auto& x : param->out_shape) {
    oshape.push_back(x);
  }
  DataType out_dtype = param->out_dtype;
  // we are supporting float32 and float64 at the moment.
  if (!(out_dtype.is_float() && (out_dtype.bits() == 32 || out_dtype.bits() == 64))) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "We only support generating Normal random value of "
                                     << "type float32 or float64, got " << out_dtype << ".");
    return false;
  }
  reporter->Assign(types[0], ThreefryKeyType());
  reporter->Assign(types[1], TensorType({}, out_dtype));
  reporter->Assign(types[2], TensorType({}, out_dtype));
  // generate returns the next key and an array of random values
  reporter->Assign(types[3], TupleType({ThreefryKeyType(), TensorType(oshape, out_dtype)}));
  return true;
}

Expr MakeNormal(Expr key, Expr mean, Expr scale, Array<Integer> out_shape, DataType out_dtype) {
  auto attrs = make_object<NormalAttrs>();
  attrs->out_shape = out_shape;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("random.normal");
  return Call(op, {key, mean, scale}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.random._make.normal").set_body_typed(MakeNormal);

RELAY_REGISTER_OP("random.normal")
    .describe(
        R"doc(Generate an array of random numbers under normal distribution.)doc" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .set_attrs_type<NormalAttrs>()
    .add_argument("key", "Tensor", "Input Threefry key")
    .add_argument("mean", "Tensor", "Mean of the distribution")
    .add_argument("scale", "Tensor", "Standard deviation of the distribution")
    .add_type_rel("Normal", NormalRel);

////////////////////////////// Dynamic batching
bool DBUniformRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  const UniformAttrs* param = attrs.as<UniformAttrs>();
  ICHECK_EQ(types.size(), 4) << "Uniform should have three inputs and one output";

  std::vector<IndexExpr> oshape;
  for (auto& x : param->out_shape) {
    oshape.push_back(x);
  }

  if (!oshape.size() == 0) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "We only support generating uniform random value of "
                                     << "scalars, got tensor of shape " << param->out_shape << ".");
  }
  DataType out_dtype = param->out_dtype;
  // // we are supporting float32 and float64 at the moment.
  // if (!(out_dtype.is_float() && (out_dtype.bits() == 32 || out_dtype.bits() == 64))) {
  //   reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
  //                                    << "We only support generating uniform random value of "
  //                                    << "type float32 or float64, got " << out_dtype << ".");
  //   return false;
  // }
  // we are supporting in32.
  if (!(out_dtype.is_int() && out_dtype.bits() == 32)) {
    reporter->GetDiagCtx().EmitFatal(Diagnostic::Error(reporter->GetSpan())
                                     << "We only support generating uniform random value of "
                                     << "type int32, got " << out_dtype << ".");
    return false;
  }
  reporter->Assign(types[0], TensorType({}, out_dtype, true));
  reporter->Assign(types[1], TensorType({}, out_dtype, true));
  // generate returns the next key and an array of random values
  reporter->Assign(types[2], types[2]);
  reporter->Assign(types[3], TensorType(oshape, out_dtype, true));

  return true;
}

Expr MakeDBRandomUniform(Expr low, Expr high, Expr dummy, Array<Integer> out_shape,
                         DataType out_dtype) {
  auto attrs = make_object<UniformAttrs>();
  attrs->out_shape = out_shape;
  attrs->out_dtype = out_dtype;
  static const Op& op = Op::Get("random.db_uniform");
  return Call(op, {low, high, dummy}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.random._make.db_uniform").set_body_typed(MakeDBRandomUniform);

RELAY_REGISTER_OP("random.db_uniform")
    .describe(
        R"doc(Generate an array of random numbers under uniform distribution.)doc" TVM_ADD_FILELINE)
    .set_num_inputs(3)
    .set_attrs_type<UniformAttrs>()
    .add_argument("low", "Tensor", "Lower bound of the distribution")
    .add_argument("high", "Tensor", "Higher bound of the distribution")
    .add_argument("dummy_tensors", "Tuple", "dummy tensors for dependency reasons")
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<TOpIsStateful>("TOpIsStateful", true)
    .add_type_rel("DBUniform", DBUniformRel);

const Op& GetDBRandomUniformOp() {
  static auto op = Op::Get("random.db_uniform");
  return op;
}

DBRandomUniformProps GetDBRandomUniformProps(const CallNode* call_node) {
  if (call_node->op == GetDBRandomUniformOp()) {
    ICHECK_EQ(call_node->args.size(), 3) << "db_random_uniform expects three argument";
    ICHECK(call_node->attrs.defined()) << "db_random_uniform requires attributes";
    const auto* db_random_uniform_attrs = call_node->attrs.as<UniformAttrs>();
    ICHECK(db_random_uniform_attrs != nullptr) << "db_random_uniform requires UniformAttrs";
    return {call_node->args[0], call_node->args[1], db_random_uniform_attrs->out_shape,
            db_random_uniform_attrs->out_dtype};
  }
  return {};
}

DBRandomUniformProps GetDBRandomUniformProps(const Expr& expr) {
  if (const auto* call_node = expr.as<CallNode>()) {
    return GetDBRandomUniformProps(call_node);
  }
  return {};
}

}  // namespace relay
}  // namespace tvm
