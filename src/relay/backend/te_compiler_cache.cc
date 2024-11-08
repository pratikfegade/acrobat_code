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

#include "./te_compiler_cache.h"

#include <tvm/driver/driver_api.h>
#include <tvm/ir/transform.h>
#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/dynamic_batching.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/topi/tags.h>

#include <functional>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../support/utils.h"
#include "../op/memory/memory.h"
#include "../transforms/pass_utils.h"
#include "batch_te_graph.h"
#include "utils.h"

namespace tvm {
namespace relay {
namespace tec {

TVM_REGISTER_NODE_TYPE(LoweredOutputNode);
TVM_REGISTER_NODE_TYPE(CachedFuncNode);
TVM_REGISTER_NODE_TYPE(CCacheKeyNode);
TVM_REGISTER_NODE_TYPE(CCacheValueNode);

LoweredOutput::LoweredOutput(tvm::Array<te::Tensor> outputs, OpImplementation impl) {
  auto n = make_object<LoweredOutputNode>();
  n->outputs = std::move(outputs);
  n->implementation = std::move(impl);
  data_ = std::move(n);
}

CCacheKey::CCacheKey(Function source_func, Target target, Array<Bool> static_reuse_flags,
                     int static_batch_size) {
  auto n = make_object<CCacheKeyNode>();
  n->source_func = std::move(source_func);
  n->target = std::move(target);
  n->static_reuse_flags = std::move(static_reuse_flags);
  n->static_batch_size = Integer(std::move(static_batch_size));
  data_ = std::move(n);
}

CachedFunc::CachedFunc(tvm::Target target, GlobalVar prim_fn_var,
                       tvm::Array<tir::Var> input_variables, tvm::Array<te::Tensor> inputs,
                       tvm::Array<te::Tensor> outputs, tvm::Array<Integer> batched_arg_mode,
                       te::Schedule schedule, tir::PrimFunc prim_func,
                       tvm::Array<Integer> shape_func_param_states, IRModule funcs,
                       tvm::String workload_key) {
  auto n = make_object<CachedFuncNode>();
  n->target = target;
  n->prim_fn_var = prim_fn_var;
  n->input_variables = input_variables;
  n->inputs = inputs;
  n->outputs = outputs;
  n->batched_arg_mode = batched_arg_mode;
  n->schedule = schedule;
  n->shape_func_param_states = shape_func_param_states;
  n->funcs = funcs;
  n->workload_key = workload_key;
  data_ = std::move(n);
}

Array<IndexExpr> GetShape(const Array<IndexExpr>& shape) {
  // for now, we always use int32 shape when possible
  // even if the result of shape inference becomes int64.
  Array<IndexExpr> res;
  for (IndexExpr val : shape) {
    const int64_t* pval = tir::as_const_int(val);
    if (pval != nullptr) {
#ifndef TVM_INDEX_DEFAULT_I64
      ICHECK_LE(pval[0], std::numeric_limits<int32_t>::max())
          << "dimension must be less then int32_t's max value";
      ICHECK_GE(pval[0], std::numeric_limits<int32_t>::min())
          << "dimension must be less then int32_t's max value";
      res.push_back(IntImm(DataType::Int(32), *pval));
#else
      res.push_back(val);
#endif  // TVM_INDEX_DEFAULT_I64
    } else if (val->IsInstance<tir::AnyNode>()) {
      // currently all 'any' we meet in shape function are non-negative.
      res.push_back(val.as<tir::AnyNode>()->ToSizeVar());
    } else {
      res.push_back(val);
    }
  }
  return res;
}

// Construct a schedule for a given Relay primitive function and target.
class ScheduleBuilder : public backend::MemoizedExprTranslator<Array<te::Tensor>> {
 public:
  explicit ScheduleBuilder(Target target, bool create_schedule = true)
      : target_(target),
        device_copy_op_(Op::Get("device_copy")),
        create_schedule_(create_schedule) {
    // Whether to use auto_scheduler schedule.
    use_auto_scheduler_ = backend::IsAutoSchedulerEnabled();
    use_meta_schedule_ = backend::IsMetaScheduleEnabled();
  }

  std::pair<CachedFunc, CachedFunc> Create(const Function& relay_func,
                                           std::function<std::string(std::string)> renamer,
                                           Array<Bool> model_parameter_taints,
                                           Array<Bool> static_reuse_flags, int static_batch_size,
                                           int task_weight, bool create_batched,
                                           bool scattered_kernels) {
    // std::cout << "Lowering " << task_weight << "\n" << relay_func << "\n\n" << std::endl;
    Array<tvm::te::Tensor> fn_inputs;
    int ctr = 0;
    for (Var param : relay_func->params) {
      Array<tvm::te::Tensor> inputs;
      for (const auto& ttype : FlattenTupleType(param->checked_type())) {
        tvm::te::Tensor tensor = tvm::te::placeholder(
            // GetShape(ttype->shape), ttype->dtype, param->vid->name_hint + std::to_string(ctr++));
            GetShape(ttype->shape), ttype->dtype, "placeholder" + std::to_string(ctr++));
        fn_inputs.push_back(tensor);
        inputs.push_back(tensor);
      }
      memo_[param] = inputs;
    }
    readable_name_stream_ << "fused";
    auto outputs = this->VisitExpr(relay_func->body);

    bool static_batched =
        (fn_inputs.size() > 0) && (static_reuse_flags.size() > 0) && (static_batch_size > 0);
    if (static_batched) {
      readable_name_stream_ << "_sb";
    }

    auto candidate_name = readable_name_stream_.str();
    constexpr static size_t kMaxFuncNameLength = 80;
    // WARNING: Please make sure to also update TVM_CRT_MAX_STRLEN_FUNCTION_NAME
    //          whenever the value of kMaxFuncNameLength changes
    if (candidate_name.size() > kMaxFuncNameLength) {
      std::stringstream truncated_name;
      truncated_name << candidate_name.substr(0, kMaxFuncNameLength);
      truncated_name << "_" << std::hash<std::string>{}(candidate_name) << "_";
      candidate_name = truncated_name.str();
    }

    // TODO(mbs): This should be the definitive global by which the
    // PrimFunc is known and no other GlobalVar ctors should appear
    // inside the lowering machinery.
    auto unique_name = renamer(candidate_name);
    auto prim_fn_var = GlobalVar(unique_name);
    prim_fn_var->checked_type_ = relay_func->checked_type();

    if (static_batched) {
      auto type = relay_func->checked_type().as<FuncTypeNode>();
      Array<Type> inputs;
      for (size_t i = 0; i < type->arg_types.size(); ++i) {
        size_t to_repeat = 1;
        if (!static_reuse_flags[i].operator bool()) {
          to_repeat = static_batch_size;
        }
        for (int j = 0; j < to_repeat; ++j) {
          inputs.push_back(type->arg_types[i]);
        }
      }

      Array<Type> outputs;
      if (auto rn = type->ret_type.as<TupleTypeNode>()) {
        for (auto field_type : rn->fields) {
          for (size_t i = 0; i < static_batch_size; ++i) {
            outputs.push_back(field_type);
          }
        }
      } else {
        for (size_t i = 0; i < static_batch_size; ++i) {
          outputs.push_back(type->ret_type);
        }
      }
      prim_fn_var->checked_type_ = FuncType(inputs, TupleType(outputs), {}, {});
    }

    // std::cout << "------------------------------------------------------ " << std::endl;
    // std::cout << unique_name << std::endl;
    // std::cout << relay_func << std::endl;
    // std::cout << fn_inputs << std::endl;
    // std::cout << outputs << std::endl;

    // Fusion over tupled results may leave identity relationships
    // between inputs and outputs, and those should not be scheduled.
    // Hence schedule only non PlaceholderOp outputs.
    tvm::Array<te::Tensor> tensor_outs;
    for (const auto& tensor : outputs) {
      if (!tensor->op.as<te::PlaceholderOpNode>()) {
        tensor_outs.push_back(tensor);
      }
    }

    if (static_batched) {
      auto res =
          StaticBatchifyTEGraph(fn_inputs, tensor_outs, static_reuse_flags, static_batch_size);
      auto new_fn_inputs = res.first;
      auto new_tensor_outs = res.second;
      Array<Bool> updated_model_parameter_taints;
      for (size_t i = 0; i < fn_inputs.size(); ++i) {
        size_t repeat_count = 1;
        if (!(static_reuse_flags[i].operator bool())) {
          repeat_count = static_batch_size;
        }
        for (size_t j = 0; j < repeat_count; ++j) {
          updated_model_parameter_taints.push_back(model_parameter_taints[i]);
        }
      }

      for (size_t i = 0; i < tensor_outs.size(); ++i) {
        for (size_t j = 0; j < static_batch_size; ++j) {
          updated_model_parameter_taints.push_back(model_parameter_taints[i + fn_inputs.size()]);
        }
      }
      model_parameter_taints = updated_model_parameter_taints;
      fn_inputs = new_fn_inputs;
      tensor_outs = new_tensor_outs;
    }

    int batched_task_weight = task_weight;
    int unbatched_task_weight = 1;
    if (!create_batched) {
      unbatched_task_weight = task_weight;
    }

    te::Schedule schedule{nullptr};
    String workload_key = "";
    tir::PrimFunc prim_func{nullptr};
    // No need to register schedule for device copy op.
    if (anchor_attrs_.as<DeviceCopyAttrs>() == nullptr && create_schedule_) {
      if (use_auto_scheduler_) {
        const auto* fauto_schedule =
            runtime::Registry::Get("auto_scheduler.relay_integration.auto_schedule_topi_compute");
        ICHECK(fauto_schedule != nullptr)
            << "auto_scheduler.relay_integration.auto_schedule_topi_compute is not registered";
        // std::cout << "[TCC] Invoking relay integration autoscheduler " << prim_fn_var->name_hint
        //           << std::endl;
        // for (auto to : tensor_outs) {
        //   std::cout << "[TCC]   Tensor: " << to << std::endl;
        // }

        ObjectRef obj =
            (*fauto_schedule)(prim_fn_var->name_hint, tensor_outs, unbatched_task_weight);
        if (obj.defined()) {
          auto arr = Downcast<Array<ObjectRef>>(obj);
          schedule = Downcast<te::Schedule>(arr[0]);

          if (arr[1].defined()) {
            workload_key = Downcast<String>(arr[1]);
          }
        }
      }
      if (use_meta_schedule_) {
        const auto* f_create_func = runtime::Registry::Get("te.CreatePrimFuncFromOutputs");
        const auto* f_meta_schedule =
            runtime::Registry::Get("meta_schedule.MetaScheduleContextQueryInsideWithScope");
        ICHECK(f_create_func) << "te.CreatePrimFuncFromOutputs is not registered";
        ICHECK(f_meta_schedule)
            << "meta_schedule.MetaScheduleContextQueryInsideWithScope is not registered ";
        prim_func = (*f_create_func)(tensor_outs);
        Optional<ObjectRef> opt_mod_or_base_func =
            (*f_meta_schedule)(prim_fn_var->name_hint, IRModule({{prim_fn_var, relay_func}}),
                               Array<IRModule>{IRModule({{prim_fn_var, prim_func}})});
        if (const auto* result = opt_mod_or_base_func.as<tir::PrimFuncNode>()) {
          prim_func = GetRef<tir::PrimFunc>(result);
        } else {
          prim_func = tir::PrimFunc(nullptr);
        }
      }

      // Use TOPI schedule if user specificed, or the function has no
      // auto_scheduler schedule.
      if (!schedule.defined() && !prim_func.defined()) {
        // PPF DEBUG
        // ICHECK(anchor_implementation_.defined());
        // schedule = anchor_implementation_.Schedule(anchor_attrs_, tensor_outs, target_);
        // PPF DEBUG

        Array<te::Operation> output_operations;
        for (auto tensor : tensor_outs) {
          output_operations.push_back(tensor->op);
        }
        schedule = te::create_schedule(output_operations);
      }
      if (schedule.defined()) {
        for (const auto& scalar : scalars_) {
          if (schedule->Contain(scalar)) {
            schedule[scalar].compute_inline();
          }
        }
      }
    }

    CachedFunc generated_prim_func =
        CachedFunc(target_, prim_fn_var, Array<tir::Var>(), fn_inputs, tensor_outs,
                   Array<Integer>(), schedule, prim_func, {}, {}, workload_key);
    CachedFunc generated_batched_func;

    // std::cout << "[TCC] Create batched functions? " << create_batched << std::endl;
    if (create_batched) {
      bool print = false;
      // bool print = (unique_name == "vm_mod_fused_zeros");

      if (fn_inputs.size() == 0) {
        Array<Bool> true_taints;
        for (size_t i = 0; i < tensor_outs.size(); ++i) {
          true_taints.push_back(Bool(true));
        }
        model_parameter_taints = true_taints;
        std::cout << "[TCC] TOTOTOTOTOTOTOTO " << unique_name << std::endl;
      }

      if (print) {
        std::cout << "[TCC] ParamTaint " << unique_name << " " << model_parameter_taints
                  << std::endl;
      }

      auto construct_reuse_taints = [&](int num_tensors, std::vector<bool>* p_reuse_taints,
                                        size_t offset) {
        for (size_t i = 0; i < num_tensors; ++i) {
          auto val = model_parameter_taints[i + offset].operator bool();
          p_reuse_taints->push_back(val);
        }
      };
      std::vector<bool> reuse_taints;
      construct_reuse_taints(fn_inputs.size(), &reuse_taints, 0);
      construct_reuse_taints(tensor_outs.size(), &reuse_taints, fn_inputs.size());

      auto res = BatchifyTEGraph(fn_inputs, tensor_outs, reuse_taints, unique_name);
      Map<te::Operation, te::Operation> mapping = res.first;
      tir::Var batch_size_var = res.second;

      auto map_tensors = [&](const Array<te::Tensor>& tensors, Array<Integer>* p_arg_modes,
                             size_t offset) {
        Array<te::Tensor> output;
        for (size_t i = 0; i < tensors.size(); ++i) {
          auto tensor = tensors[i];
          auto it = mapping.find(tensor->op);
          runtime::vm::DBBatchedArgMode arg_mode = runtime::vm::DBBatchedArgMode::kIgnore;
          if (it != mapping.end()) {
            auto operation = (*it).second;
            output.push_back(operation.output(tensor->value_index));
            arg_mode = runtime::vm::DBBatchedArgMode::kConcat;
            if (model_parameter_taints[i + offset].operator bool()) {
              arg_mode = runtime::vm::DBBatchedArgMode::kReuse;
            } else if (scattered_kernels) {
              arg_mode = runtime::vm::DBBatchedArgMode::kScatter;
            }
          }
          // std::cout << "[TECC]  Arg Mode " << tensor->op->name << " " << arg_mode << std::endl;
          p_arg_modes->push_back(tvm::Integer(static_cast<int>(arg_mode)));
        }
        return output;
      };

      ICHECK_EQ(fn_inputs.size() + tensor_outs.size(), model_parameter_taints.size());
      // std::cout << "[TECC] Function " << prim_fn_var->name_hint << "_batched" << std::endl;
      Array<Integer> arg_modes;
      Array<te::Tensor> batched_inputs = map_tensors(fn_inputs, &arg_modes, 0);
      Array<te::Tensor> batched_outputs = map_tensors(tensor_outs, &arg_modes, fn_inputs.size());

      auto tensor_to_batched_tensor_types = [](const Array<te::Tensor>& tensors) {
        Array<Type> tensor_types;
        for (auto tensor : tensors) {
          Array<PrimExpr> shape;
          shape.push_back(tir::Any());
          shape.push_back_all(tensor->shape);
          tensor_types.push_back(TensorType(shape, tensor->dtype));
        }
        return tensor_types;
      };

      Array<Type> input_tensor_types = tensor_to_batched_tensor_types(batched_inputs);
      Array<Type> output_tensor_types = tensor_to_batched_tensor_types(batched_outputs);

      auto batched_fn_var = GlobalVar(runtime::vm::GetBatchedName(unique_name));
      batched_fn_var->checked_type_ = FuncType(input_tensor_types, TupleType(output_tensor_types),
                                               Array<TypeVar>(), Array<TypeConstraint>());

      te::Schedule batched_schedule{nullptr};
      String batched_workload_key = "";
      // No need to register schedule for device copy op.
      // std::cout << "[TCC]  Other conditions " << (anchor_attrs_.as<DeviceCopyAttrs>() ==
      // nullptr)
      // << " " << create_schedule_ << " " << use_auto_scheduler_ << std::endl;
      if (anchor_attrs_.as<DeviceCopyAttrs>() == nullptr && create_schedule_) {
        if (use_auto_scheduler_) {
          int dynamic_batch_size_estimate =
              transform::PassContext::Current()
                  ->GetConfig<Integer>("relay.db_dynamic_batch_size_estimate", Integer(64))
                  .value();
          auto opt_dynamic_batch_size_estimate =
              relay_func->GetAttr<Integer>("DynamicBatchSizeEstimate");
          if (opt_dynamic_batch_size_estimate) {
            dynamic_batch_size_estimate = opt_dynamic_batch_size_estimate.value()->value;
          }
          // std::cout << "[SHPWTR] lowering "
          // << relay_func->GetAttr<Integer>("DynamicBatchSizeEstimate") << " "
          // << relay_func.get() << std::endl;
          Map<tir::Var, Integer> vmap{{batch_size_var, dynamic_batch_size_estimate}};
          const auto* fauto_schedule =
              runtime::Registry::Get("auto_scheduler.relay_integration.auto_schedule_topi_compute");
          ICHECK(fauto_schedule != nullptr)
              << "auto_scheduler.relay_integration.auto_schedule_topi_compute is not registered";
          // std::cout << "[TCC]   On batched " << batched_fn_var->name_hint << std::endl;
          // for (auto to : batched_outputs) {
          //   std::cout << "[TCC]    Tensor: " << to << " " << to->op.get() << std::endl;
          // }
          ObjectRef obj = (*fauto_schedule)(batched_fn_var->name_hint, batched_outputs,
                                            batched_task_weight, vmap);
          // if (obj.defined()) {
          // batched_schedule = Downcast<te::Schedule>(obj);
          // }
          if (obj.defined()) {
            auto arr = Downcast<Array<ObjectRef>>(obj);
            batched_schedule = Downcast<te::Schedule>(arr[0]);
            if (arr[1].defined()) {
              batched_workload_key = Downcast<String>(arr[1]);
            }
          }
        }
        ICHECK(!use_meta_schedule_) << "Do not support meta-schedule for batched operators yet!";

        // Use TOPI schedule if user specificed, or the function has no
        // auto_scheduler schedule.
        if (!batched_schedule.defined() && !prim_func.defined()) {
          Array<te::Operation> output_operations;
          for (auto tensor : batched_outputs) {
            output_operations.push_back(tensor->op);
          }

          batched_schedule = te::create_schedule(output_operations);
        }
        if (batched_schedule.defined()) {
          for (const auto& scalar : scalars_) {
            if (batched_schedule->Contain(scalar)) {
              batched_schedule[scalar].compute_inline();
            }
          }
        }
      }

      ICHECK(batched_schedule.defined());

      // std::cout << "[TEC] BatchedFnVar " << batched_fn_var << std::endl;
      generated_batched_func =
          CachedFunc(target_, batched_fn_var, Array<tir::Var>({batch_size_var}), batched_inputs,
                     batched_outputs, arg_modes, batched_schedule, tir::PrimFunc{nullptr}, {}, {},
                     batched_workload_key);
    }

    return std::make_pair(generated_prim_func, generated_batched_func);
  }

  Array<te::Tensor> VisitExpr_(const VarNode* op) final {
    LOG(FATAL) << "Unexpected free variable " << PrettyPrint(GetRef<Var>(op));
    return {};
  }

  Array<te::Tensor> VisitExpr_(const ConstantNode* op) final {
    using tir::make_const;
    ICHECK(op->is_scalar());
    void* data = op->data->data;
    DataType dtype = DataType(op->data->dtype);
    auto value = te::compute(
        {},
        [&](const Array<tvm::tir::Var>&) {
          if (dtype == DataType::Int(32)) {
            return make_const(dtype, static_cast<const int32_t*>(data)[0]);
          } else if (dtype == DataType::Int(64)) {
            return make_const(dtype, static_cast<const int64_t*>(data)[0]);
          } else if (dtype == DataType::Float(32)) {
            return make_const(dtype, static_cast<const float*>(data)[0]);
          } else if (dtype == DataType::Float(64)) {
            return make_const(dtype, static_cast<const double*>(data)[0]);
          } else if (dtype == DataType::Bool()) {
            return make_const(dtype, static_cast<const uint8_t*>(data)[0]);
          } else {
            LOG(FATAL) << "not handled";
            return tvm::PrimExpr();
          }
        },
        "compile_engine_const", topi::kBroadcast);
    scalars_.push_back(value->op);
    return {value};
  }

  Array<te::Tensor> VisitExpr_(const CallNode* call_node) final {
    static auto fpattern = Op::GetAttrMap<TOpPattern>("TOpPattern");
    static auto flower_call = tvm::runtime::Registry::Get("relay.backend.lower_call");
    ICHECK(flower_call) << "relay.backend.lower_call is not registered.";

    Array<te::Tensor> inputs;
    int count_tuple = 0;
    for (Expr arg : call_node->args) {
      if (arg->checked_type().as<TupleTypeNode>()) {
        ++count_tuple;
      }
      for (te::Tensor tensor : VisitExpr(arg)) {
        inputs.push_back(tensor);
      }
    }

    if (count_tuple) {
      ICHECK_EQ(call_node->args.size(), 1U)
          << "Only functions with a single tuple input are allowed, but " << count_tuple
          << " were provided.";
    }

    ICHECK(call_node->op.as<OpNode>()) << "Primitive function only allows call into primitive ops";
    Op op = Downcast<Op>(call_node->op);

    Array<te::Tensor> outputs;
    OpImplementation impl;
    // TODO(mbs): device_copy cleanup
    ICHECK_NE(op, device_copy_op_) << "device_copy cannot be lowered";
    LoweredOutput lowered_out = (*flower_call)(GetRef<Call>(call_node), inputs, target_);
    outputs = lowered_out->outputs;
    impl = lowered_out->implementation;

    if (create_schedule_) {
      int op_pattern = fpattern[op];
      if (!use_auto_scheduler_ && op_pattern >= kCommReduce) {
        ICHECK(!anchor_op_.defined() || anchor_op_pattern_ < kCommReduce)
            << "Cannot apply TOPI schedule to a primitive function with two complicated ops"
            << " anchor=" << anchor_op_ << " current=" << op;
      }
      if (op_pattern >= anchor_op_pattern_) {
        anchor_op_ = op;
        anchor_attrs_ = call_node->attrs;
        anchor_op_pattern_ = op_pattern;
        anchor_implementation_ = impl;
      }
    }
    if (outputs.size() != 1) {
      const auto* tuple_type = call_node->checked_type().as<TupleTypeNode>();
      ICHECK(tuple_type) << "Expected output to be a tuple type "
                         << PrettyPrint(call_node->checked_type());

      ICHECK_EQ(tuple_type->fields.size(), outputs.size());
    }

    // TODO(mbs): device_copy cleanup
    ICHECK_NE(op, device_copy_op_) << "device_copy cannot be lowered";
    readable_name_stream_ << '_' << op->name;
    return outputs;
  }

  Array<te::Tensor> VisitExpr_(const FunctionNode* op) final {
    LOG(FATAL) << "Primitive Functions can not contain nested functions.";
    return Array<te::Tensor>();
  }

  Array<te::Tensor> VisitExpr_(const LetNode* op) final {
    Array<te::Tensor> val = VisitExpr(op->value);
    ICHECK(!memo_.count(op->var));
    memo_[op->var] = val;
    return VisitExpr(op->body);
  }

  Array<te::Tensor> VisitExpr_(const TupleNode* op) final {
    Array<te::Tensor> fields;
    for (Expr field : op->fields) {
      // TODO(mbs): Generalize to be equivalent to FlattenTupleType.
      ICHECK(field->checked_type().as<TensorTypeNode>()) << "Only allow Tuple of Tensor";
      Array<te::Tensor> res = VisitExpr(field);
      ICHECK_EQ(res.size(), 1);
      fields.push_back(res[0]);
    }
    return fields;
  }

  Array<te::Tensor> VisitExpr_(const TupleGetItemNode* op) final {
    const auto* tuple_type = op->tuple->type_as<TupleTypeNode>();
    Array<te::Tensor> tuple = VisitExpr(op->tuple);
    ICHECK_EQ(tuple_type->fields.size(), tuple.size()) << " " << GetRef<Expr>(op);
    ICHECK_GE(op->index, 0);
    ICHECK_LT(static_cast<size_t>(op->index), tuple.size());
    return {tuple[op->index]};
  }

 private:
  tvm::Target target_;
  Op anchor_op_;
  Attrs anchor_attrs_;
  int anchor_op_pattern_{0};
  OpImplementation anchor_implementation_;
  std::ostringstream readable_name_stream_;
  Array<te::Operation> scalars_;
  bool use_auto_scheduler_;
  bool use_meta_schedule_;
  // Cache device copy op for equivalence checking to reduce registry lookup
  // overhead for each invocation of call node when retrieving schedules.
  const Op& device_copy_op_;
  bool create_schedule_;
};

/*!
 * \brief Create schedule for target.
 * \param source_func The primitive function to be lowered.
 * \param target The target we want to create schedule for.
 * \return Pair of schedule and cache.
 *  The funcs field in cache is not yet populated.
 */
std::pair<CachedFunc, CachedFunc> PrimFuncFor(const Function& source_func, const Target& target,
                                              std::function<std::string(std::string)> renamer,
                                              Array<Bool> model_parameter_taints,
                                              Array<Bool> static_reuse_flags, int static_batch_size,
                                              int task_weight, bool create_batched,
                                              bool scattered_kernels) {
  return ScheduleBuilder(target).Create(source_func, renamer, model_parameter_taints,
                                        static_reuse_flags, static_batch_size, task_weight,
                                        create_batched, scattered_kernels);
}

// Creates shape function from functor.
class MakeShapeFunc : public backend::MemoizedExprTranslator<Array<te::Tensor>> {
 public:
  MakeShapeFunc() {}

  CachedFunc Create(const Function& prim_func, const Target& target,
                    std::function<std::string(std::string)> renamer) {
    TShapeDataDependent shape_func_param_states;

    for (auto param : prim_func->params) {
      param_states_[param] = kNoNeed;
      Array<tvm::te::Tensor> data_inputs;
      Array<tvm::te::Tensor> shape_inputs;

      for (const auto& ttype : FlattenTupleType(param->checked_type())) {
        // Add data placeholder (in case we discover we need it below)
        Shape shape = GetShape(ttype->shape);
        tvm::te::Tensor data_tensor = tvm::te::placeholder(shape, ttype->dtype);
        data_inputs.push_back(data_tensor);
        // Add shape placeholder (in case we discover we need it below)
        int64_t ndim = shape.size();
        Shape sshape;
        if (ndim > 0) {
          sshape.push_back(tvm::Integer(ndim));
        }
        tvm::te::Tensor shape_tensor = tvm::te::placeholder(sshape, DataType::Int(64));
        shape_inputs.push_back(shape_tensor);
      }
      param_data_[param] = data_inputs;
      param_shapes_[param] = shape_inputs;
    }

    // Setup the name;
    readable_name_stream_ << "shape_func";

    // Create the `te::Tensor`s which represent the output.
    auto outputs = VisitExpr(prim_func->body);

    // Generate a name.
    auto candidate_name = readable_name_stream_.str();
    constexpr static size_t kMaxFuncNameLength = 80;
    // WARNING: Please make sure to also update TVM_CRT_MAX_STRLEN_FUNCTION_NAME
    //          whenever the value of kMaxFuncNameLength changes
    if (candidate_name.size() > kMaxFuncNameLength) {
      std::stringstream truncated_name;
      truncated_name << candidate_name.substr(0, kMaxFuncNameLength);
      truncated_name << "_" << std::hash<std::string>{}(candidate_name) << "_";
      candidate_name = truncated_name.str();
    }

    // Set all the inputs correctly, and accumulate their types from the p.o.v. of the
    // shape function rather than the primitive it is derived for.
    Array<te::Tensor> inputs;
    Array<Type> shape_function_arg_types;
    for (auto param : prim_func->params) {
      int state = param_states_[param];
      shape_func_param_states.push_back(IntImm(DataType::Int(32), state));
      if (state & kNeedInputData) {
        // Pass the primitive arguments directly (though in flattened form and on the host)
        for (auto t : param_data_[param]) {
          inputs.push_back(t);
          shape_function_arg_types.push_back(TensorType(t->GetShape(), t->GetDataType()));
        }
      }
      if (state & kNeedInputShape) {
        // Pass the shapes of the primitive arguments (also on the host)
        for (auto t : param_shapes_[param]) {
          inputs.push_back(t);
          shape_function_arg_types.push_back(TensorType(t->GetShape(), t->GetDataType()));
        }
      }
    }

    // TODO(mbs): This should be the definitive global by which the PrimFunc is known and
    // no  other GlobalVar ctors should appear inside the lowering machinery.
    auto func_name = renamer(candidate_name);
    auto prim_fn_gvar = GlobalVar(func_name);

    // Gather the result types, again from the p.o.v. of the shape function rather than
    // the primitive it is derived for.
    Array<Type> shape_function_res_types;
    for (const auto& t : outputs) {
      shape_function_res_types.push_back(TensorType(t->GetShape(), t->GetDataType()));
    }

    // Assign the shape function its true type.
    FuncType type(shape_function_arg_types, TupleType(shape_function_res_types),
                  /*type_params=*/{}, /*type_constraints=*/{});
    VLOG(1) << "shape function '" << prim_fn_gvar->name_hint << "' has type:" << std::endl
            << PrettyPrint(type) << std::endl
            << "corresponding to primitive of type:" << std::endl
            << PrettyPrint(prim_func->checked_type());
    prim_fn_gvar->checked_type_ = std::move(type);

    // generate schedule for shape func
    Array<te::Operation> out_ops;
    for (auto t : outputs) {
      out_ops.push_back(t->op);
    }
    auto schedule = te::create_schedule(out_ops);
    tvm::te::AutoInlineInjective(schedule);
    for (const auto& scalar : scalars_) {
      auto scalar_op = scalar->op;
      if (schedule->Contain(scalar_op)) {
        schedule[scalar_op].compute_inline();
      }
    }

    Array<te::Tensor> all_args = Array<te::Tensor>(inputs);
    for (te::Tensor arg : outputs) {
      all_args.push_back(arg);
    }

    using tvm::transform::PassContext;
    With<PassContext> fresh_pass_ctx_scope(PassContext::Create());

    std::unordered_map<te::Tensor, tir::Buffer> binds;
    Map<te::Tensor, tir::Buffer> scatter_buffers;
    IRModule lowered_module =
        tvm::LowerSchedule(schedule, all_args, func_name, binds, scatter_buffers);

    // Unfortunately the above machinery creates its own GlobalVars instead of using *the*
    // GlobalVar we established above. Fix this before the confusion spreads any further.
    // TODO(mbs): LowerSchedule should be given prim_fn_gvar instead of func_name.
    IRModule fixed_lowered_module;
    for (const auto& kv : lowered_module->functions) {
      GlobalVar global_var =
          kv.first->name_hint == prim_fn_gvar->name_hint ? prim_fn_gvar : kv.first;
      fixed_lowered_module->Add(global_var, kv.second);
    }
    return CachedFunc(target, prim_fn_gvar, Array<tir::Var>(), inputs, outputs, Array<Integer>(),
                      schedule, tir::PrimFunc{nullptr}, shape_func_param_states,
                      fixed_lowered_module);
  }

  Array<te::Tensor> VisitExpr(const Expr& expr) final {
    if (expr.as<VarNode>()) {
      // Do not memoize vars because shape functions could use either the data
      // or the shape of a var each time.
      return ExprFunctor::VisitExpr(expr);
    }
    // For other case, do memoized visit
    return backend::MemoizedExprTranslator<Array<te::Tensor>>::VisitExpr(expr);
  }

  Array<te::Tensor> VisitExpr_(const VarNode* var_node) final {
    auto var = GetRef<Var>(var_node);
    auto it = param_arg_map_.find(var);
    if (it != param_arg_map_.end()) {
      // This var is a parameter of a nested function. Visit the corresponding argument in the
      // function call site.
      return VisitExpr(it->second);
    }
    if (param_states_.find(var) == param_states_.end()) {
      LOG(FATAL) << "Unexpected free variable " << PrettyPrint(var);
      return {};
    } else {
      ICHECK(data_dependents_per_input_.size());
      auto data_dependent = data_dependents_per_input_.back();
      if (data_dependent) {
        param_states_[var] |= kNeedInputData;
        return param_data_[var];
      } else {
        param_states_[var] |= kNeedInputShape;
        return param_shapes_[var];
      }
    }
  }

  Array<te::Tensor> VisitExpr_(const ConstantNode* op) final {
    using tir::make_const;
    ICHECK(data_dependents_per_input_.size());
    bool data_dependent = data_dependents_per_input_.back();
    if (!op->is_scalar()) {
      // This is a constant weight, extract the shape of the weight tensor.
      // This can not be data dependent.
      CHECK(!data_dependent);
      auto ttype = op->checked_type().as<TensorTypeNode>();
      int ndim = static_cast<int>(ttype->shape.size());
      Array<PrimExpr> out_shape{ndim};
      te::Tensor value = tvm::te::compute(
          out_shape,
          [&](const Array<tvm::tir::Var>& indices) {
            auto idx = indices[0];
            PrimExpr ret = make_const(DataType::Int(64), 0);
            for (int i = 0; i < ndim; i++) {
              ret = tvm::if_then_else(idx == i, ttype->shape[i], ret);
            }
            return ret;
          },
          "shape_const", topi::kBroadcast);
      scalars_.push_back(value);
      return {value};
    }
    if (data_dependent) {
      void* data = op->data->data;
      DataType dtype = DataType(op->data->dtype);
      auto value = tvm::te::compute(
          {},
          [&](const Array<tvm::tir::Var>&) {
            if (dtype == DataType::Int(32)) {
              return make_const(dtype, static_cast<const int32_t*>(data)[0]);
            } else if (dtype == DataType::Int(64)) {
              return make_const(dtype, static_cast<const int64_t*>(data)[0]);
            } else if (dtype == DataType::Float(32)) {
              return make_const(dtype, static_cast<const float*>(data)[0]);
            } else if (dtype == DataType::Float(64)) {
              return make_const(dtype, static_cast<const double*>(data)[0]);
            } else if (dtype == DataType::Bool()) {
              return make_const(dtype, static_cast<const uint8_t*>(data)[0]);
            } else {
              LOG(FATAL) << "not handled";
              return tvm::PrimExpr();
            }
          },
          "data_const", topi::kBroadcast);
      scalars_.push_back(value);
      return {value};
    } else {
      auto value = tvm::te::compute(
          {}, [&](const Array<tvm::tir::Var>&) { return tir::make_const(DataType::Int(64), 0); },
          "shape_const", topi::kBroadcast);
      scalars_.push_back(value);
      return {value};
    }
  }

  Array<te::Tensor> VisitExpr_(const CallNode* call_node) final {
    if (auto* func = call_node->op.as<FunctionNode>()) {
      for (size_t i = 0; i < func->params.size(); ++i) {
        param_arg_map_[func->params[i]] = call_node->args[i];
      }
      return VisitExpr(func->body);
    }
    static auto fshape_func = Op::GetAttrMap<FShapeFunc>("FShapeFunc");
    static auto tshape_data_dependent = Op::GetAttrMap<TShapeDataDependent>("TShapeDataDependent");
    ICHECK(call_node->op.as<OpNode>()) << "Primitive function only allows call into primitive ops";
    Op op = Downcast<Op>(call_node->op);
    ICHECK(data_dependents_per_input_.empty() || !data_dependents_per_input_.back())
        << "Error in op fusion: output of the shape func is fed to a "
        << "data-dependent shape func";
    ICHECK_GT(fshape_func.count(op), 0) << "Internal error, cannot find ShapeFunc for " << op->name;
    ICHECK_GT(tshape_data_dependent.count(op), 0)
        << "Internal error, cannot find TShapeDataDependent for " << op->name;

    Array<Integer> dep_spec = tshape_data_dependent[op];
    if (dep_spec.size() == 1) {
      // This is for cases when data dependence is specified per op
      // Replicate 0 or 1 flag to all arguments
      for (size_t i = 1; i < call_node->args.size(); ++i) {
        dep_spec.push_back(dep_spec[0]);
      }
    }

    // Visit all inputs
    Array<te::Tensor> inputs;
    int count_tuple = 0;
    for (size_t i = 0; i < call_node->args.size(); ++i) {
      Expr arg = call_node->args[i];
      if (arg->checked_type().as<TupleTypeNode>()) {
        ++count_tuple;
      }
      data_dependents_per_input_.push_back(dep_spec[i]->value != 0);
      for (te::Tensor tensor : VisitExpr(arg)) {
        inputs.push_back(tensor);
      }
      data_dependents_per_input_.pop_back();
    }
    if (count_tuple) {
      ICHECK_EQ(call_node->args.size(), 1U) << "Only allow function with a single tuple input";
    }
    // Get output ndims
    auto ret_type = call_node->checked_type();
    Array<IndexExpr> out_ndims;
    if (const auto* ttype = ret_type.as<TensorTypeNode>()) {
      out_ndims.push_back(IntImm(DataType::Int(32), ttype->shape.size()));
    } else {
      auto rtype = ret_type.as<TupleTypeNode>();
      // TODO(@icemelon): Allow recursive tuple
      ICHECK(rtype);
      for (size_t i = 0; i < rtype->fields.size(); ++i) {
        auto ttype = rtype->fields[i].as<TensorTypeNode>();
        ICHECK(ttype);
        out_ndims.push_back(IntImm(DataType::Int(32), ttype->shape.size()));
      }
    }
    // Call shape function
    auto outputs = fshape_func[op](call_node->attrs, inputs, out_ndims);
    readable_name_stream_ << "_" << op->name;
    return outputs;
  }

  Array<te::Tensor> VisitExpr_(const FunctionNode* op) final {
    LOG(FATAL) << "Nested functions are not allowed to be visited.";
    return Array<te::Tensor>();
  }

  Array<te::Tensor> VisitExpr_(const LetNode* op) final {
    Array<te::Tensor> val = VisitExpr(op->value);
    ICHECK(!memo_.count(op->var));
    memo_[op->var] = val;
    return VisitExpr(op->body);
  }

  Array<te::Tensor> VisitExpr_(const TupleNode* op) final {
    Array<te::Tensor> fields;
    for (Expr field : op->fields) {
      ICHECK(field->checked_type().as<TensorTypeNode>())
          << "Expected a Tuple of Tensor, but got " << PrettyPrint(field->checked_type());
      Array<te::Tensor> res = VisitExpr(field);
      ICHECK_EQ(res.size(), 1);
      fields.push_back(res[0]);
    }
    return fields;
  }

  Array<te::Tensor> VisitExpr_(const TupleGetItemNode* op) final {
    Array<te::Tensor> input_shapes = VisitExpr(op->tuple);
    Array<te::Tensor> out;
    out.push_back(input_shapes[op->index]);
    return out;
  }

 private:
  /*! \brief String stream for function name */
  std::ostringstream readable_name_stream_;
  /*! \brief Map from parameter to its shape function usage state */
  std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual> param_states_;
  /*! \brief Map from parameter to list of data placeholder */
  std::unordered_map<Expr, Array<te::Tensor>, ObjectPtrHash, ObjectPtrEqual> param_data_;
  /*! \brief Map from parameter to list of shape placeholder */
  std::unordered_map<Expr, Array<te::Tensor>, ObjectPtrHash, ObjectPtrEqual> param_shapes_;
  /*! \brief Stack of data dependencies for shape function, specified per each op input */
  std::vector<bool> data_dependents_per_input_;
  /*! \brief Scalars used in the shape function */
  Array<te::Tensor> scalars_;
  /*! \brief Map from parameters of a nested function to corresponding arguments in a function
   * call site.
   */
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> param_arg_map_;
};

CachedFunc ShapeFuncFor(const Function& prim_func, const Target& target,
                        std::function<std::string(std::string)> renamer) {
  return MakeShapeFunc().Create(prim_func, target, renamer);
}

/*!
 * \brief Get unique name from name.
 * \param name The orginal name.
 * \return Updated name which is unique.
 */
std::string GetUniqueName(std::string name, std::unordered_map<std::string, int>* name_map_) {
  for (size_t i = 0; i < name.length(); ++i) {
    if (name[i] == '.') name[i] = '_';
  }
  while (true) {
    auto it = name_map_->find(name);
    if (it == name_map_->end()) {
      (*name_map_)[name] = 1;
      return name;
    } else {
      std::ostringstream os;
      os << name << "_" << it->second;
      ++(it->second);
      name = os.str();
    }
  }
  return name;
}

TVM_REGISTER_GLOBAL("relay.backend.LowerToTE").set_body_typed([](Function prim_func) {
  return ScheduleBuilder(tvm::Target("ext_dev"), false)
      .Create(
          prim_func, [&](std::string name) { return name; }, {}, {}, 1, 1, false, false)
      .first;
});

}  // namespace tec
}  // namespace relay
}  // namespace tvm
