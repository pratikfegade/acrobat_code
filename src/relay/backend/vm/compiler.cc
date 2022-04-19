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
 * \file src/relay/backend/vm/compiler.cc
 * \brief A compiler from relay::Module to the VM byte code.
 */

#include "compiler.h"

#include <tvm/driver/driver_api.h>
#include <tvm/ir/error.h>
#include <tvm/parser/parser.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/qnn/transform.h>
#include <tvm/relay/runtime.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/vm/dynamic_batching.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/te/operation.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../../../target/metadata_module.h"
#include "../../../target/source/codegen_source_base.h"
#include "../../op/annotation/annotation.h"
#include "../../op/memory/device_copy.h"
#include "../../op/op_common.h"
#include "../../transforms/device_aware_visitors.h"
#include "../../transforms/pass_utils.h"
#include "../utils.h"
#include "aot_compiler.h"
#include "compiler.h"

namespace tvm {
TVM_REGISTER_PASS_CONFIG_OPTION("relay.db_coarsen_granularity", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.db_lazy_execution", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.db_batched_execution", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.db_scattered_kernels", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.db_concurrent_execution", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.db_generate_aot_code", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.db_dynamic_batch_size_estimate", Integer);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.db_aot_output_directory", String);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.db_autoscheduler_pass", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.db_use_depth_tracking", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.db_perform_static_scheduling", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.db_model_name", String);

namespace relay {

namespace transform {

Pass LambdaLift();
Pass LabelOps();

Pass MemoryPlan() {
  auto f = tvm::runtime::Registry::Get("relay.transform.MemoryPlan");
  std::cout << "Obtained MemoryPlan function " << std::endl;
  ICHECK(f != nullptr) << "unable to load the memory planning pass";
  return (*f)();
}

Pass LiftConstants() {
  auto f = tvm::runtime::Registry::Get("relay.transform.LiftConstants");
  ICHECK(f != nullptr) << "unable to load the constant lifting pass";
  return (*f)();
}

}  // namespace transform

namespace vm {

using namespace tvm::runtime;
using namespace tvm::runtime::vm;
using namespace relay::transform;

/*! \brief The host device is always stored at device index 0. */
constexpr Index kHostDeviceIndex = 0;

// (@jroesch): VM passes, eventually declare as passes.
bool IsClosure(const Function& func);

// Represent a runtime object that's going to be matched by pattern match expressions
struct MatchValue {
  Type type;

  explicit MatchValue(Type typ) : type(typ) {}

  virtual ~MatchValue() {}
};
using MatchValuePtr = std::shared_ptr<MatchValue>;

// A runtime object that resides in a register
struct RegisterValue : MatchValue {
  // The register num
  RegName register_num;

  explicit RegisterValue(RegName reg, Type type) : MatchValue(type), register_num(reg) {}

  ~RegisterValue() {}
};

// The value is a field of another runtime object
struct AccessField : MatchValue {
  MatchValuePtr parent;
  // Tag of the constructor the field is associated with
  int32_t tag;
  // Field index
  size_t index;
  // Runtime register num after compiling the access field path
  RegName reg{-1};

  AccessField(MatchValuePtr parent, int32_t tag, size_t index, Type type)
      : MatchValue(type), parent(parent), tag(tag), index(index) {}

  ~AccessField() {}
};

/*!
 * \brief Condition in a decision tree
 */
struct ConditionNode {
  virtual ~ConditionNode() {}
};

using ConditionObjectPtr = std::shared_ptr<ConditionNode>;

/*!
 * \brief A var binding condition
 */
struct VarBinding : ConditionNode {
  Var var;
  MatchValuePtr val;

  VarBinding(Var var, MatchValuePtr val) : var(var), val(val) {}

  ~VarBinding() {}
};

/*!
 * \brief Compare the tag of the object
 */
struct TagCompare : ConditionNode {
  /*! \brief The object to be examined */
  MatchValuePtr obj;

  /*! \brief The expected tag */
  int target_tag;

  TagCompare(MatchValuePtr obj, size_t target) : obj(obj), target_tag(target) {}

  ~TagCompare() {}
};

using TreeObjectPtr = typename relay::TreeNode<ConditionObjectPtr>::pointer;
using TreeLeafNode = relay::TreeLeafNode<ConditionObjectPtr>;
using TreeLeafFatalNode = relay::TreeLeafFatalNode<ConditionObjectPtr>;
using TreeBranchNode = relay::TreeBranchNode<ConditionObjectPtr>;

Type GetFieldType(const Type& type, size_t index, const IRModule& module,
                  const Constructor constructor = NullValue<Constructor>()) {
  bool print = false;  //(!constructor.defined());
  if (print) {
    std::cout << "[FIELD_TYPE]  " << type << " " << index << " " << std::endl;
  }
  if (auto tc = type.as<TypeCallNode>()) {
    auto base_type_gv = tc->func.as<GlobalTypeVarNode>();
    auto base_type = module->LookupTypeDef(base_type_gv->name_hint);
    ICHECK(base_type.defined()) << type;
    ICHECK_EQ(base_type->type_vars.size(), tc->args.size());

    if (print) {
      std::cout << "[FIELD_TYPE]   Baseype " << base_type << std::endl;
    }

    Map<TypeVar, Type> bind_map;
    for (size_t i = 0; i < tc->args.size(); ++i) {
      auto type_var = base_type->type_vars[i];
      auto type = tc->args[i];
      bind_map.Set(type_var, type);
      if (print) {
        std::cout << "[FIELD_TYPE]     Bind " << type_var << " " << type << std::endl;
      }
    }
    if (constructor.defined()) {
      auto field_base_type = constructor->inputs[index];
      auto ret = Bind(field_base_type, bind_map);
      // std::cout << "[FIELD_TYPE]  Constructor " << type << " " << index << " " << constructor <<
      // " "
      // << ret << std::endl;
      return ret;
    }
  }
  return type;
}

TreeObjectPtr BuildDecisionTreeFromPattern(MatchValuePtr data, Pattern pattern,
                                           TreeObjectPtr then_branch, TreeObjectPtr else_branch,
                                           const IRModule& module) {
  if (pattern.as<PatternWildcardNode>()) {
    // We ignore wildcard binding since it's not producing new vars
    return then_branch;
  } else if (const auto* pvn = pattern.as<PatternVarNode>()) {
    auto cond = std::make_shared<VarBinding>(pvn->var, data);
    return TreeBranchNode::Make(cond, then_branch, else_branch);
  } else if (const auto* pcn = pattern.as<PatternConstructorNode>()) {
    auto tag = pcn->constructor->tag;

    size_t field_index = 0;
    for (auto& p : pcn->patterns) {
      auto d = std::make_shared<AccessField>(
          data, tag, field_index, GetFieldType(data->type, field_index, module, pcn->constructor));
      then_branch = BuildDecisionTreeFromPattern(d, p, then_branch, else_branch, module);
      field_index++;
    }
    auto cond = std::make_shared<TagCompare>(data, tag);
    return TreeBranchNode::Make(cond, then_branch, else_branch);
  } else {
    // std::cout << "[FIELD_TYPE] Pattern " << pattern << std::endl;
    const auto* pt = pattern.as<PatternTupleNode>();
    ICHECK(pt) << "unhandled case: " << AsText(pattern, false);
    size_t field_index = 0;
    for (auto& p : pt->patterns) {
      auto d = std::make_shared<AccessField>(data, 0, field_index,
                                             GetFieldType(data->type, field_index++, module));
      then_branch = BuildDecisionTreeFromPattern(d, p, then_branch, else_branch, module);
    }
    return then_branch;
  }
}

TreeObjectPtr BuildDecisionTreeFromClause(MatchValuePtr data, Clause clause,
                                          TreeObjectPtr else_branch, const IRModule& module) {
  return BuildDecisionTreeFromPattern(data, clause->lhs, TreeLeafNode::Make(clause->rhs),
                                      else_branch, module);
}

TreeObjectPtr BuildDecisionTreeFromClauses(MatchValuePtr data, tvm::Array<Clause> clauses,
                                           const IRModule& module) {
  // When nothing matches, the VM throws fatal error
  TreeObjectPtr else_branch = TreeLeafFatalNode::Make();
  // Start from the last clause
  for (auto it = clauses.rbegin(); it != clauses.rend(); ++it) {
    else_branch = BuildDecisionTreeFromClause(data, *it, else_branch, module);
  }
  return else_branch;
}

std::vector<int64_t> ToAllocTensorShape(NDArray shape) {
  std::vector<int64_t> raw_shape;
  if (shape->ndim == 0) {
    return raw_shape;
  }
  ICHECK_EQ(shape->ndim, 1u);
  ICHECK_EQ(shape->dtype.code, 0U) << "The dtype of constant shape must be int32 or int64, but got "
                                   << DLDataType2String(shape->dtype);
  ICHECK(shape->dtype.bits == 64 || shape->dtype.bits == 32)
      << "The dtype of constant shape must be int32 or int64, but got"
      << DLDataType2String(shape->dtype);

  if (shape->dtype.bits == 64) {
    int64_t* int_ptr = reinterpret_cast<int64_t*>(shape->data);
    for (auto i = 0; i < shape->shape[0]; i++) {
      raw_shape.push_back(int_ptr[i]);
    }
  } else {  // int32
    int32_t* int_ptr = reinterpret_cast<int32_t*>(shape->data);
    for (auto i = 0; i < shape->shape[0]; i++) {
      raw_shape.push_back(static_cast<int64_t>(int_ptr[i]));
    }
  }
  return raw_shape;
}

class TIRCalleeCollector : public tir::StmtExprVisitor {
 public:
  Array<String> Collect(const tir::PrimFunc& func) {
    StmtExprVisitor::VisitStmt(func->body);
    return callees_;
  }

 private:
  void VisitExpr_(const tir::CallNode* op) override {
    if (auto callee = op->args[0].as<tir::StringImmNode>()) {
      callees_.push_back(callee->value);
    } else if (auto gvn = op->op.as<GlobalVarNode>()) {
      callees_.push_back(gvn->name_hint);
    }
  }

  Array<String> callees_;
};

struct VMFunctionCompilerResult {
 public:
  Function compiled_function;
  VMFunction vm_func;
  std::unordered_map<size_t, Type> register_types;
  std::unordered_map<Index, Array<Type>> invoke_type_vars;
  std::unordered_map<Index, int32_t> get_field_tags;
  std::unordered_map<Index, DictAttrs> call_attrs;
  std::unordered_map<Index, std::array<Index, 4>> if_offsets;
};

int64_t NDToInt64(const NDArray& nd) {
  DLDevice cpu_ctx{kDLCPU, 0};
  NDArray cpu_array = nd.CopyTo(cpu_ctx);
  CHECK_EQ(DataType(cpu_array->dtype), DataType::Int(64));
  return reinterpret_cast<int64_t*>(cpu_array->data)[0];
}

class VMFunctionCompiler : DeviceAwareExprFunctor<void(const Expr& n)> {
 public:
  VMFunctionCompiler(VMCompilerContext* context, SEScope host_se_scope, bool batched_execution,
                     bool generate_register_type_information)
      : DeviceAwareExprFunctor(context->module),
        last_register_(0),
        registers_num_(0),
        context_(context),
        host_se_scope_(std::move(host_se_scope)),
        batched_execution_(batched_execution),
        generate_aot_information_(generate_register_type_information) {}

  VMFunctionCompilerResult Compile(const GlobalVar& var, const Function& func) {
    std::vector<Index> param_device_indexes;
    Function compiled_function = func;
    if (IsClosure(func)) {
      // After lifting we'll have functions of the form:
      //   fn(closure args) { fn(lifted function args) { body } }
      // But we want the closure's function to be:
      //   fn(closure args, lifter function args) { body }
      // Do that flattening on-the-fly here.
      Function inner_func = Downcast<Function>(func->body);
      std::vector<Var> params;
      Array<Type> param_types;
      std::vector<SEScope> param_se_scopes;
      params.reserve(func->params.size() + inner_func->params.size());
      param_se_scopes.reserve(func->params.size() + inner_func->params.size());
      param_device_indexes.reserve(func->params.size() + inner_func->params.size());
      for (size_t i = 0; i < func->params.size(); ++i) {
        params.emplace_back(func->params[i]);
        param_types.push_back(func->params[i]->checked_type_);
        SEScope param_se_scope = GetFunctionParamSEScope(func.get(), i);
        param_se_scopes.push_back(param_se_scope);
        param_device_indexes.push_back(GetDeviceIndex(param_se_scope));
      }
      for (size_t i = 0; i < inner_func->params.size(); ++i) {
        params.emplace_back(inner_func->params[i]);
        param_types.push_back(inner_func->params[i]->checked_type_);
        SEScope param_se_scope = GetFunctionParamSEScope(inner_func.get(), i);
        param_se_scopes.push_back(param_se_scope);
        param_device_indexes.push_back(GetDeviceIndex(param_se_scope));
      }
      std::vector<TypeVar> type_params;
      type_params.reserve(func->type_params.size() + inner_func->type_params.size());
      for (const auto& tyvar : func->type_params) {
        type_params.push_back(tyvar);
      }
      for (const auto& tyvar : inner_func->type_params) {
        type_params.push_back(tyvar);
      }
      Function flattened_func = Function(params, inner_func->body, inner_func->ret_type,
                                         type_params, func->attrs, func->span);
      auto function_type = FuncType(param_types, inner_func->ret_type, type_params, {});
      flattened_func->checked_type_ = function_type;
      compiled_function = flattened_func;
      VisitExpr(MaybeFunctionOnDevice(flattened_func, param_se_scopes,
                                      GetFunctionResultSEScope(inner_func.get())));
    } else {
      param_device_indexes.reserve(func->params.size());
      for (size_t i = 0; i < func->params.size(); ++i) {
        param_device_indexes.push_back(GetDeviceIndex(GetFunctionParamSEScope(func.get(), i)));
      }
      VisitExpr(func);
    }
    // std::cout << RemoveOnDeviceCalls(compiled_function) << std::endl;
    return {compiled_function,
            VMFunction(var->name_hint, params_, instructions_, registers_num_,
                       std::move(param_device_indexes)),
            register_types_,
            invoke_type_vars_,
            get_field_tags_,
            call_attrs_,
            if_offsets_};
  }

  /*! \brief Attrs objects for each op. */
  std::map<Index, Map<String, ObjectRef>> op_attrs;

  /*! \brief Attrs objects for each callsite. */
  std::map<Index, Map<String, ObjectRef>> callsite_attrs;

 protected:
  size_t NewRegister() { return registers_num_++; }

  void AddRegisterTypeInfo(size_t reg, const Type& type) {
    ICHECK(type.defined());
    if (generate_aot_information_) {
      auto it = register_types_.find(reg);
      if (it == register_types_.end()) {
        register_types_[reg] = type;
      } else {
        auto old_type = it->second;
        ICHECK_EQ(old_type, type) << "Register " << reg << " has multiple types";
      }
    }
  }

  // Returns the pc of the last instruction inserted
  inline Index Emit(const Instruction& instr) {
    size_t instruction_index = instructions_.size();
    VLOG(2) << "instruction[" << instruction_index << "] = " << instr;
    ICHECK((int)instr.op < 100) << "Invalid opcode " << (int)instr.op;
    switch (instr.op) {
      case Opcode::AllocADT:
      case Opcode::AllocTensor:
      case Opcode::AllocTensorReg:
      case Opcode::GetField:
      case Opcode::GetTag:
      case Opcode::LoadConst:
      case Opcode::LoadConsti:
      case Opcode::Invoke:
      case Opcode::AllocClosure:
      case Opcode::AllocStorage:
      case Opcode::ShapeOf:
      case Opcode::ReshapeTensor:
      case Opcode::Move:
      case Opcode::InvokeClosure:
      case Opcode::DeviceCopy:
        last_register_ = instr.dst;
        break;
      case Opcode::InvokePacked:
      case Opcode::If:
      case Opcode::Ret:
      case Opcode::Goto:
      case Opcode::Fatal:
        break;
    }
    instructions_.push_back(instr);
    return instructions_.size() - 1;
  }

  /*!
   * \brief Returns the "device index" to represent \p se_scope for primitives
   * in emitted code. Note that the host device is always at index 0.
   */
  Index GetDeviceIndex(const SEScope& se_scope) {
    ICHECK(!se_scope->IsFullyUnconstrained());
    auto itr = std::find(context_->se_scopes_.begin(), context_->se_scopes_.end(), se_scope);
    if (itr != context_->se_scopes_.end()) {
      return std::distance(context_->se_scopes_.begin(), itr);
    }

    ICHECK_GT(context_->se_scopes_.size(), 0);
    ICHECK_NE(se_scope, host_se_scope_);  // the host scope is always at index 0

    if (se_scope->device_type() == context_->se_scopes_.front()->device_type()) {
      // It's ok if we see distinct scopes which share the host device type. This is because
      // we allow the SEScope for the host to be different from the SEScope for primitive
      // operations which both happen to be on the same device (typically CPU).
      return 0;
    }

    // However, otherwise we allow at most one SEScope per device type.
    // TODO(mbs): This will eventually need to account for memory scopes somehow so device_copy
    // instructions can do the right thing.
    itr = std::find_if(context_->se_scopes_.begin() + 1, context_->se_scopes_.end(),
                       [&se_scope](const SEScope& existing_se_scope) {
                         return existing_se_scope->device_type() == se_scope->device_type();
                       });
    CHECK(itr == context_->se_scopes_.end())
        << "The VM does not currently support using more than one device with the same device type "
           "for primitives, however the program is using the distinct scopes "
        << se_scope << " and " << *itr << " of device type " << se_scope->device_type();

    ICHECK(se_scope != host_se_scope_);
    Index index = context_->se_scopes_.size();
    VLOG(2) << "se_scope[" << index << "] = " << se_scope;
    context_->se_scopes_.push_back(se_scope);

    return index;
  }

  using DeviceAwareExprFunctor<void(const Expr&)>::VisitExpr_;

  void VisitExpr_(const ConstantNode* const_node) final {
    // Check the shape is valid
    NDArray data = const_node->data;
    size_t const_index = context_->constants.size();
    auto con = GetRef<Constant>(const_node);
    Index device_index = GetDeviceIndex(GetSEScope(con));
    VLOG(2) << "constant[" << const_index << "] on device[" << device_index << "]";
    context_->const_device_indexes.push_back(device_index);
    auto const_ndarr = const_node->data;
    context_->constants.push_back(const_ndarr);
    auto new_register = NewRegister();

    // std::cout << "[COM] Encountered constant " << const_ndarr << std::endl;
    // std::cout << "[COM]   " << const_ndarr.Shape().size() << " " << const_ndarr.DataType()
    // << std::endl;
    if (const_ndarr.Shape().size() == 0 && const_ndarr.DataType() == DataType::Int(64)) {
      auto const_int = NDToInt64(const_ndarr);
      Emit(Instruction::LoadConsti(const_int, new_register));
      AddRegisterTypeInfo(new_register, PrimType(const_ndarr.DataType()));
    } else {
      Emit(Instruction::LoadConst(const_index, new_register));
      AddRegisterTypeInfo(new_register, const_node->checked_type_);
    }
  }

  void VisitExpr_(const VarNode* var_node) final {
    auto var = GetRef<Var>(var_node);
    auto reg_it = this->var_register_map_.find(var);
    ICHECK(reg_it != this->var_register_map_.end());
    last_register_ = reg_it->second;
  }

  void VisitExpr_(const TupleNode* tuple_node) final {
    auto tuple = GetRef<Tuple>(tuple_node);
    std::vector<Index> fields_registers;

    for (auto& field : tuple->fields) {
      this->VisitExpr(field);
      fields_registers.push_back(last_register_);
    }

    // TODO(@jroesch): use correct tag
    auto new_register = NewRegister();
    Emit(Instruction::AllocADT(0, tuple->fields.size(), fields_registers, new_register));
    if (!tuple_node->checked_type_.defined()) {
      std::cout << "[TYP] " << tuple << std::endl;
    }
    AddRegisterTypeInfo(new_register, tuple_node->checked_type_);
  }

  void VisitExpr_(const MatchNode* match_node) final {
    auto match = GetRef<Match>(match_node);

    this->VisitExpr(match->data);
    CompileMatch(match);
  }

  void PreVisitLetBinding_(const Var& var, const Expr& value) final {
    ICHECK(!value.as<FunctionNode>())
        << "unexpected function:" << std::endl
        << PrettyPrint(value) << std::endl
        << "bound to var '" << var->name_hint() << "'. Did you set opt_level = 2?";
    VisitExpr(value);
    var_register_map_.emplace(var, this->last_register_);
  }

  void VisitExpr_(const TupleGetItemNode* get_node) final {
    auto get = GetRef<TupleGetItem>(get_node);
    this->VisitExpr(get->tuple);
    auto tuple_register = last_register_;
    auto new_register = NewRegister();
    Emit(Instruction::GetField(tuple_register, get->index, new_register));
    AddRegisterTypeInfo(new_register, get_node->checked_type_);
  }

  void VisitExpr_(const GlobalVarNode* gvar) final {
    auto var = GetRef<GlobalVar>(gvar);
    auto func = context_->module->Lookup(var);
    auto it = context_->global_map.find(var);
    ICHECK(it != context_->global_map.end()) << PrettyPrint(var);
    // Allocate closure with zero free vars
    auto new_register = NewRegister();
    Emit(Instruction::AllocClosure(it->second, 0, {}, new_register));
    AddRegisterTypeInfo(new_register, func->checked_type_);
  }

  void VisitExpr_(const IfNode* if_node) final {
    this->VisitExpr(if_node->cond);

    size_t test_register = last_register_;

    auto new_register = NewRegister();
    this->Emit(Instruction::LoadConsti(1, new_register));
    AddRegisterTypeInfo(new_register, if_node->cond->checked_type_);
    auto after_cond = instructions_.size();
    auto target_register = last_register_;
    auto if_pc = this->Emit(Instruction::If(test_register, target_register, 0, 0));
    auto before_true = this->instructions_.size();
    this->VisitExpr(if_node->true_branch);

    // It saves the result of If-Else expression.
    auto merge_register = NewRegister();
    Emit(Instruction::Move(last_register_, merge_register));
    AddRegisterTypeInfo(merge_register, if_node->checked_type_);
    Emit(Instruction::Goto(0));

    // Finally store how many instructions there are in the
    // true branch.
    auto after_true = this->instructions_.size();
    auto true_branch_instructions = after_true - 1 - before_true;

    auto before_false = this->instructions_.size();
    this->VisitExpr(if_node->false_branch);
    size_t false_register = last_register_;

    // In else-branch, override the then-branch register
    Emit(Instruction::Move(false_register, merge_register));
    // Compute the total number of instructions
    // after generating false.
    auto after_false = this->instructions_.size();
    auto false_branch_instructions = after_false - before_false;

    // Now we will compute the jump targets in order
    // to properly patch the instruction with the
    // the requiste targets.

    // After we emit the true body, and false body,
    // we patch up the if instruction, and goto.
    auto true_offset = 1;
    auto false_offset = after_true - after_cond;
    instructions_[after_cond].if_op.true_offset = true_offset;
    instructions_[after_cond].if_op.false_offset = false_offset;

    // Patch the Goto.
    this->instructions_[after_true - 1].pc_offset = (after_false - after_true) + 1;

    Index true_start_offset = true_offset;
    Index true_end_offset = true_offset + true_branch_instructions;
    Index false_start_offset = false_offset;
    Index false_end_offset = false_offset + false_branch_instructions;

    if_offsets_[if_pc] = std::array<Index, 4>(
        {true_start_offset, true_end_offset, false_start_offset, false_end_offset});

    this->last_register_ = merge_register;
  }

  Index AddPrimFuncToContext(const std::string& name, const DictAttrs& attrs) {
    // std::cout << "[CO] Contexting " << name << std::endl;
    Index op_index;
    auto itr = context_->primitive_map.find(name);
    if (itr == context_->primitive_map.end()) {
      op_index = context_->primitive_map.size();
      context_->primitive_map.emplace(name, op_index);
    } else {
      op_index = itr->second;
    }
    // Capture the dictionary of attributes from the original
    // primitive function so that they can contribute to the hash of
    // the compiled primitive. This way we can distinguish primitives
    // with the same body expression but different attributes which
    // may arbitrarily influence code generation.
    op_attrs[op_index] = attrs->dict;
    return op_index;
  }

  void CollectAndRegisterTIRCallees(const Expr& func_var, const DictAttrs& attrs) {
    ICHECK(func_var.as<GlobalVarNode>()) << "Expecting function in invoke_tvm_op to be a global";
    auto global_var = Downcast<GlobalVar>(func_var);
    auto func = context_->module->Lookup(global_var);
    ICHECK(func.as<tir::PrimFuncNode>()) << "Can only invoke PrimFuncs from TIR";
    auto callees = TIRCalleeCollector().Collect(Downcast<tir::PrimFunc>(func));
    for (auto callee_name : callees) {
      AddPrimFuncToContext(callee_name, attrs);
      if (batched_execution_) {
        AddPrimFuncToContext(GetBatchedName(callee_name), attrs);
      }
    }
  }

  Index EmitInvokeTVMOp(const Expr& func, const Expr& inputs, const Expr& outputs,
                        const DictAttrs& attrs) {
    std::vector<Index> argument_registers;

    const auto* global_var_node = func.as<GlobalVarNode>();
    ICHECK(global_var_node) << "Expecting function in invoke_tvm_op to be a global";

    auto input_tuple = inputs.as<TupleNode>();
    ICHECK(input_tuple) << "internal error: invoke_tvm_op inputs must be a tuple,"
                        << "please file a bug in the memory manifestation pass";

    auto output_tuple = outputs.as<TupleNode>();
    ICHECK(output_tuple) << "internal error: invoke_tvm_op outputs must be a tuple,"
                         << "please file a bug in the memory manifestation pass";

    for (auto input : input_tuple->fields) {
      VisitExpr(input);
      argument_registers.push_back(last_register_);
    }

    for (auto output : output_tuple->fields) {
      ICHECK(output->IsInstance<VarNode>()) << "output should be var, found:" << std::endl
                                            << PrettyPrint(output);
      auto reg = var_register_map_.find(Downcast<Var>(output));
      ICHECK(reg != var_register_map_.end())
          << "internal error: all variables should be in the register mapping";
      argument_registers.push_back(reg->second);
    }

    auto op_index = AddPrimFuncToContext(global_var_node->name_hint, attrs);
    if (batched_execution_) {
      AddPrimFuncToContext(GetBatchedName(global_var_node->name_hint), attrs);
    }

    return Emit(Instruction::InvokePacked(op_index, argument_registers.size(),
                                          output_tuple->fields.size(), argument_registers));
  }

  void DeviceAwareVisitExpr_(const CallNode* call_node) final {
    DeviceCopyProps device_copy_props = GetDeviceCopyProps(call_node);
    CallLoweredProps call_lowered_props = GetCallLoweredProps(call_node);
    ICHECK(!call_lowered_props.lowered_func.defined());
    if (device_copy_props.body.defined()) {
      // TODO(mbs): device_copy cleanup.
      VisitExpr(device_copy_props.body);
      RegName src_reg = last_register_;
      Index src_index = GetDeviceIndex(device_copy_props.src_se_scope);
      Index dst_index = GetDeviceIndex(device_copy_props.dst_se_scope);
      // Since scopes distinguish by targets (including any target hosts) but at runtime we
      // deal only with devices, the copy may be unnecessary.
      if (src_index != dst_index) {
        auto new_register = NewRegister();
        Emit(Instruction::DeviceCopy(src_reg, src_index, dst_index, new_register));
        AddRegisterTypeInfo(new_register, device_copy_props.body->checked_type_);
      }
      return;
    }

    // Now we handle the case in which we are using an opaque operator used to define a
    // sub-dialect, such as memory allocation operations.
    if (call_node->op.as<OpNode>()) {
      OpMatch<void> matcher;
      matcher
          .Match(
              "vm.invoke_tvm_op",
              [this, call_node](const Array<Expr>& args, const Attrs& attrs,
                                const Array<Type>& type_arg) {
                ICHECK_EQ(args.size(), 3);
                auto pc = EmitInvokeTVMOp(args[0], args[1], args[2], Downcast<DictAttrs>(attrs));
                CollectAndRegisterTIRCallees(args[0], Downcast<DictAttrs>(attrs));
                if (call_node->attrs.defined() && call_node->attrs->IsInstance<DictAttrsNode>()) {
                  call_attrs_[pc] = Downcast<DictAttrs>(call_node->attrs);
                }
              })
          .Match("memory.alloc_tensor",
                 [this, call_node](const Array<Expr>& args, const Attrs& attrs,
                                   const Array<Type>& type_arg) {
                   // std::cout << "[TENSOR] Call type " << call_node->checked_type_ << std::endl;
                   ICHECK_EQ(args.size(), 3);

                   // Get the attributes.
                   auto alloc_attrs = attrs.as<AllocTensorAttrs>();
                   ICHECK(alloc_attrs != nullptr) << "must be the alloc tensor attrs";
                   auto dtype = alloc_attrs->dtype;

                   // The storage will be passed dynamically.
                   this->VisitExpr(args[0]);
                   auto storage_register = last_register_;

                   // The offset will be passed dynamically.
                   this->VisitExpr(args[1]);
                   auto offset_register = last_register_;

                   // If the shape is constant then we will emit a static tensor allocation
                   // instruction. It may be wrapped by an on_device, but it will be on the host
                   // which is assumed by the alloc_tensor instruction anyway.
                   auto const_shape = AsIgnoringOnDevice<ConstantNode>(args[2]);

                   auto tensor_register = NewRegister();
                   if (const_shape) {
                     NDArray shape = const_shape->data;
                     // TODO(@jroesch): we need to get an RFC done to standarize shape dtype
                     std::vector<int64_t> raw_shape = ToAllocTensorShape(shape);
                     // Add context field.
                     Emit(Instruction::AllocTensor(storage_register, offset_register, raw_shape,
                                                   dtype, tensor_register));
                   } else {
                     this->VisitExpr(args[2]);
                     auto shape_register = last_register_;
                     Emit(Instruction::AllocTensorReg(storage_register, offset_register,
                                                      shape_register, dtype, tensor_register));
                   }
                   AddRegisterTypeInfo(tensor_register, call_node->checked_type_);
                 })
          .Match("memory.alloc_storage",
                 [this, call_node](const Array<Expr>& args, const Attrs& attrs,
                                   const Array<Type>& type_arg) {
                   // std::cout << "[STORAGE] Call type " << call_node->checked_type_ << std::endl;
                   ICHECK_EQ(args.size(), 2);
                   // Compute the size of the allocation.
                   this->VisitExpr(args[0]);
                   auto size_register = last_register_;

                   ICHECK(args[1].as<ConstantNode>()) << args[1];  // Always a literal.
                   NDArray alignment_arr = args[1].as<ConstantNode>()->data;
                   ICHECK_EQ(alignment_arr->dtype.code, 0U)
                       << "The dtype of constant shape must be int32 or int64, but got "
                       << DLDataType2String(alignment_arr->dtype);
                   ICHECK_EQ(alignment_arr->dtype.bits, 64U);
                   Index alignment = reinterpret_cast<int64_t*>(alignment_arr->data)[0];

                   // Get the dtype hint from the attributes.
                   auto alloc_attrs = attrs.as<AllocStorageAttrs>();
                   ICHECK(alloc_attrs != nullptr) << "must be the AllocStorage attrs";
                   auto dtype = alloc_attrs->dtype;

                   auto storage_register = NewRegister();
                   Emit(Instruction::AllocStorage(size_register, alignment, dtype,
                                                  GetDeviceIndex(alloc_attrs->se_scope),
                                                  storage_register));
                   AddRegisterTypeInfo(storage_register, call_node->checked_type_);
                 })
          .Match("vm.shape_of",
                 [this, call_node](const Array<Expr>& args, const Attrs& attrs,
                                   const Array<Type>& type_arg) {
                   ICHECK_EQ(args.size(), 1U);
                   // Get the attributes.
                   const auto* shape_of_attrs = attrs.as<ShapeOfAttrs>();
                   ICHECK(shape_of_attrs) << "Must be the shape_of attrs";
                   ICHECK_EQ(shape_of_attrs->dtype.bits(), 64)
                       << "The dtype of shape of must be int64, but got"
                       << DLDataType2String(shape_of_attrs->dtype);
                   this->VisitExpr(args[0]);
                   auto shape_register = NewRegister();
                   Emit(Instruction::ShapeOf(last_register_, shape_register));
                   AddRegisterTypeInfo(shape_register, call_node->checked_type_);
                 })
          .Match("vm.reshape_tensor",
                 [this, call_node](const Array<Expr>& args, const Attrs& attrs,
                                   const Array<Type>& type_arg) {
                   ICHECK_EQ(args.size(), 2u);
                   this->VisitExpr(args[0]);
                   auto tensor_reg = last_register_;
                   this->VisitExpr(args[1]);
                   auto shape_reg = last_register_;
                   auto tensor_register = NewRegister();
                   Emit(Instruction::ReshapeTensor(tensor_reg, shape_reg, tensor_register));
                   AddRegisterTypeInfo(tensor_register, call_node->checked_type_);
                 })
          .Match("memory.kill",
                 [](const Array<Expr>& args, const Attrs& attrs, const Array<Type>& type_arg) {
                   LOG(FATAL) << "memory.kill is not yet supported";
                 });
      matcher(GetRef<Call>(call_node));
      return;
    }

    // In the case it's not one of these specialized operators we will generate code
    // for one of the "standard" cases.
    std::vector<Index> args_registers;

    // Evaluate the call arguments.
    for (auto arg : call_node->args) {
      VisitExpr(arg);
      args_registers.push_back(last_register_);
    }

    if (const auto* global_var_node = call_node->op.as<GlobalVarNode>()) {
      // In the case we are invoking a global we need to find its
      // global ID, and then check whether it is closure invocation
      // or whether it is a standard global, and emit the correct
      // calling convention.
      auto global = GetRef<GlobalVar>(global_var_node);
      auto it = context_->global_map.find(global);
      ICHECK(it != context_->global_map.end()) << PrettyPrint(global);
      VLOG(2) << "VisitExpr_: generating invoke for " << global->name_hint
              << " with func_index=" << it->second;

      // TODO(tvm-team):
      // Think about mixed call into global that is not a relay::Function
      // perhaps establish as an invariance(all functions in mod must be relay::Function)
      auto func = Downcast<Function>(context_->module->Lookup(global));

      auto new_register = NewRegister();
      if (IsClosure(func)) {
        auto arity = func->params.size();
        Emit(Instruction::AllocClosure(it->second, arity, args_registers, new_register));
        AddRegisterTypeInfo(new_register, call_node->checked_type_);
      } else {
        auto invoke_pc = Emit(Instruction::Invoke(it->second, args_registers, new_register));
        if (generate_aot_information_) {
          invoke_type_vars_[invoke_pc] = call_node->type_args;
          if (call_node->attrs.defined() && call_node->attrs->IsInstance<DictAttrsNode>()) {
            call_attrs_[invoke_pc] = Downcast<DictAttrs>(call_node->attrs);
          }
        }
        AddRegisterTypeInfo(new_register, call_node->checked_type_);
      }
    } else if (const auto* constructor_node = call_node->op.as<ConstructorNode>()) {
      // In the constructor case, we simply need to find its tag
      // and emit a call to allocate the data structure.
      auto constructor = GetRef<Constructor>(constructor_node);
      auto new_register = NewRegister();
      auto invoke_pc = Emit(Instruction::AllocADT(constructor->tag, call_node->args.size(),
                                                  args_registers, new_register));
      if (generate_aot_information_) {
        invoke_type_vars_[invoke_pc] = call_node->type_args;
      }
      AddRegisterTypeInfo(new_register, call_node->checked_type_);
    } else if (const auto* var_node = call_node->op.as<VarNode>()) {
      // If we are calling a variable, it must be the case that it is a closure so we
      // emit invoke closure here.
      VisitExpr(GetRef<Var>(var_node));
      auto new_register = NewRegister();
      Emit(Instruction::InvokeClosure(last_register_, args_registers, new_register));
      AddRegisterTypeInfo(new_register, call_node->checked_type_);
    } else if (auto inner_call_node = call_node->op.as<CallNode>()) {
      VisitExpr(GetRef<Call>(inner_call_node));
      auto new_register = NewRegister();
      Emit(Instruction::InvokeClosure(last_register_, args_registers, new_register));
      AddRegisterTypeInfo(new_register, call_node->checked_type_);
    } else {
      // Finally if there are any other cases this is a bug.
      LOG(FATAL) << "internal error: unreachable code,"
                 << "should be transformed away by previous passes:" << std::endl
                 << PrettyPrint(GetRef<Expr>(call_node));
    }
  }

  void DeviceAwareVisitExpr_(const FunctionNode* func_node) final {
    if (function_nesting() > 1) {
      ICHECK(func_node->HasNonzeroAttr(attr::kPrimitive))
          << "local functions should have been removed by lambda lifting:" << std::endl
          << "Program: " << AsText(GetRef<Function>(func_node), false) << std::endl
          << "AST: " << GetRef<Function>(func_node);
      return;
    }

    // We're processing a top-level function which has possibly been rejigged to capture
    // both closure and function arguments. Those functions retain their 'Closure' attribute,
    // but we can just process them like any other function here.

    // Assign a register num to each parameter.
    size_t i = 0;
    for (auto param : func_node->params) {
      auto arg_register = NewRegister();
      ICHECK_EQ(i, arg_register);
      var_register_map_.insert({param, arg_register});
      params_.push_back(param->name_hint());
      ++i;
      AddRegisterTypeInfo(arg_register, param->checked_type_);
    }

    VisitExpr(func_node->body);

    instructions_.push_back(Instruction::Ret(last_register_));
  }

  /*!
   * \brief Compile a match value
   * Generate byte code that compute the value specificed in val
   *
   * \return The register number assigned for the final value
   */
  RegName CompileMatchValue(MatchValuePtr val) {
    if (std::dynamic_pointer_cast<RegisterValue>(val)) {
      auto r = std::dynamic_pointer_cast<RegisterValue>(val);
      return r->register_num;
    } else {
      auto path = std::dynamic_pointer_cast<AccessField>(val);
      auto p = CompileMatchValue(path->parent);
      auto field_register = NewRegister();
      auto get_field_pc = Emit(Instruction::GetField(p, path->index, field_register));
      if (generate_aot_information_) {
        get_field_tags_[get_field_pc] = path->tag;
      }
      AddRegisterTypeInfo(field_register, path->type);
      path->reg = last_register_;
      return path->reg;
    }
  }

  void CompileTreeNode(TreeObjectPtr tree) {
    if (auto node = std::dynamic_pointer_cast<TreeLeafNode>(tree)) {
      VisitExpr(node->body);
    } else if (std::dynamic_pointer_cast<TreeLeafFatalNode>(tree)) {
      Emit(Instruction::Fatal());
    } else if (auto node = std::dynamic_pointer_cast<TreeBranchNode>(tree)) {
      if (auto cond = std::dynamic_pointer_cast<TagCompare>(node->cond)) {
        // For Tag compariton, generate branches
        auto r = CompileMatchValue(cond->obj);
        Emit(Instruction::GetTag(r, NewRegister()));
        auto operand1 = last_register_;
        AddRegisterTypeInfo(last_register_, PrimType(DataType::Int(64)));
        Emit(Instruction::LoadConsti(cond->target_tag, NewRegister()));
        auto operand2 = last_register_;
        AddRegisterTypeInfo(last_register_, PrimType(DataType::Int(64)));

        auto if_pc = Emit(Instruction::If(operand1, operand2, 1, 0));
        auto cond_offset = instructions_.size() - 1;
        auto before_true = instructions_.size();
        CompileTreeNode(node->then_branch);
        auto after_true = instructions_.size();
        auto if_reg = last_register_;
        Emit(Instruction::Goto(1));
        auto goto_offset = instructions_.size() - 1;
        auto before_false = instructions_.size();
        CompileTreeNode(node->else_branch);
        auto else_reg = last_register_;
        Emit(Instruction::Move(else_reg, if_reg));
        auto after_false = instructions_.size();
        last_register_ = if_reg;
        auto else_offset = instructions_.size() - 1;
        // Fixing offsets
        instructions_[cond_offset].if_op.false_offset = goto_offset - cond_offset + 1;
        instructions_[goto_offset].pc_offset = else_offset - goto_offset + 1;

        auto true_start_offset = 1;
        auto true_end_offset = true_start_offset + (after_true - before_true);
        auto false_start_offset = goto_offset - cond_offset + 1;
        auto false_end_offset = false_start_offset + (after_false - before_false);

        if_offsets_[if_pc] = std::array<Index, 4>(
            {true_start_offset, true_end_offset, false_start_offset, false_end_offset});
      } else {
        // For other non-branch conditions, move to then_branch directly
        auto var_bind = std::dynamic_pointer_cast<VarBinding>(node->cond);
        var_register_map_[var_bind->var] = CompileMatchValue(var_bind->val);
        CompileTreeNode(node->then_branch);
      }
    }
  }

  /*!
   * \brief Compile a pattern match expression
   * It first converts the pattern match expression into a decision tree, the condition
   * could be object comparison or variable binding. If any of the condition fails in a clause,
   * the decision tree switches to check the conditions of next clause and so on. If no clause
   * matches the value, a fatal node is inserted.
   *
   * After the decision tree is built, we convert it into bytecodes using If/Goto.
   */
  void CompileMatch(Match match) {
    auto data = std::make_shared<RegisterValue>(last_register_, match->data->checked_type_);
    auto decision_tree = BuildDecisionTreeFromClauses(data, match->clauses, context_->module);
    CompileTreeNode(decision_tree);
  }

 protected:
  /*! \brief Store the expression a variable points to. */
  std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual> expr_map_;
  /*! \brief Instructions in the VMFunction. */
  std::vector<Instruction> instructions_;
  /*! \brief Parameter names of the function. */
  std::vector<std::string> params_;
  /*! \brief Map from var to register number. */
  std::unordered_map<Var, RegName, ObjectPtrHash, ObjectPtrEqual> var_register_map_;
  /*! \brief Last used register number. */
  size_t last_register_;
  /*! \brief Total number of virtual registers allocated. */
  size_t registers_num_;
  /*! \brief Global shared meta data */
  VMCompilerContext* context_;
  /*! \brief SEScope for data and computation which must reside on a CPU. */
  SEScope host_se_scope_;
  /*! \brief create code for batched execution. */
  bool batched_execution_;
  /*! \brief generate additional type information for AOT code generation. */
  bool generate_aot_information_;
  /*! \brief Type information for registers, used in AOT code generation. */
  std::unordered_map<size_t, Type> register_types_;
  /*! \brief Type var information for calls, used in AOT code generation. */
  std::unordered_map<Index, Array<Type>> invoke_type_vars_;
  /*! \brief Tag information for GetField, used in AOT code generation. */
  std::unordered_map<Index, int32_t> get_field_tags_;
  /*! \brief Statically computed depth information, used in AOT code generation. */
  std::unordered_map<Index, DictAttrs> call_attrs_;
  /*! \brief If offset information, used in AOT code generation. */
  std::unordered_map<Index, std::array<Index, 4>> if_offsets_;
};

PackedFunc VMCompiler::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  if (name == "lower") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.num_args, 3);
      this->Lower(args[0], args[1], args[2]);
    });
  } else if (name == "codegen") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.num_args, 0);
      this->Codegen();
    });
  } else if (name == "get_executable") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = runtime::Module(exec_); });
  } else if (name == "set_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      Map<String, Constant> params = args[0];
      for (const auto& kv : params) {
        this->SetParam(kv.first, kv.second->data);
      }
    });
  } else if (name == "get_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      Map<String, Constant> ret;
      for (const auto& kv : params_) {
        ret.Set(kv.first, Constant(kv.second));
      }
      *rv = ret;
    });
  } else if (name == "optimize") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_EQ(args.num_args, 3);
      *rv = this->OptimizeModule(args[0], args[1], args[2]);
    });
  } else {
    LOG(FATAL) << "Unknown packed function: " << name;
    return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
  }
}

void VMCompiler::SetParam(const std::string& name, runtime::NDArray data_in) {
  params_[name] = data_in;
}

void VMCompiler::Lower(IRModule mod, TargetMap targets, tvm::Target target_host) {
  VLOG_CONTEXT << "VM Lower";
  exec_ = make_object<Executable>();
  config_ = CompilationConfig(PassContext::Current(), std::move(targets), std::move(target_host));

  // The first device is always for the host.
  CHECK(context_.se_scopes_.empty());
  VLOG(2) << "se_scope[0] = " << config_->host_se_scope << " (host)";
  context_.se_scopes_.push_back(config_->host_se_scope);

  // Run the optimizations necessary to target the VM.
  context_.module = OptimizeModuleImpl(std::move(mod));

  // Build the map from global variables bound to Functions to a global index in the
  // VMFunction table.
  size_t num_functions = PopulateGlobalMap();

  // Next we get ready by allocating space for
  // the global state.
  exec_->functions.resize(num_functions);

  std::vector<std::string> prim_func_names;
  for (const auto& pair : context_.module->functions) {
    if (pair.second.as<tir::PrimFuncNode>()) {
      prim_func_names.push_back(pair.first->name_hint);
    }
  }
  std::sort(prim_func_names.begin(), prim_func_names.end());
  for (auto& name : prim_func_names) {
    if (!context_.primitive_map.count(name)) {
      context_.primitive_map.emplace(name, context_.primitive_map.size());
    }
  }

  bool batched_execution =
      PassContext::Current()->GetConfig<Bool>("relay.db_batched_execution", Bool(false)).value();
  std::unordered_map<std::string, std::unordered_map<size_t, Type>> register_types;
  std::unordered_map<std::string, std::unordered_map<Index, Array<Type>>> invoke_type_vars;
  std::unordered_map<std::string, Function> compiled_functions;
  std::unordered_map<std::string, std::unordered_map<Index, int32_t>> get_field_tags;
  std::unordered_map<std::string, std::unordered_map<Index, DictAttrs>> call_attrs;
  std::unordered_map<std::string, std::unordered_map<Index, std::array<Index, 4>>> if_offsets;
  for (const auto& pair : context_.module->functions) {
    auto gvar = pair.first;
    if (auto* n = pair.second.as<FunctionNode>()) {
      if (n->GetAttr<String>(attr::kExternalSymbol).defined()) {
        // Already compiled during lowering.
        continue;
      }
      auto func = GetRef<Function>(n);
      VMFunctionCompiler func_compiler(&context_, config_->host_se_scope, batched_execution, true);
      auto result = func_compiler.Compile(gvar, func);
      auto vm_func = result.vm_func;
      auto func_register_types = result.register_types;
      auto func_invoke_type_vars = result.invoke_type_vars;
      register_types[vm_func.name] = func_register_types;
      invoke_type_vars[vm_func.name] = func_invoke_type_vars;
      compiled_functions[vm_func.name] = result.compiled_function;
      get_field_tags[vm_func.name] = result.get_field_tags;
      call_attrs[vm_func.name] = result.call_attrs;
      if_offsets[vm_func.name] = result.if_offsets;

      size_t func_index = context_.global_map.at(gvar);
      ICHECK(func_index < exec_->functions.size());
      exec_->functions[func_index] = vm_func;

      // update structural hashes for tvm ops
      for (auto p : func_compiler.op_attrs) {
        exec_->op_attrs.insert(p);
      }
    }
  }

  // Populate virtual devices and the host device index.
  for (const auto& se_scope : context_.se_scopes_) {
    ICHECK(!se_scope->IsFullyUnconstrained());
    ICHECK_GT(se_scope->device_type(), 0);
    // TODO(mbs): We forget the memory scope.
    exec_->virtual_devices.push_back(Device{/*device_type=*/se_scope->device_type(),
                                            /*device_id=*/se_scope->virtual_device_id});
  }
  exec_->host_device_index = kHostDeviceIndex;

  // populate constants
  for (const auto& data : context_.constants) {
    exec_->constants.push_back(data);
  }

  for (auto index : context_.const_device_indexes) {
    exec_->const_device_indexes.push_back(index);
  }

  // update global function map
  for (const auto& gv : context_.global_map) {
    exec_->global_map.insert({gv.first->name_hint, gv.second});
  }

  // update primitive function map
  for (const auto& pair : context_.primitive_map) {
    exec_->primitive_map.insert(pair);
  }

  auto arg_modes = context_.module->batched_arg_modes;
  // std::cout << "[CO] ARGMODES2" << std::endl;
  // for (auto it : arg_modes) {
  //   std::cout << "[CO]  " << it.first->name_hint << " " << it.second << std::endl;
  // }
  // update batched arg modes
  for (auto pair : arg_modes) {
    ICHECK(exec_->primitive_map.count(pair.first->name_hint)) << pair.first->name_hint;
    auto index = exec_->primitive_map.at(pair.first->name_hint);
    if (static_cast<Index>(exec_->batched_func_arg_mode.size()) <= index) {
      exec_->batched_func_arg_mode.resize(index + 1);
    }
    auto modes = pair.second;
    std::vector<DBBatchedArgMode> arg_modes_vec;
    arg_modes_vec.reserve(modes.size());
    for (auto mode : modes) {
      arg_modes_vec.push_back(static_cast<DBBatchedArgMode>(mode->value));
    }
    exec_->batched_func_arg_mode[index] = std::move(arg_modes_vec);
  }

  for (auto pair : context_.module->functions) {
    if (pair.second.as<tir::PrimFuncNode>()) {
      auto it = exec_->primitive_map.find(pair.first->name_hint);
      ICHECK(it != exec_->primitive_map.end());
      auto index = (*it).second;
      if (static_cast<Index>(exec_->prim_func_arg_access_mode.size()) <= index) {
        exec_->prim_func_arg_access_mode.resize(index + 1);
      }
      auto access_modes = pair.second->GetAttr<Array<Integer>>(tir::attr::kDBArgAccessModes);
      if (access_modes) {
        std::vector<DBArgAccessMode> access_modes_vec;
        access_modes_vec.reserve(access_modes.value().size());
        for (auto mode : access_modes.value()) {
          access_modes_vec.push_back(static_cast<DBArgAccessMode>(mode->value));
        }
        exec_->prim_func_arg_access_mode[index] = std::move(access_modes_vec);

        if (false) {
          std::cout << "[COMP]   ArgAccessModes: [";
          for (size_t i = 0; i < exec_->prim_func_arg_access_mode[index].size(); ++i) {
            std::cout << exec_->prim_func_arg_access_mode[index][i] << " ";
          }
          std::cout << "]" << std::endl;
        }
      }
    }
  }

  for (auto pair : arg_modes) {
    ICHECK(exec_->primitive_map.count(pair.first->name_hint)) << pair.first->name_hint;
    auto index = exec_->primitive_map.at(pair.first->name_hint);
    if (static_cast<Index>(exec_->batched_func_arg_mode.size()) <= index) {
      exec_->batched_func_arg_mode.resize(index + 1);
    }
    auto modes = pair.second;
    std::vector<DBBatchedArgMode> arg_modes_vec;
    arg_modes_vec.reserve(modes.size());
    for (auto mode : modes) {
      arg_modes_vec.push_back(static_cast<DBBatchedArgMode>(mode->value));
    }
    exec_->batched_func_arg_mode[index] = std::move(arg_modes_vec);
  }

  VLOG(1) << "Compiled to:" << std::endl
          << "-------------------------------------------------" << std::endl
          << exec_->GetVirtualDevices()  //
          << exec_->GetConstants()       //
          << exec_->GetPrimitives()      //
          << exec_->GetBytecode()        //
          << "-------------------------------------------------";

  if (backend::IsAutoSchedulerEnabled()) {
    backend::UpdateAutoSchedulerOpWeights(context_.module);
  }

  if (PassContext::Current()->GetConfig<Bool>("relay.db_generate_aot_code", Bool(false)).value()) {
    auto output_directory = PassContext::Current()
                                ->GetConfig<String>("relay.db_aot_output_directory", String("./"))
                                .value();
    auto model_name = PassContext::Current()
                          ->GetConfig<String>("relay.db_model_name", String("model_name"))
                          .value();
    VMAOTCompiler(*exec_, context_.module, register_types, invoke_type_vars, compiled_functions,
                  get_field_tags, call_attrs, if_offsets, output_directory, model_name)
        .Codegen();
  }
}

transform::Sequential VMCompiler::MemoryOpt(const SEScope& host_se_scope) {
  bool batched_execution =
      PassContext::Current()->GetConfig<Bool>("relay.db_batched_execution", Bool(false)).value();

  Array<Pass> pass_seqs;
  // Remove unused functions
  Array<runtime::String> entry_functions{"main"};
  pass_seqs.push_back(transform::RemoveUnusedFunctions(entry_functions, batched_execution));
  // Manifest the allocations.
  pass_seqs.push_back(transform::ManifestAlloc(host_se_scope));

  // Compute away possibly introduced constant computation.
  pass_seqs.push_back(transform::FoldConstant());

  // Fuse & lower any new shape functions and device_copies.
  pass_seqs.push_back(FuseAndLowerOperators(host_se_scope));

  // Manifest the allocations needed for the shape functions.
  pass_seqs.push_back(transform::ManifestAlloc(host_se_scope));

  // Fuse & lower any new allocations.
  pass_seqs.push_back(FuseAndLowerOperators(host_se_scope));

  // Perform memory planning in order to coalesce/reduce allocations.

  pass_seqs.push_back(transform::PrintCurrentIR("FuseAndLowerOperators", false, true));
  pass_seqs.push_back(transform::CPPMemoryPlan());
  pass_seqs.push_back(transform::PrintCurrentIR("CPPMemoryPlan", true, true));

  // Compute away constant computation introduced by coalescing allocations.
  pass_seqs.push_back(transform::FoldConstant());

  // Fuse & lower yet again
  pass_seqs.push_back(FuseAndLowerOperators(host_se_scope));

  // Create allocations for math introduced by dynamic region math.
  pass_seqs.push_back(transform::ManifestAlloc(host_se_scope));

  // Compute away possibly introduced constant computation.
  pass_seqs.push_back(transform::FoldConstant());

  // Lift constants to the top-level of the block to simplify VM code generation.
  // TODO(@icemelon9, @jroesch): Remove this pass for now because some
  //  instructions need to access to constant
  // pass_seqs.push_back(transform::LiftConstants());

  return transform::Sequential(std::move(pass_seqs));
}

transform::Sequential VMCompiler::FuseAndLowerOperators(const SEScope& host_se_scope) {
  Array<Pass> pass_seqs;
  // Hoist operators to "primitive" Functions.
  pass_seqs.push_back(FuseOps());
  // Give each "primitive" Function a hash.
  pass_seqs.push_back(LabelOps());
  // Lower "primitive" Functions to PrimFuncs and rewrite calls.
  pass_seqs.push_back(tec::LowerTEPass(/*module_name=*/"vm_mod",
                                       [this](const BaseFunc& func) {
                                         if (func->GetAttr<String>(attr::kCompiler).defined()) {
                                           backend::UpdateConstants(func, &params_);
                                         }
                                       },
                                       host_se_scope));

  // Since lowered functions are bound in the IRModule, we can now eliminate any unused
  // let-bound functions.
  pass_seqs.push_back(DeadCodeElimination(/*inline_once=*/false));
  return transform::Sequential(std::move(pass_seqs));
}

IRModule VMCompiler::OptimizeModule(IRModule mod, const TargetMap& targets,
                                    const Target& target_host) {
  config_ = CompilationConfig(PassContext::Current(), targets, target_host);
  // The first device always corresponds to the host.
  CHECK(context_.se_scopes_.empty());
  context_.se_scopes_.push_back(config_->host_se_scope);
  // TODO(mbs): exec_ is not allocated. What is the API here?
  CHECK(exec_ == nullptr);
  return OptimizeModuleImpl(std::move(mod));
}

IRModule VMCompiler::OptimizeModuleImpl(IRModule mod) {
  VLOG_CONTEXT << "VM Optimize";
  if (params_.size()) {
    BaseFunc base_func = mod->Lookup("main");
    ICHECK(base_func->IsInstance<FunctionNode>())
        << "VM compiler expects to compile relay::Function";
    auto f = relay::backend::BindParamsByName(Downcast<Function>(base_func), params_);
    auto gvar = mod->GetGlobalVar("main");
    mod->Add(gvar, f);
  }

  transform::PassContext pass_ctx = PassContext::Current();
  bool batched_execution =
      pass_ctx->GetConfig<Bool>("relay.db_batched_execution", Bool(false)).value();
  bool scattered_kernels =
      pass_ctx->GetConfig<Bool>("relay.db_scattered_kernels", Bool(false)).value();

  Array<Pass> pass_seqs = relay::backend::GetPassPrefix(
      /*is_homogenous=*/config_->optional_homogeneous_target.defined(), /*is_vm=*/true);

  // pass_seqs.push_back(transform::PrintCurrentIR("Beginning", true, true));// Always plan devices
  // so the remaining passes don't need to distinguish homogeneous vs
  // hetrogeneous execution.
  pass_seqs.push_back(transform::PlanDevices(config_));

  pass_seqs.push_back(transform::InferType());
  pass_seqs.push_back(transform::FoldReduceSumsIdentifierPass());
  pass_seqs.push_back(transform::MarkScalarCalls());

  // pass_seqs.push_back(transform::PrintCurrentIR("MarkScalarCalls", true, true));
  pass_seqs.push_back(transform::FuseOps());
  // pass_seqs.push_back(transform::PrintCurrentIR("FuseOps", true, true));

  // Do layout rewrite for auto-scheduler.
  // pass_seqs.push_back(transform::PrintCurrentIR("FuseOps", true, false));
  // if (backend::IsAutoSchedulerEnabled() && config_->optional_homogeneous_target.defined()) {
  //   Pass major_pass = transform::AutoSchedulerLayoutRewrite(batched_execution,
  //   scattered_kernels); bool enable_layout_rewrite_targets =
  //       config_->optional_homogeneous_target->kind->device_type == kDLCPU ||
  //       config_->optional_homogeneous_target->GetAttr<String>("device", "") == "mali";
  //   if (enable_layout_rewrite_targets && pass_ctx.PassEnabled(major_pass->Info())) {
  //     With<Target> tctx(config_->optional_homogeneous_target);
  //     pass_seqs.push_back(major_pass);
  //     // Defuse ops to fold constants, then fuse them again
  //     pass_seqs.push_back(transform::DefuseOps());
  //     pass_seqs.push_back(transform::FoldConstant());
  //     pass_seqs.push_back(transform::FuseOps());
  //   }
  // }

  pass_seqs.push_back(transform::Inline());
  pass_seqs.push_back(transform::ToANormalForm());
  pass_seqs.push_back(transform::InferType());
  pass_seqs.push_back(transform::LambdaLift());

  // Eliminate dead-code before we lower. We don't track the purity of PrimFuncs, thus after
  // lowering all calls to lowered functions will be kept.
  pass_seqs.push_back(DeadCodeElimination(/*inline_once=*/false));
  pass_seqs.push_back(transform::LabelOps());

  // lower all functions annotated as "primitive" by FuseOps.
  pass_seqs.push_back(tec::LowerTEPass(/*module_name=*/"vm_mod",
                                       [this](const BaseFunc& func) {
                                         if (func->GetAttr<String>(attr::kCompiler).defined()) {
                                           backend::UpdateConstants(func, &params_);
                                         }
                                       },
                                       config_->host_se_scope));

  // Since lowered functions are bound in the IRModule, we can now eliminate any unused
  // let-bound functions.
  pass_seqs.push_back(DeadCodeElimination(/*inline_once=*/false));

  // Now that we have PrimFuncs, flow and solve SEScope constraints again to account for
  // any memory scopes which lowering has settled on.
  pass_seqs.push_back(transform::PlanDevices(config_));

  // Inline the functions that are lifted to the module scope. We perform this
  // pass after all other optimization passes but before the memory allocation
  // pass. This is because memory allocation pass will insert `invoke_tvm_op`
  // and we use these ops to invoke the symbols in the module generated by
  // external codegen.

  pass_seqs.push_back(MemoryOpt(config_->host_se_scope));

  if (pass_ctx->GetConfig<Bool>("relay.db_use_depth_tracking", Bool(false)).value()) {
    pass_seqs.push_back(transform::InferType());
    pass_seqs.push_back(transform::HoistNonSequentialOps());
  }

  if (pass_ctx->GetConfig<Bool>("relay.db_coarsen_granularity", Bool(false)).value()) {
    pass_seqs.push_back(transform::InferType());
    // pass_seqs.push_back(transform::PrintCurrentIR("Before coarsen", true, true));
    pass_seqs.push_back(
        transform::CoarsenPrimitiveFuncGranularity(batched_execution, scattered_kernels));
    pass_seqs.push_back(transform::InferType());
  } else {
    // Compute prim func access modes for all prim funcs
    pass_seqs.push_back(transform::ComputePrimFuncAccessModes());
  }

  if (true) {
    // pass_seqs.push_back(transform::InferType());
    // pass_seqs.push_back(transform::TensorDependentControlIdentifierPass());
  }

  pass_seqs.push_back(transform::PrintCurrentIR("Coarsen", true, false));
  transform::Sequential seq(pass_seqs);
  tvm::With<relay::transform::PassContext> ctx(pass_ctx);
  if (config_->optional_homogeneous_target.defined()) {
    With<Target> tctx(config_->optional_homogeneous_target);
    return seq(std::move(mod));
  } else {
    return seq(std::move(mod));
  }
}

size_t VMCompiler::PopulateGlobalMap() {
  // Allocate a VMFunction index for every Relay Function we could call.
  // Excludes PrimFuncs and externs, which are managed by the primitive_map_.
  for (const auto& kv : context_.module->functions) {
    if (const auto* function_node = kv.second.as<FunctionNode>()) {
      if (!function_node->GetAttr<String>(attr::kExternalSymbol)) {
        context_.global_map.emplace(kv.first, context_.global_map.size());
      }
    }
  }
  return context_.global_map.size();
}

void VMCompiler::Codegen() {
  VLOG_CONTEXT << "VM Codegen";
  if (!context_.module.defined()) {
    LOG(WARNING) << "No compiled module to codegen from. Did you forget to call VMCompiler::Lower?";
    return;
  }

  // At this point context_.module will contain only:
  //  - non-external Relay functions, which we've compiled into VMFunctions.
  //  - external Relay functions, which will have definitions within some external runtime module
  //    in the "external_mods" attribute
  //  - PrimFuncs annotated with their targets.
  // Only the PrimFuncs will appear in per_target_modules, and there may legitimately be none.
  Map<Target, IRModule> per_tvm_target_modules = tec::GetPerTargetModules(context_.module, true);
  for (const auto& kv : per_tvm_target_modules) {
    ICHECK(kv.first->kind->device_type != kDLExtDev);
  }
  Array<runtime::Module> ext_mods =
      context_.module->GetAttr<Array<runtime::Module>>("external_mods", Array<runtime::Module>())
          .value();
  VLOG(0) << "have " << per_tvm_target_modules.size() << " targets to build and " << ext_mods.size()
          << " external runtime modules";

  runtime::Module lib;
  if (per_tvm_target_modules.empty()) {
    // There is no function handled by TVM. We create a virtual main module
    // to make sure a DSO module will be also available.
    LOG(INFO) << "All lowered functions have been build by BYOC -- generating an empty TVM module";
    lib = codegen::CSourceModuleCreate(";", "", Array<String>{});
  } else {
    lib = tvm::build(per_tvm_target_modules, config_->host_target, true);
  }

  lib = codegen::CreateMetadataModule(params_, lib, ext_mods, config_->host_target,
                                      Runtime::Create("cpp"), runtime::Metadata());
  exec_->SetLib(lib);

  if (PassContext::Current()->GetConfig<Bool>("relay.db_generate_aot_code", Bool(false)).value()) {
    auto output_directory = PassContext::Current()
                                ->GetConfig<String>("relay.db_aot_output_directory", String("./"))
                                .value();
    auto model_name = PassContext::Current()
                          ->GetConfig<String>("relay.db_model_name", String("model_name"))
                          .value();

    std::string exe_file_name = model_name + ".ro";
    exec_->SaveToFileByteArray(output_directory + "/" + exe_file_name, "ro");
    std::string lib_file_name = model_name + "_lib.so";
    // lib->SaveToFile(output_directory + "/" + lib_file_name, "o");
    std::cout << "[AOT]  " << output_directory + "/" + exe_file_name << std::endl;
    std::cout << "[AOT]  " << output_directory + "/" + lib_file_name << std::endl;

    const auto* fexport_lib = runtime::Registry::Get("relay.db.llvm_module.export_lib");
    ICHECK(fexport_lib != nullptr) << "relay.db.llvm_module.export_lib";
    (*fexport_lib)(lib, output_directory + "/" + lib_file_name);
  }
}

runtime::Module CreateVMCompiler() {
  auto exec = make_object<VMCompiler>();
  return runtime::Module(exec);
}

TVM_REGISTER_GLOBAL("relay._vm._VMCompiler").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = CreateVMCompiler();
});

}  // namespace vm
}  // namespace relay
}  // namespace tvm
