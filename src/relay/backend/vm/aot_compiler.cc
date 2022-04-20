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
 * \file src/relay/backend/vm/compiler.h
 * \brief A compiler from relay::Module to the VM byte code.
 */

#include "aot_compiler.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../../support/utils.h"
#include "../utils.h"

namespace tvm {
namespace relay {
namespace vm {

using namespace tvm::runtime;
using namespace tvm::runtime::vm;
using namespace relay::transform;

std::string ToLowerCase(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) -> unsigned char { return std::tolower(c); });
  return s;
}
std::string ToUpperCase(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) -> unsigned char { return std::toupper(c); });
  return s;
}

std::string DTypeToStr(const DLDataType& dtype) {
  return "{" + std::to_string(dtype.code) + ", " + std::to_string(dtype.bits) + ", " +
         std::to_string(dtype.lanes) + "}";
}

std::string DTypeToTypeStr(const DataType& dtype) {
  auto bits = dtype.bits();
  auto code = dtype.code();
  if (code == kDLInt || code == kDLUInt) {
    if (code == kDLUInt && bits == 1) {
      return "bool";
    }
    std::stringstream os;
    if (code == kDLUInt) {
      os << "u";
    }
    os << "int";
    switch (bits) {
      case 8:
      case 16:
      case 32:
      case 64:
        os << bits << "_t";
        break;
      default:
        std::cout << "[SFF] No type for " << dtype << std::endl;
        return "";
    }
    return os.str();
  } else if (code == kDLFloat) {
    switch (bits) {
      case 32:
        return "float";
      case 64:
        return "double";
      default:
        std::cout << "[SFF] No type for " << dtype << std::endl;
        return "";
    }
  }
  return "";
}

bool lazy_execution() {
  static bool lazy_execution_ = tvm::transform::PassContext::Current()
                                    ->GetConfig<Bool>("relay.db_lazy_execution", Bool(false))
                                    .value();
  return lazy_execution_;
}

bool use_depth_tracking_executor() {
  static bool use_depth_tracking_executor_ =
      tvm::transform::PassContext::Current()
          ->GetConfig<Bool>("relay.db_use_depth_tracking", Bool(false))
          .value();
  return use_depth_tracking_executor_;
}

inline std::string GetTensorType() {
  if (lazy_execution()) {
    return "DLTensor*";
  } else {
    return "NDArray";
  }
}

void RelayTypeToCppStr(std::ostream& os, const Type& type, bool no_shared_ptr = false,
                       const std::string& replacement = "", bool scalarize = true) {
  if (auto tn = type.as<TensorTypeNode>()) {
    if (scalarize && tn->shape.size() == 0) {
      auto dtype_str = DTypeToTypeStr(tn->dtype);
      if (dtype_str.size() > 0) {
        os << dtype_str;
        return;
      }
    }
    os << GetTensorType();
  } else if (auto pt = type.as<PrimTypeNode>()) {
    os << pt->dtype << "_t";
  } else if (auto td = type.as<TypeDataNode>()) {
    if (td->header->name_hint == "Storage") {
      os << "Storage";
    } else {
      auto type_name = td->header->name_hint;
      if (replacement.size() > 0) {
        type_name = replacement;
      }
      if (!no_shared_ptr) {
        os << "std::shared_ptr<";
      }
      os << type_name;
      if (!no_shared_ptr) {
        os << ">";
        // os << "*";
      }
    }
  } else if (auto ft = type.as<FuncTypeNode>()) {
    ICHECK_EQ(ft->type_params.size(), 0);
    os << "std::function<";
    RelayTypeToCppStr(os, ft->ret_type);
    os << "(";
    for (size_t i = 0; i < ft->arg_types.size(); ++i) {
      RelayTypeToCppStr(os, ft->arg_types[i]);
      if (i < ft->arg_types.size() - 1) {
        os << ",";
      }
    }
    if (use_depth_tracking_executor()) {
      os << ", int&";
    }
    os << ")>";
  } else if (auto tv = type.as<TypeVarNode>()) {
    os << tv->name_hint;
  } else if (auto tt = type.as<TupleTypeNode>()) {
    if (!no_shared_ptr) {
      os << "std::shared_ptr<";
    }
    os << "std::tuple<";
    for (size_t i = 0; i < tt->fields.size(); ++i) {
      RelayTypeToCppStr(os, tt->fields[i]);
      if (i < tt->fields.size() - 1) {
        os << ",";
      }
    }
    os << ">";
    if (!no_shared_ptr) {
      os << ">";
      // os << "*";
    }
  } else if (auto tc = type.as<TypeCallNode>()) {
    auto type_func_gv = tc->func.as<GlobalTypeVarNode>();
    ICHECK(type_func_gv) << type;

    if (type_func_gv->name_hint != "Storage" && !no_shared_ptr) {
      os << "std::shared_ptr<";
    }
    if (replacement.size() > 0) {
      os << replacement;
    } else {
      os << type_func_gv->name_hint;
    }
    if (tc->args.size() > 0) {
      os << "<";
      for (size_t i = 0; i < tc->args.size(); ++i) {
        RelayTypeToCppStr(os, tc->args[i]);
        if (i < tc->args.size() - 1) {
          os << ",";
        }
      }
      os << ">";
    }
    if (type_func_gv->name_hint != "Storage" && !no_shared_ptr) {
      os << ">";
      // os << "*";
    }
  } else {
    std::cout << "DEFAULT " << type << std::endl;
    os << "DEFAULT";
  }
}

std::string RelayTypeToCppStrString(const Type& type, bool no_shared_ptr = false,
                                    const std::string& replacement = "", bool scalarize = true) {
  std::stringstream ss;
  RelayTypeToCppStr(ss, type, no_shared_ptr, replacement, scalarize);
  return ss.str();
}

std::string GetModelMainFunctionName() { return "model_main"; }

std::string GetDepthTrackingMapFunctionName() { return "pmap"; }

std::string GetCppFunctionName(const std::string& name) {
  if (name == "main") {
    return GetModelMainFunctionName();
  } else if (use_depth_tracking_executor() && name == "map") {
    return GetDepthTrackingMapFunctionName();
  }
  return name;
}

std::string GetExecutorType() {
  if (use_depth_tracking_executor()) {
    return "DepthTrackingExecutor";
  } else {
    return "LazyExecutor<" + GetTensorType() + ">";
  }
}

inline std::string GetRuntimeType() {
  return "DynBatchRuntime<" + GetExecutorType() + ", " + GetTensorType() + ">";
}

Function GetCompiledRelayFunction(
    const std::unordered_map<std::string, Function>& compiled_functions, const IRModule& mod,
    const std::string& name) {
  auto it = compiled_functions.find(name);
  if (it != compiled_functions.end()) {
    return it->second;
  } else {
    return Downcast<Function>(mod->Lookup(name));
  }
}

template <typename T>
void PrintArray(const T& e) {
  std::cout << e << ' ';
}

template <typename T, std::size_t N>
void PrintArray(const std::array<T, N>& A) {
  for (const auto& e : A) PrintArray(e);
}

class VMAOTFunctionCompiler : SourcePrinter {
 public:
  VMAOTFunctionCompiler(const Executable& exec, const IRModule& mod, const VMFunction& vm_func,
                        const Function& relay_func,
                        const std::unordered_map<size_t, Type>& register_types,
                        const std::unordered_map<Index, Array<Type>>& invoke_type_vars,
                        const std::unordered_map<std::string, Function>& compiled_functions,
                        const std::unordered_map<Index, int32_t>& get_field_tags,
                        const std::unordered_map<Index, DictAttrs>& call_attrs,
                        const std::unordered_map<Index, std::array<Index, 4>>& if_offsets,
                        std::unordered_set<const TensorTypeNode*>* p_invoke_reduce_sum_types,
                        std::ostream& stream)
      : exec_(exec),
        mod_(mod),
        vm_func_(vm_func),
        relay_func_(relay_func),
        register_types_(register_types),
        invoke_type_vars_(invoke_type_vars),
        compiled_functions_(compiled_functions),
        get_field_tags_(get_field_tags),
        call_attrs_(call_attrs),
        if_offsets_(if_offsets),
        p_invoke_reduce_sum_types_(p_invoke_reduce_sum_types),
        stream_(stream) {
    inbuilt_ops_.insert({"subtract", "-"});
    inbuilt_ops_.insert({"add", "+"});
    inbuilt_ops_.insert({"equal", "=="});
    inbuilt_ops_.insert({"greater_equal", ">="});
  }

  int GenerateCPPForFunction(bool definition) {
    int max_static_depth = -1;
    if (use_depth_tracking_executor() && vm_func_.name == "map") {
      EmitPMapCPP(definition);
    } else {
      // std::cout << "[FUN] Visiting " << vm_func_.name << std::endl;
      Type function_type = relay_func_->checked_type_;
      CreateFunctionDeclaration(vm_func_, relay_func_);

      if (definition) {
        stream_ << " {\n";
        this->BeginScope();

        max_static_depth = this->VisitBytecode();

        this->EndScope();
        stream_ << "}\n";
      } else {
        stream_ << ";\n";
      }
    }
    return max_static_depth;
  }

 private:
  int32_t GetCallGraphDepth(int pc) {
    auto it = call_attrs_.find(pc);
    if (it != call_attrs_.end()) {
      auto attrs = it->second;
      return attrs.GetAttr(tir::attr::kDBGraphDepth, Integer(-1)).value()->value;
    }
    return -1;
  }

  const CallNode* GetScalarOp(int pc) {
    auto it = call_attrs_.find(pc);
    if (it != call_attrs_.end()) {
      auto attrs = it->second;
      auto opt_op = attrs.GetAttr(tir::attr::kDBScalarCall, NullValue<Expr>());
      if (opt_op) {
        return opt_op.value().as<CallNode>();
      }
    }
    return nullptr;
  }

  bool IsOpOutputScalar(int pc) {
    auto it = call_attrs_.find(pc);
    if (it != call_attrs_.end()) {
      auto attrs = it->second;
      return attrs.GetAttr(tir::attr::kDBScalarOutputOp, Bool(false)).value()->value;
    }
    return false;
  }

  bool DoIncrementDepth(int pc) {
    auto it = call_attrs_.find(pc);
    if (it != call_attrs_.end()) {
      auto attrs = it->second;
      return attrs.GetAttr(tir::attr::kDBIncrementDepth, Bool(true)).value()->value;
    }
    return false;
  }

  Op GetFoldReductionOp(int pc) {
    auto it = call_attrs_.find(pc);
    if (it != call_attrs_.end()) {
      return it->second.GetAttr(tir::attr::kDBFoldReduction, NullValue<Op>()).value();
    }
    return NullValue<Op>();
  }

  void CreateFunctionDeclaration(const VMFunction& vm_func, const Function& relay_func) {
    FuncType function_type = Downcast<FuncType>(relay_func->checked_type_);
    ICHECK_EQ(vm_func.params.size(), function_type->arg_types.size())
        << vm_func.params.size() << " " << function_type->arg_types.size() << " "
        << relay_func->params.size();

    if (relay_func->type_params.size() > 0) {
      stream_ << "template<";
      for (size_t i = 0; i < function_type->type_params.size(); ++i) {
        auto tvar = function_type->type_params[i];
        stream_ << "class " << tvar->name_hint;
        if (i != function_type->type_params.size() - 1) {
          stream_ << ", ";
        }
      }
      stream_ << ">\n";
    }
    RelayTypeToCppStr(stream_, function_type->ret_type);
    stream_ << " " << GetCppFunctionName(vm_func.name) << "(";

    for (size_t i = 0; i < function_type->arg_types.size(); ++i) {
      auto arg_type = function_type->arg_types[i];
      RelayTypeToCppStr(stream_, arg_type);
      if (arg_type.as<TensorTypeNode>() && !lazy_execution()) {
        stream_ << "&";
      }
      stream_ << " " << GetVarForReg(i);
      if (i != function_type->arg_types.size() - 1) {
        stream_ << ", ";
      }
    }
    if (use_depth_tracking_executor()) {
      stream_ << ", int& depth";
    }
    stream_ << ")";
  }

  bool IsStorageType(const Type& type) {
    if (auto tc = type.as<TypeCallNode>()) {
      return IsStorageType(tc->func);
    } else if (auto td = type.as<TypeDataNode>()) {
      return td->header->name_hint == "Storage";
    } else {
      return false;
    }
  }

  bool IsTupleMapFunction(const std::string& name) {
    return use_depth_tracking_executor() && support::StartsWith(name, "tuple_map_");
  }

  void GenerateLocalDecls(const std::vector<bool>& used_regs,
                          const std::unordered_set<Index>& scalar_op_outputs) {
    std::unordered_map<std::string, std::vector<std::string>> type2vars;
    for (int i = vm_func_.params.size(); i < vm_func_.register_file_size; ++i) {
      if (!used_regs[i]) {
        continue;
      }
      auto it = register_types_.find(i);
      ICHECK(it != register_types_.end()) << i;
      Type reg_type = it->second;
      if (scalar_op_outputs.count(i)) {
        ICHECK(reg_type.as<TensorTypeNode>());
        auto tensor_type = RelayTypeToCppStrString(reg_type, false, "", false);
        auto scalar_type = RelayTypeToCppStrString(reg_type, false, "", true);
        auto tensor_var = GetVarForReg(i);
        auto scalar_var = GetVarForReg(i, true);
        type2vars[tensor_type].push_back(tensor_var);
        type2vars[scalar_type].push_back(scalar_var);
      } else {
        if (lazy_execution() && IsStorageType(reg_type)) {
          continue;
        }
        if (reg_type.as<TensorTypeNode>() || reg_type.as<PrimTypeNode>() ||
            IsStorageType(reg_type)) {
          std::stringstream ss;
          RelayTypeToCppStr(ss, reg_type);
          auto type_str = ss.str();
          type2vars[type_str].push_back(GetVarForReg(i));
        } else {
          auto type_str = RelayTypeToCppStrString(reg_type);
          this->PrintIndent(stream_);
          stream_ << type_str << " " << GetVarForReg(i) << ";\n";
        }
      }
    }

    for (auto kv : type2vars) {
      this->PrintIndent(stream_);
      stream_ << kv.first << " ";
      auto vars = kv.second;
      for (size_t i = 0; i < vars.size(); ++i) {
        auto var_str = vars[i];
        if (kv.first == "DLTensor*" && i > 0) {
          var_str = "*" + var_str;
        }
        stream_ << var_str;
        if (i < vars.size() - 1) {
          stream_ << ", ";
        }
      }
      stream_ << ";\n";
    }
    stream_ << "\n";
  }

  int VisitBytecode() {
    std::cout << "\n[BT]  Visiting code " << vm_func_.name << std::endl;
    bool print = (vm_func_.name == "main");
    std::vector<bool> used_regs(vm_func_.register_file_size, false);
    for (size_t i = 0; i < vm_func_.instructions.size(); ++i) {
      auto& instr = vm_func_.instructions[i];
      if (lazy_execution() && instr.op == Opcode::AllocStorage) {
        continue;
      }
      if (print && instr.op == Opcode::Ret) {
        std::cout << "[BT]  Read registers instruction " << instr << std::endl;
        std::cout << "[BT]    " << support::PrintVector(Instruction::ReadRegisters(instr))
                  << std::endl;
      }
      for (auto reg : Instruction::ReadRegisters(instr)) {
        used_regs[reg] = true;
      }
    }

    // Registers that are scalar outputs of tensor ops
    std::unordered_set<Index> scalar_outs_of_tensor_ops;
    std::unordered_set<Index> scalar_outs_of_scalar_ops;
    for (size_t i = 0; i < vm_func_.instructions.size(); ++i) {
      auto& instr = vm_func_.instructions[i];

      if (instr.op == Opcode::InvokePacked && IsOpOutputScalar(i)) {
        if (GetScalarOp(i)) {
          ICHECK(IsScalarTensorType(register_types_.at(instr.packed_args[instr.arity - 1])));
          scalar_outs_of_scalar_ops.insert(instr.packed_args[instr.arity - 1]);

        } else {
          for (int j = 0; j < instr.arity; ++j) {
            if (IsScalarTensorType(register_types_.at(instr.packed_args[j]))) {
              scalar_outs_of_tensor_ops.insert(instr.packed_args[j]);
            }
          }
        }
      }
    }

    this->GenerateLocalDecls(used_regs, scalar_outs_of_tensor_ops);

    // std::cout << "[BT] Visiting BT" << std::endl;
    std::unordered_map<Index, std::string> targets;
    int target_count = 0;

    // std::cout << "[BT]  Visiting targets" << std::endl;
    std::unordered_set<Index> ignorable_gotos;
    for (size_t i = 0; i < vm_func_.instructions.size(); ++i) {
      auto& instr = vm_func_.instructions[i];

      // std::cout << "[BT]   " << i << ": " << instr;
      // if (instr.op == Opcode::If) {
      //   std::cout << "|";
      //   PrintArray(if_offsets_.at(i));
      // }
      // std::cout << std::endl;

      if (instr.op == Opcode::Goto && !ignorable_gotos.count(i)) {
        targets[instr.pc_offset + i] = "target" + std::to_string(target_count++);
      } else if (instr.op == Opcode::If) {
        auto it = if_offsets_.find(i);
        if (it != if_offsets_.end()) {
          auto true_end_offset = it->second[1];
          auto goto_pc = i + true_end_offset;
          ignorable_gotos.insert(goto_pc);
        } else {
          targets[instr.if_op.true_offset + i] = "target" + std::to_string(target_count++);
          targets[instr.if_op.false_offset + i] = "target" + std::to_string(target_count++);
        }
      }
    }

    for (size_t i = 0; i < vm_func_.instructions.size(); ++i) {
      auto& instr = vm_func_.instructions[i];
      if (instr.op == Opcode::InvokePacked) {
        auto callee_idx = instr.packed_index;
      }
    }

    std::unordered_map<RegName, Index> storage_device_indices;
    int tmp_var_counter = 0;

    bool same_depth_for_all_calls = IsTupleMapFunction(vm_func_.name);
    int max_static_depth = -1;
    std::vector<std::string> all_depth_vars;
    std::function<void(int, int)> generate_code = [&](int start_pc, int end_pc) {
      for (int i = start_pc; i < end_pc; ++i) {
        auto& instr = vm_func_.instructions[i];
        if (print && instr.op == Opcode::Ret) {
          std::cout << "[BT] MAIN RET" << std::endl;
        }

        std::cout << "[BT]     " << instr << std::endl;

        if (Instruction::UsesDst(instr) && !used_regs[instr.dst]) {
          continue;
        }

        auto it = targets.find(i);
        if (it != targets.end()) {
          this->PrintIndent(stream_, -2);
          stream_ << it->second << ":\n";
        }

        switch (instr.op) {
          case Opcode::Move: {
            auto src_var = GetVarForReg(instr.from);
            auto dst_var = GetVarForReg(instr.dst);
            this->PrintIndent(stream_);
            stream_ << dst_var << " = " << src_var << ";\n";
            break;
          }
          case Opcode::Fatal: {
            this->PrintIndent(stream_);
            stream_ << "throw std::runtime_error(\"Fatal error\");\n";
            break;
          }
          case Opcode::LoadConst: {
            auto dst_var = GetVarForReg(instr.dst);
            ICHECK(register_types_.at(instr.dst).as<TensorTypeNode>());
            this->PrintIndent(stream_);
            stream_ << dst_var << " = const_cast<DLTensor*>(" << GetRuntimeType()
                    << "::Current()->GetConstant(" << instr.const_index << ").operator->());\n";
            break;
          }
          case Opcode::LoadConsti: {
            auto dst_var = GetVarForReg(instr.dst);
            this->PrintIndent(stream_);
            stream_ << dst_var << " = " << instr.load_consti.val << ";\n";
            break;
          }
          case Opcode::Invoke: {
            if (instr.packed_index == DB_RANDOM_UNIFORM_INDEX) {
              this->PrintIndent(stream_);
              auto dst_var = GetVarForReg(instr.dst);
              auto lo_var = GetVarForReg(instr.invoke_args_registers[0]);
              auto hi_var = GetVarForReg(instr.invoke_args_registers[1]);
              stream_ << dst_var << " = GetRandom(" << lo_var << ", " << hi_var << ");\n";
            } else {
              auto fold_reduction_op = GetFoldReductionOp(i);
              auto dst_type = register_types_.at(instr.dst).as<TensorTypeNode>();
              std::vector<int64_t> int_shape;
              if (dst_type) {
                int_shape = backend::GetIntShape(dst_type->shape);
              }

              std::string depth_var;
              if (same_depth_for_all_calls) {
                this->PrintIndent(stream_);
                depth_var = "__depth" + std::to_string(tmp_var_counter++);
                stream_ << "int " << depth_var << " = __orig_depth;\n";
                all_depth_vars.push_back(depth_var);
              } else if (use_depth_tracking_executor()) {
                depth_var = "depth";
              }

              if (fold_reduction_op.defined() && fold_reduction_op == GetAddOp() &&
                  (dst_type && int_shape.size() == dst_type->shape.size())) {
                ICHECK_EQ(instr.num_args, 3);
                auto dst_var = GetVarForReg(instr.dst);
                auto list_var = GetVarForReg(instr.invoke_args_registers[2]);
                auto init_var = GetVarForReg(instr.invoke_args_registers[1]);

                this->PrintIndent(stream_);
                stream_ << dst_var << " = invoke_reduce_sum(append(" << list_var << ", " << init_var
                        << ")";
                if (use_depth_tracking_executor()) {
                  stream_ << ", " << depth_var;
                }
                stream_ << ");\n";

                p_invoke_reduce_sum_types_->insert(dst_type);
              } else {
                auto dst_var = GetVarForReg(instr.dst);
                auto callee_name = exec_.functions[instr.func_index].name;
                this->PrintIndent(stream_);
                stream_ << dst_var << " = " << GetCppFunctionName(callee_name);
                auto it = invoke_type_vars_.find(i);
                if (it != invoke_type_vars_.end() && it->second.size() > 0) {
                  auto types = it->second;
                  stream_ << "<";
                  for (size_t j = 0; j < types.size(); ++j) {
                    RelayTypeToCppStr(stream_, types[j]);
                    if (j < types.size() - 1) {
                      stream_ << ",";
                    }
                  }
                  stream_ << ">";
                }
                stream_ << "(";
                for (int i = 0; i < instr.num_args; ++i) {
                  stream_ << GetVarForReg(instr.invoke_args_registers[i]);
                  if (i < instr.num_args - 1) {
                    stream_ << ",";
                  }
                }
                if (use_depth_tracking_executor()) {
                  stream_ << ", " << depth_var;
                }
                stream_ << ");\n";
              }
            }
            break;
          }
          case Opcode::InvokePacked: {
            auto get_scalar_var_for_reg = [&](Index reg) {
              if (scalar_outs_of_tensor_ops.count(reg)) {
                return GetVarForReg(reg, true);
              } else {
                return GetVarForReg(reg);
              }
            };

            std::vector<std::string> flattened_args;
            for (int j = 0; j < instr.arity; ++j) {
              if (auto tt = register_types_.at(instr.packed_args[j]).as<TupleTypeNode>()) {
                auto tuple_var = GetVarForReg(instr.packed_args[j]);
                for (size_t k = 0; k < tt->fields.size(); ++k) {
                  flattened_args.push_back("std::get<" + std::to_string(k) + ">(*" + tuple_var +
                                           ")");
                }
              } else {
                flattened_args.push_back(get_scalar_var_for_reg(instr.packed_args[j]));
              }
            }

            auto scalar_op_call = GetScalarOp(i);
            auto scalar_op = scalar_op_call ? scalar_op_call->op.as<OpNode>() : nullptr;
            if (scalar_op_call && inbuilt_ops_.count(scalar_op->name)) {
              ICHECK(scalar_op);
              auto op_str = inbuilt_ops_.at(scalar_op->name);
              auto dst_var = flattened_args.back();
              this->PrintIndent(stream_);
              if (flattened_args.size() == 3) {
                auto op1 = flattened_args[0];
                auto op2 = flattened_args[1];
                stream_ << dst_var << " = (" << op1 << " " << op_str << " " << op2 << ");\n";
              } else {
                ICHECK_EQ(flattened_args.size(), 2);
                ICHECK_EQ(scalar_op_call->args.size(), 2);
                if (auto cn = scalar_op_call->args[0].as<ConstantNode>()) {
                  auto op1 = backend::NDToInt(cn->data);
                  auto op2 = flattened_args[0];
                  stream_ << dst_var << " = (" << op1 << " " << op_str << " " << op2 << ");\n";
                } else if (auto cn = scalar_op_call->args[1].as<ConstantNode>()) {
                  auto op1 = flattened_args[0];
                  auto op2 = backend::NDToInt(cn->data);
                  stream_ << dst_var << " = (" << op1 << " " << op_str << " " << op2 << ");\n";
                }
              }
            } else {
              auto args_vec = "args_tmp" + std::to_string(tmp_var_counter++);
              this->PrintIndent(stream_);
              stream_ << "std::vector<" << GetTensorType() << "> " << args_vec << " = {";

              for (size_t j = 0; j < flattened_args.size(); ++j) {
                stream_ << flattened_args[j];
                if (j < flattened_args.size() - 1) {
                  stream_ << ", ";
                }
              }

              stream_ << "};\n";
              this->PrintIndent(stream_);
              if (use_depth_tracking_executor()) {
                std::string depth_str = ", ";
                auto depth = GetCallGraphDepth(i);
                if (depth >= 0) {
                  max_static_depth = std::max(depth, max_static_depth);
                  depth_str += std::to_string(depth);
                } else if (DoIncrementDepth(i)) {
                  depth_str += "depth++";
                } else {
                  depth_str += "depth";
                }
                stream_ << GetRuntimeType() << "::Current()->InvokePackedWithDepth("
                        << instr.packed_index << depth_str << ", " << args_vec << ".data(), "
                        << flattened_args.size() << ");\n";
              } else {
                if (lazy_execution()) {
                  stream_ << GetRuntimeType() << "::Current()->InvokePacked(" << instr.packed_index
                          << ", " << args_vec << ".data(), " << flattened_args.size() << ");\n";
                } else {
                  stream_ << GetRuntimeType() << "::Current()->InvokePacked(" << instr.packed_index
                          << ", " << args_vec << ".data(), " << flattened_args.size() << ");\n";
                }
              }

              if (IsOpOutputScalar(i) && GetScalarOp(i) == nullptr) {
                for (int j = 0; j < instr.arity; ++j) {
                  if (IsScalarTensorType(register_types_.at(instr.packed_args[j]))) {
                    this->PrintIndent(stream_);
                    stream_ << GetVarForReg(instr.packed_args[j], true) << " = Scalarize("
                            << GetVarForReg(instr.packed_args[j]) << ");\n";
                  }
                }
              }
            }

            break;
          }
          case Opcode::InvokeClosure: {
            auto dst_var = GetVarForReg(instr.dst);
            auto callee_var = GetVarForReg(instr.closure);
            this->PrintIndent(stream_);
            stream_ << dst_var << " = " << callee_var;
            stream_ << "(";
            for (int i = 0; i < instr.num_closure_args; ++i) {
              stream_ << GetVarForReg(instr.closure_args[i]);
              if (i < instr.num_closure_args - 1) {
                stream_ << ", ";
              }
            }
            if (use_depth_tracking_executor()) {
              stream_ << ", depth";
            }
            stream_ << ");\n";
            break;
          }
          case Opcode::GetField: {
            auto object_var = GetVarForReg(instr.object);
            auto dst_var = GetVarForReg(instr.dst);

            this->PrintIndent(stream_);
            if (register_types_.at(instr.object).as<TupleTypeNode>()) {
              stream_ << dst_var << " = std::get<" << instr.field_index << ">(*" << object_var
                      << ");\n";
            } else {
              // auto tag = get_field_tags_.at(i);
              // auto constructor_name = ToLowerCase(mod_->LookupTag(tag)->name_hint);
              // stream_ << dst_var << " = " << object_var << "->" << constructor_name << "."
              //         << this->GetFieldName(instr.field_index) << ";\n";

              auto tag = get_field_tags_.at(i);
              auto constructor_name = mod_->LookupTag(tag)->name_hint;
              stream_ << dst_var << " = static_cast<";
              RelayTypeToCppStr(stream_, register_types_.at(instr.object), true, constructor_name);
              stream_ << "*>(" << object_var << ".get())->" << GetFieldName(instr.field_index)
                      << ";\n";
            }
            break;
          }
          case Opcode::GetTag: {
            auto object_var = GetVarForReg(instr.object);
            auto dst_var = GetVarForReg(instr.dst);
            this->PrintIndent(stream_);
            stream_ << dst_var << " = " << object_var << "->tag;\n";
            break;
          }
          case Opcode::Goto: {
            Index target = i + instr.pc_offset;
            if (!targets.count(target)) {
              for (auto kv : targets) {
              }
            }
            auto label = targets.at(target);
            this->PrintIndent(stream_);
            stream_ << "goto " << label << ";\n";
            break;
          }
          case Opcode::If: {
            auto it = if_offsets_.find(i);
            if (it != if_offsets_.end()) {
              auto offsets = it->second;
              auto true_start = offsets[0];
              auto true_end = offsets[1];
              auto false_start = offsets[2];
              auto false_end = offsets[3];

              auto test_var = GetVarForReg(instr.if_op.test);
              auto target_var = GetVarForReg(instr.if_op.target);

              this->PrintIndent(stream_);
              stream_ << "if (" << test_var << " == " << target_var << ") {\n";
              this->BeginScope();
              generate_code(i + true_start, i + true_end);
              this->EndScope();
              this->PrintIndent(stream_);
              stream_ << "} else {\n";
              this->BeginScope();
              generate_code(i + false_start, i + false_end);
              this->PrintIndent(stream_);
              this->EndScope();
              stream_ << "}\n";

              i = i + false_end - 1;
            } else {
              Index true_target = i + instr.if_op.true_offset;
              auto true_label = targets.at(true_target);

              Index false_target = i + instr.if_op.false_offset;
              auto false_label = targets.at(false_target);

              auto test_var = GetVarForReg(instr.if_op.test);
              auto target_var = GetVarForReg(instr.if_op.target);

              this->PrintIndent(stream_);
              stream_ << "if (" << test_var << " == " << target_var << ") {\n";
              this->BeginScope();
              this->PrintIndent(stream_);
              stream_ << "goto " << true_label << ";\n";
              this->EndScope();
              this->PrintIndent(stream_);
              stream_ << "} else {\n";
              this->BeginScope();
              this->PrintIndent(stream_);
              stream_ << "goto " << false_label << ";\n";
              this->EndScope();
              this->PrintIndent(stream_);
              stream_ << "}\n";
            }
            break;
          }
          case Opcode::AllocTensor: {
            if (scalar_outs_of_scalar_ops.count(instr.dst)) {
              break;
            }
            auto dst_var = GetVarForReg(instr.dst);
            auto storage_var = GetVarForReg(instr.alloc_tensor.storage);
            auto offset_var = GetVarForReg(instr.alloc_tensor.offset);
            std::string dtype_str = DTypeToStr(instr.alloc_tensor.dtype);

            std::string offset_var_str = offset_var;
            if (register_types_.at(instr.alloc_tensor.offset).as<TensorTypeNode>()) {
              this->PrintIndent(stream_);
              offset_var_str = "NDToInt64(" + offset_var_str + ")";
            }

            std::stringstream shape_arr;
            shape_arr << "{";
            for (size_t j = 0; j < instr.alloc_tensor.ndim; ++j) {
              shape_arr << instr.alloc_tensor.shape[j];
              if (j < instr.alloc_tensor.ndim - 1) {
                shape_arr << ", ";
              }
            }
            shape_arr << "}";

            if (lazy_execution()) {
              auto it = storage_device_indices.find(instr.alloc_tensor.storage);
              ICHECK(it != storage_device_indices.end());
              auto device_index = it->second;

              auto tmp_var = "shape_data" + std::to_string(tmp_var_counter++);

              this->PrintIndent(stream_);
              stream_ << "auto " << tmp_var << " = Arena::Current()->allocate_<int64_t>("
                      << instr.alloc_tensor.ndim << ");\n";
              this->PrintIndent(stream_);
              stream_ << tmp_var << " = new(" << tmp_var << ") int64_t[" << instr.alloc_tensor.ndim
                      << "]" << shape_arr.str() << ";\n";

              this->PrintIndent(stream_);
              stream_ << dst_var << " = " << GetRuntimeType() << "::Current()"
                      << "->AllocArrayWrapper(" << tmp_var << ", " << instr.alloc_tensor.ndim
                      << ", " << dtype_str << ", " << device_index << ");\n";
            } else {
              this->PrintIndent(stream_);
              stream_ << dst_var << " = " << storage_var << "->AllocNDArray(" << offset_var_str
                      << ", " << shape_arr.str() << ", " << dtype_str << ");\n";
            }
            break;
          }
          case Opcode::AllocTensorReg: {
            ICHECK(!lazy_execution());
            if (scalar_outs_of_scalar_ops.count(instr.dst)) {
              break;
            }
            auto dst_var = GetVarForReg(instr.dst);
            auto storage_var = GetVarForReg(instr.alloc_tensor_reg.storage);
            auto offset_var = GetVarForReg(instr.alloc_tensor.offset);
            std::string dtype_str = DTypeToStr(instr.alloc_tensor_reg.dtype);

            std::string shape_var = GetVarForReg(instr.alloc_tensor_reg.shape_register);
            this->PrintIndent(stream_);
            stream_ << dst_var << " = " << GetRuntimeType() << "::Current()"
                    << "->AllocateTensorReg(" << storage_var << ", " << offset_var << ", "
                    << shape_var << ", " << dtype_str << ");\n";
            break;
          }
          case Opcode::AllocADT: {
            auto dst_var = GetVarForReg(instr.dst);
            this->PrintIndent(stream_);
            if (instr.constructor_tag == 0) {
              std::stringstream types_str, args_str;
              for (int j = 0; j < instr.num_fields; ++j) {
                ICHECK(register_types_.count(instr.datatype_fields[j])) << instr.datatype_fields[j];
                auto field_type = register_types_.at(instr.datatype_fields[j]);
                auto field_var = GetVarForReg(instr.datatype_fields[j]);
                RelayTypeToCppStr(types_str, field_type);
                args_str << field_var;
                if (j < instr.num_fields - 1) {
                  args_str << ", ";
                  types_str << ", ";
                }
              }

              stream_ << dst_var << " = std::shared_ptr<std::tuple<" << types_str.str()
                      << ">>(new std::tuple<" << types_str.str() << ">(" << args_str.str()
                      << "));\n";
              // stream_ << dst_var << " = new std::tuple<" << types_str.str() << ">(" <<
              // args_str.str()
              // << ");\n";
            } else {
              auto constructor = mod_->LookupTag(instr.constructor_tag);
              auto concrete_type_without_shptr =
                  RelayTypeToCppStrString(register_types_.at(instr.dst), true);
              auto concrete_constructor_without_shptr = RelayTypeToCppStrString(
                  register_types_.at(instr.dst), true, constructor->name_hint);

              stream_ << dst_var << " = std::static_pointer_cast<" << concrete_type_without_shptr
                      << ">(std::make_shared<" << concrete_constructor_without_shptr << ">());\n";

              // stream_ << dst_var << " = new " << concrete_constructor_without_shptr << "();\n";

              this->PrintIndent(stream_);
              stream_ << dst_var << "->tag = " << instr.constructor_tag << ";\n";
              for (int j = 0; j < instr.num_fields; ++j) {
                ICHECK(register_types_.count(instr.datatype_fields[j])) << instr.datatype_fields[j];
                auto field_var = GetVarForReg(instr.datatype_fields[j]);
                this->PrintIndent(stream_);
                stream_ << "static_cast<" << concrete_constructor_without_shptr << "*>(" << dst_var
                        << ".get())->" << GetFieldName(j) << " = " << field_var << ";\n";
              }
            }
            break;
          }
          case Opcode::AllocClosure: {
            // std::cout << "[Closure] " << exec_.functions[instr.clo_index].name << " "
            // << exec_.functions[instr.clo_index].params.size() << " " << instr.num_freevar
            // << std::endl;
            auto& closure_func = exec_.functions[instr.clo_index];
            auto closure_relay_func =
                GetCompiledRelayFunction(compiled_functions_, mod_, closure_func.name);
            auto dst_var = GetVarForReg(instr.dst);
            this->PrintIndent(stream_);
            stream_ << dst_var << " = [";
            for (int i = 0; i < instr.num_freevar; ++i) {
              stream_ << "&" << GetVarForReg(instr.invoke_args_registers[i]);
              if (i < instr.num_freevar - 1) {
                stream_ << ",";
              }
            }
            stream_ << "](";
            int closure_params_size = static_cast<int>(closure_func.params.size());
            for (int j = instr.num_freevar; j < closure_params_size; ++j) {
              Type arg_type = closure_relay_func->params[j]->checked_type_;
              RelayTypeToCppStr(stream_, arg_type);
              stream_ << " " << GetTmpVarName(j);
              if (j < closure_params_size - 1) {
                stream_ << ", ";
              }
            }
            if (use_depth_tracking_executor()) {
              stream_ << ", int& depth";
            }
            stream_ << ") {\n";
            this->BeginScope();
            this->PrintIndent(stream_);
            stream_ << "return " << GetCppFunctionName(closure_func.name) << "(";
            int j = 0;
            for (; j < instr.num_freevar; ++j) {
              stream_ << GetVarForReg(instr.invoke_args_registers[j]);
              if (j < closure_params_size - 1) {
                stream_ << ", ";
              }
            }
            for (; j < closure_params_size; ++j) {
              stream_ << GetTmpVarName(j);
              if (j < closure_params_size - 1) {
                stream_ << ", ";
              }
            }
            if (use_depth_tracking_executor()) {
              stream_ << ", depth";
            }
            stream_ << ");\n";
            this->EndScope();
            this->PrintIndent(stream_);
            stream_ << "};\n";
            break;
          }
          case Opcode::AllocStorage: {
            if (lazy_execution()) {
              storage_device_indices[instr.dst] = instr.alloc_storage.device_index;
            } else {
              auto dst_var = GetVarForReg(instr.dst);
              auto allocation_size_var = GetVarForReg(instr.alloc_storage.allocation_size);
              auto dtype = instr.alloc_storage.dtype_hint;
              std::string dtype_str = "{" + std::to_string(dtype.code) + ", " +
                                      std::to_string(dtype.bits) + ", " +
                                      std::to_string(dtype.lanes) + "}";
              this->PrintIndent(stream_);
              std::string allocation_size_str = allocation_size_var;
              if (register_types_.at(instr.alloc_storage.allocation_size).as<TensorTypeNode>()) {
                allocation_size_var = "NDToInt64(" + allocation_size_var + ")";
              }
              stream_ << dst_var << " = " << GetRuntimeType() << "::Current()"
                      << "->AllocateStorage(" << allocation_size_str << ", "
                      << instr.alloc_storage.alignment << ", " << dtype_str << ", "
                      << instr.alloc_storage.device_index << ");\n";
            }
            break;
          }
          case Opcode::ShapeOf: {
            auto tensor_var = GetVarForReg(instr.reshape_tensor.tensor);
            auto dst_var = GetVarForReg(instr.dst);
            this->PrintIndent(stream_);
            stream_ << dst_var << " = " << GetRuntimeType() << "::Current()"
                    << "->ShapeOf(" << tensor_var << ");\n";
            break;
          }
          case Opcode::Ret: {
            if (same_depth_for_all_calls) {
              this->PrintIndent(stream_);
              stream_ << "depth = max({" << support::PrintVector(all_depth_vars, false) << "});\n";
            }
            auto result_var = GetVarForReg(instr.result);
            this->PrintIndent(stream_);
            stream_ << "return " << result_var << ";\n";
            break;
          }
          case Opcode::ReshapeTensor: {
            auto tensor_var = GetVarForReg(instr.reshape_tensor.tensor);
            auto shape_var = GetVarForReg(instr.reshape_tensor.newshape);
            auto dst_var = GetVarForReg(instr.dst);
            this->PrintIndent(stream_);
            stream_ << dst_var << " = " << GetRuntimeType() << "::Current()"
                    << "->ReshapeTensor(" << tensor_var << ", " << shape_var << ");\n";
            break;
          }
          case Opcode::DeviceCopy: {
            auto src_var = GetVarForReg(instr.device_copy.src);
            auto dst_var = GetVarForReg(instr.dst);

            this->PrintIndent(stream_);
            stream_ << dst_var << " = " << GetRuntimeType() << "::Current()"
                    << "->DeviceCopy(" << src_var << ", " << instr.device_copy.src_device_index
                    << ", " << instr.device_copy.dst_device_index << ");\n";
            break;
          }
          default:
            LOG(FATAL) << "Unknown instruction opcode: " << int(instr.op);
            return;
        }
      }
    };

    if (same_depth_for_all_calls) {
      this->PrintIndent(stream_);
      stream_ << "int __orig_depth = depth;\n";
    }
    generate_code(0, vm_func_.instructions.size());
    return max_static_depth;
  }

  void EmitPMapCPP(bool definition) {
    auto pmap_func_declaration =
        "template <class A, class B>\n"
        "std::shared_ptr<List<B>> pmap(std::function<B(A, int&)> local_0, "
        "std::shared_ptr<List<A>> "
        "local_1, int& depth)";
    stream_ << pmap_func_declaration;
    if (definition) {
      auto pmap_func_body =
          "{\n"
          "  auto current = local_1;\n"
          "  auto nil_node = std::static_pointer_cast<List<B>>(std::make_shared<Nil<B>>());\n"
          "  nil_node->tag = LIST_NIL_TAG;\n"
          "  auto new_list_head = nil_node;\n"
          "  auto new_list_tail = nil_node;\n"
          "  int map_depth_value = depth;\n"
          "  while (true) {\n"
          "    if (current->tag == LIST_NIL_TAG) {\n"
          "      break;\n"
          "    }\n"
          "    int tmp_depth = map_depth_value;\n"
          "    auto new_node = std::static_pointer_cast<List<B>>(std::make_shared<Cons<B>>());\n"
          "    new_node->tag = LIST_CONS_TAG;\n"
          "    static_cast<Cons<B>*>(new_node.get())->field_0 =\n"
          "        local_0(static_cast<Cons<A>*>(current.get())->field_0, tmp_depth);\n"
          "    depth = std::max(depth, tmp_depth);\n"
          "    if (new_list_tail->tag != LIST_NIL_TAG) {\n"
          "      static_cast<Cons<B>*>(new_list_tail.get())->field_1 = new_node;\n"
          "    } else {\n"
          "      new_list_head = new_node;\n"
          "    }\n"
          "    static_cast<Cons<B>*>(new_node.get())->field_1 = nil_node;\n"
          "    new_list_tail = new_node;\n"
          "    current = static_cast<Cons<A>*>(current.get())->field_1;\n"
          "  }\n"
          ""
          "  return new_list_head;\n"
          "}\n";
      stream_ << pmap_func_body << "\n" << std::endl;
    } else {
      stream_ << ";";
    }
  }

  const Executable& exec_;
  const IRModule& mod_;
  const VMFunction& vm_func_;
  const Function& relay_func_;
  const std::unordered_map<size_t, Type>& register_types_;
  const std::unordered_map<Index, Array<Type>>& invoke_type_vars_;
  const std::unordered_map<std::string, Function>& compiled_functions_;
  const std::unordered_map<Index, int32_t>& get_field_tags_;
  const std::unordered_map<Index, DictAttrs>& call_attrs_;
  const std::unordered_map<Index, std::array<Index, 4>>& if_offsets_;
  std::unordered_map<std::string, std::string> inbuilt_ops_;
  std::unordered_set<const TensorTypeNode*>* p_invoke_reduce_sum_types_;
  std::ostream& stream_;
};

void VMAOTCompiler::DeclareADT(std::ostream& os, const TypeData& adt, bool include_definitions) {
  if (adt->header->name_hint == "Storage") {
    return;
  }
  bool has_type_vars = (adt->type_vars.size() > 0);
  std::stringstream type_vars_decl_stream, type_vars_list_stream;
  std::string type_name = adt->header->name_hint;
  std::string type_name_with_type_vars = type_name;
  std::string type_vars_list, type_vars_decl;
  if (has_type_vars) {
    for (size_t i = 0; i < adt->type_vars.size(); ++i) {
      auto tvar = adt->type_vars[i];
      type_vars_decl_stream << "class " << tvar->name_hint;
      type_vars_list_stream << tvar->name_hint;
      if (i != adt->type_vars.size() - 1) {
        type_vars_decl_stream << ", ";
        type_vars_list_stream << ", ";
      }
    }
    type_vars_decl = type_vars_decl_stream.str();
    type_vars_list = type_vars_list_stream.str();
    type_name_with_type_vars += "<" + type_vars_list + ">";
  }

  // Create superclass
  if (has_type_vars) {
    os << "template<" << type_vars_decl << ">\n";
  }
  os << "class " << type_name;
  if (include_definitions) {
    os << " {\n";
    this->BeginScope();
    this->PrintIndent(os, -1);
    os << "public: \n";
    this->PrintIndent(os);
    os << "int32_t tag;\n";
    this->EndScope();
    os << "};\n";
  } else {
    os << ";\n";
  }

  // For each constructor, create a subclass
  for (auto constructor : adt->constructors) {
    std::string constructor_name = constructor->name_hint;

    os << "#define " << ToUpperCase(type_name) << "_" << ToUpperCase(constructor_name) << "_TAG "
       << constructor->tag << "\n";

    if (has_type_vars) {
      os << "template<" << type_vars_decl << ">\n";
    }

    if (include_definitions) {
      os << "class " << constructor_name << ": public " << type_name_with_type_vars;
      os << " {\n";
      this->BeginScope();
      this->PrintIndent(os, -1);
      os << "public: \n";
      for (size_t i = 0; i < constructor->inputs.size(); ++i) {
        auto field = constructor->inputs[i];
        this->PrintIndent(os);
        RelayTypeToCppStr(os, field);
        os << " " << GetFieldName(i) << ";\n";
      }
      this->EndScope();
      os << "};\n";
    } else {
      os << "class " << constructor_name << ";\n";
    }
  }
}

std::string replace_all(const std::string& str, const std::string& find,
                        const std::string& replace) {
  using namespace std;
  string result;
  size_t find_len = find.size();
  size_t pos, from = 0;
  while (string::npos != (pos = str.find(find, from))) {
    result.append(str, from, pos - from);
    result.append(replace);
    from = pos + find_len;
  }
  result.append(str, from, string::npos);
  return result;
}

void VMAOTCompiler::EmitUtilFunctions(
    std::ostream& os, const std::unordered_set<const TensorTypeNode*>& invoke_reduce_sum_types) {
  auto max_func =
      "int max(std::initializer_list<int> numbers) {\n"
      "  int res = std::numeric_limits<int>::min();\n"
      "  for (auto& num : numbers) {\n"
      "    res = (res > num) ? res : num;\n"
      "  }\n"
      "  return res;\n"
      "}\n";
  os << max_func << "\n" << std::endl;

  auto nd_to_int64_func =
      "int64_t NDToInt64(const NDArray& nd) {\n"
      "  static auto int64_dtype = DataType::Int(64);\n"
      "  DLDevice cpu_ctx{kDLCPU, 0};\n"
      "  NDArray cpu_array = nd.CopyTo(cpu_ctx);\n"
      "  CHECK_EQ(DataType(cpu_array->dtype), int64_dtype);\n"
      "  return reinterpret_cast<int64_t*>(cpu_array->data)[0];\n"
      "}\n";
  os << nd_to_int64_func << "\n" << std::endl;

  auto random_func =
      "int32_t GetRandom(int32_t lo, int32_t hi) {\n"
      "  static std::random_device rd;\n"
      "  static std::mt19937 gen(rd());\n"
      "  return std::uniform_int_distribution<>(lo, hi)(gen);\n"
      "}\n";
  os << random_func << "\n" << std::endl;

  if (invoke_reduce_sum_types.size() > 0) {
    auto list_append_func =
        "template <class A>\n"
        "std::shared_ptr<List<A>> append(std::shared_ptr<List<A>> list, A value) {\n"
        "  auto new_node = std::static_pointer_cast<List<A>>(std::make_shared<Cons<A>>());\n"
        "  new_node->tag = LIST_CONS_TAG;\n"
        "  static_cast<Cons<A>*>(new_node.get())->field_0 = value;\n"
        "  static_cast<Cons<A>*>(new_node.get())->field_1 = list;\n"
        "  return new_node;\n"
        "}\n";
    os << list_append_func << "\n" << std::endl;
  }

  for (auto tt : invoke_reduce_sum_types) {
    std::stringstream ss;
    bool use_depth = use_depth_tracking_executor();
    if (use_depth) {
      ss << "__TT__ invoke_reduce_sum(std::shared_ptr<List<__TT__>> list, int& depth) {\n";
    } else {
      ss << "__TT__ invoke_reduce_sum(std::shared_ptr<List<__TT__>> list) {\n";
    }
    ss << "  auto current = list;\n";
    ss << "  std::vector<__TT__> args;\n";
    ss << "  while (true) {\n";
    ss << "    if (current->tag == LIST_NIL_TAG) {\n";
    ss << "      break;\n";
    ss << "    }\n";
    ss << "    args.push_back(static_cast<Cons<__TT__>*>(current.get())->field_0);\n";
    ss << "    current = static_cast<Cons<__TT__>*>(current.get())->field_1;\n";
    ss << "  }\n";
    ss << "\n";
    ss << "  auto shape_data0 = Arena::Current()->allocate_<int64_t>(__ND__);\n";
    ss << "  shape_data0 = new (shape_data0) int64_t[__ND__]{__SH__};\n";
    ss << "  __TT__ sum_tensor =\n";
    ss << "      __RT__::Current()->AllocArrayWrapper(\n";
    ss << "          shape_data0, __ND__, __DT__, __DV__);\n";
    ss << "\n";
    ss << "  args.push_back(sum_tensor);\n";
    ss << "\n";
    if (use_depth) {
      ss << "  __RT__::Current()->InvokePackedWithDepth(\n";
      ss << "      __REDUCE_SUM_FUNC_INDEX__, depth++, args.data(), args.size());\n";
    } else {
      ss << "  __RT__::Current()->InvokePacked(\n";
      ss << "      __REDUCE_SUM_FUNC_INDEX__, args.data(), args.size());\n";
    }
    ss << "\n";
    ss << "  return sum_tensor;\n";
    ss << "}\n";

    auto shape = backend::GetIntShape(tt->shape);
    auto device_index = 1;

    auto body = ss.str();
    body = replace_all(body, "__TT__", GetTensorType());
    body = replace_all(body, "__RT__", GetRuntimeType());
    body = replace_all(body, "__REDUCE_SUM_FUNC_INDEX__", std::to_string(REDUCE_SUM_FUNC_INDEX));
    body = replace_all(body, "__ND__", std::to_string(shape.size()));
    body = replace_all(body, "__DV__", std::to_string(device_index));
    body = replace_all(body, "__DT__", DTypeToStr(tt->dtype));
    body = replace_all(body, "__SH__", support::PrintVector(shape, false));
    os << body;
  }
}

void VMAOTCompiler::EmitBatchedMainFunction(std::ostream& os, int start_depth) {
  // Emit a batched main function, first

  auto main_relay_func = Downcast<Function>(mod_->Lookup("main"));
  FuncType function_type = Downcast<FuncType>(main_relay_func->checked_type_);

  EmitBatchedMainFunctionHeader(os);
  os << " {\n";
  this->BeginScope();
  this->PrintIndent(os);
  os << "auto batch_size = " << GetVarForReg(0) << ".size();\n";
  this->PrintIndent(os);
  os << "std::vector<";
  RelayTypeToCppStr(os, function_type->ret_type);
  os << "> res;\n";
  this->PrintIndent(os);
  os << "res.reserve(batch_size);\n";
  this->PrintIndent(os);
  os << "for (size_t b = 0; b < batch_size; ++b) {\n";
  this->BeginScope();
  this->PrintIndent(os);
  if (use_depth_tracking_executor()) {
    os << "int depth = " << start_depth << ";\n";
  }
  this->PrintIndent(os);
  os << "res.push_back(" << GetCppFunctionName("main") << "(";
  for (size_t i = 0; i < function_type->arg_types.size(); ++i) {
    auto arg_type = function_type->arg_types[i];
    os << " " << GetVarForReg(i) << "[b]";
    if (i != function_type->arg_types.size() - 1) {
      os << ", ";
    }
  }
  if (use_depth_tracking_executor()) {
    os << ", depth";
  }
  os << "));\n";
  this->EndScope();
  this->PrintIndent(os);
  os << "}\n";
  auto pass_ctx = transform::PassContext::Current();
  // if (lazy_execution()) {
  //   this->PrintIndent(os);
  //   os << GetRuntimeType() << "::Current()->LazyExecute();\n";
  // }
  this->PrintIndent(os);
  os << "return res;\n";
  this->EndScope();
  this->PrintIndent(os);
  os << "}\n\n";
}

void VMAOTCompiler::EmitBatchedMainFunctionHeader(std::ostream& os) {
  // Emit a batched main function, first
  auto main_relay_func = Downcast<Function>(mod_->Lookup("main"));

  FuncType function_type = Downcast<FuncType>(main_relay_func->checked_type_);

  if (main_relay_func->type_params.size() > 0) {
    os << "template<";
    for (size_t i = 0; i < function_type->type_params.size(); ++i) {
      auto tvar = function_type->type_params[i];
      os << "class " << tvar->name_hint;
      if (i != function_type->type_params.size() - 1) {
        os << ", ";
      }
    }
    os << ">\n";
  }
  os << "std::vector<";
  RelayTypeToCppStr(os, function_type->ret_type);
  os << "> batched_main(";
  for (size_t i = 0; i < function_type->arg_types.size(); ++i) {
    auto arg_type = function_type->arg_types[i];
    os << "std::vector<";
    RelayTypeToCppStr(os, arg_type);
    if (arg_type.as<TensorTypeNode>() && !lazy_execution()) {
      os << "&";
    }
    os << "> " << GetVarForReg(i);
    if (i != function_type->arg_types.size() - 1) {
      os << ", ";
    }
  }
  os << ")";
}

inline std::string Bool2Str(bool a) { return a ? "true" : "false"; }

void VMAOTCompiler::EmitHarnessFunctions(std::ostream& os) {
  for (auto d : exec_.virtual_devices) {
    std::cout << "DEVICE " << d << std::endl;
  }
  os << "std::pair<float, float> measure_time(std::function<std::pair<float, float>()> "
        "runner, bool profiling) {\n";
  os << "  int w_iters = 50;\n";
  os << "  int a_iters = 100;\n";
  os << "  for (int i = 0; i < w_iters; ++i) {\n";
  os << "    runner();\n";
  os << "  }\n";

  os << "  if (profiling) {\n";
  os << "    VMDBProfiler::ProfileStart();\n";
  os << "  }\n";

  os << "  float cg_gen_time = 0.0;\n";
  os << "  float cg_exe_time = 0.0;\n";
  os << "  for (int i = 0; i < a_iters; ++i) {\n";
  os << "    auto p = runner();\n";
  os << "    cg_gen_time += p.first;\n";
  os << "    cg_exe_time += p.second;\n";
  os << "  }\n\n";

  os << "  if (profiling) {\n";
  os << "    VMDBProfiler::ProfileStop();\n";
  os << "  }\n";

  os << "  return std::make_pair(cg_gen_time / a_iters, cg_exe_time / a_iters);\n";
  os << "}\n\n";

  os << "int main(int argc, char* argv[]) {\n";
  os << "  std::string dir = argv[1];\n";
  os << "  std::string code_path = dir + \"/" << model_name_ << ".ro\";\n";
  os << "  std::string lib_path = dir + \"/" << model_name_ << "_lib.so\";\n";

  os << "  std::ifstream code_in(code_path);\n";
  os << "  std::string code((std::istreambuf_iterator<char>(code_in)), "
        "std::istreambuf_iterator<char>());\n";
  os << "  code_in.close();\n";

  os << "  auto lib = Module::LoadFromFile(lib_path, \"so\");\n";
  os << "  auto exec_module = Executable::Load(code, lib);\n";
  os << "  auto exec_ptr = const_cast<Executable*>(static_cast<const "
        "Executable*>(exec_module.get()));\n";
  os << "  auto runtime = " << GetRuntimeType() << "::CreateRuntime();\n";

  auto pass_ctx = transform::PassContext::Current();
  bool coarsened_execution =
      pass_ctx->GetConfig<Bool>("relay.db_coarsen_granularity", Bool(false)).value();
  bool lazy_execution = pass_ctx->GetConfig<Bool>("relay.db_lazy_execution", Bool(false)).value();
  bool batched_execution =
      pass_ctx->GetConfig<Bool>("relay.db_batched_execution", Bool(false)).value();
  bool scattered_kernels =
      pass_ctx->GetConfig<Bool>("relay.db_scattered_kernels", Bool(false)).value();
  bool concurrent_execution =
      pass_ctx->GetConfig<Bool>("relay.db_concurrent_execution", Bool(false)).value();
  size_t batch_size = pass_ctx->GetConfig<Integer>("relay.db_batch_size", Integer(1)).value();

  os << "  bool coarsened_execution = " << Bool2Str(coarsened_execution) << ";\n";
  os << "  bool lazy_execution = " << Bool2Str(lazy_execution) << ";\n";
  os << "  bool batched_execution = " << Bool2Str(batched_execution) << ";\n";
  os << "  bool scattered_kernels = " << Bool2Str(scattered_kernels) << ";\n";
  os << "  bool concurrent_execution = " << Bool2Str(concurrent_execution) << ";\n";
  os << "  Arena::Init();\n";
  os << "  size_t batch_size = " << batch_size << ";\n";

  os << "  VMExecutionOptions options(coarsened_execution, lazy_execution, batched_execution,\n";
  os << "                             scattered_kernels, concurrent_execution, batch_size);\n";
  os << "  runtime->SetExecutionOptions(options);\n";
  os << "  runtime->InitSharedState();\n";
  os << "  runtime->LoadExecutable(exec_ptr);\n";

  std::stringstream device_list, alloc_list;
  for (size_t i = 0; i < exec_.virtual_devices.size(); ++i) {
    auto& d = exec_.virtual_devices[i];
    device_list << "DLDevice{static_cast<DLDeviceType>(" << d.device_type << "), " << d.device_id
                << "}";
    alloc_list << "kPooled";
    if (i < exec_.virtual_devices.size() - 1) {
      device_list << ", ";
      alloc_list << ", ";
    }
  }

  os << "  std::vector<Device> devices = {" << device_list.str() << "};\n";

  os << "  runtime->Init({devices}, {" << alloc_list.str() << "});\n";

  os << "  runtime->CacheConstants();\n";
  os << "  invoke_model<" << GetTensorType() << ">(devices, argc - 2, &(argv[2]));\n";
  os << "}\n\n";
}

void VMAOTCompiler::EmitHarnessFunctionHeaders(std::ostream& os) {
  os << "template <typename TensorType>\n";
  os << "void invoke_model(std::vector<Device> devices, int argc, char* argv[]);\n";
  os << "std::pair<float, float> measure_time(std::function<std::pair<float, float>()> runner, "
        "bool profiling = false);\n";
}

void VMAOTCompiler::GenerateCppFile(std::string header_file_name) {
  std::stringstream cpp_model_stream_;
  std::stringstream cpp_utils_stream_;

  cpp_utils_stream_ << "#include \"" << header_file_name << "\"\n\n";

  std::unordered_set<const TensorTypeNode*> invoke_reduce_sum_types;
  int max_static_depth = -1;
  for (auto vm_func : exec_.functions) {
    Function relay_func = GetCompiledRelayFunction(compiled_functions_, mod_, vm_func.name);
    ICHECK(register_types_.count(vm_func.name)) << vm_func.name;
    auto function_register_types = register_types_.at(vm_func.name);
    auto function_invoke_type_vars = invoke_type_vars_.at(vm_func.name);
    auto function_get_field_tags = get_field_tags_.at(vm_func.name);
    auto function_call_attrs = call_attrs_.at(vm_func.name);
    auto function_if_offsets = if_offsets_.at(vm_func.name);
    VMAOTFunctionCompiler function_compiler(
        exec_, mod_, vm_func, relay_func, function_register_types, function_invoke_type_vars,
        compiled_functions_, function_get_field_tags, function_call_attrs, function_if_offsets,
        &invoke_reduce_sum_types, cpp_model_stream_);
    max_static_depth = std::max(max_static_depth, function_compiler.GenerateCPPForFunction(true));
    cpp_stream_ << "\n";
  }

  EmitUtilFunctions(cpp_utils_stream_, invoke_reduce_sum_types);

  EmitBatchedMainFunction(cpp_model_stream_, max_static_depth + 1);
  EmitHarnessFunctions(cpp_model_stream_);
  cpp_stream_ << cpp_utils_stream_.str() << "\n\n";
  cpp_stream_ << cpp_model_stream_.str();
}

void VMAOTCompiler::EmitMacros(std::ostream& os) {}

void VMAOTCompiler::EmitHeaderIncludes(std::ostream& os) {
  os << "#include <cstdint>\n";
  os << "#include <dlpack/dlpack.h>\n";
  os << "#include <tvm/runtime/vm/vm_profiling.h>\n";
  os << "#include <tvm/runtime/vm/db_runtime.h>\n";
  os << "#include <tvm/runtime/vm/arena.h>\n";
  os << "#include <stdexcept>\n";
  os << "#include <vector>\n";
  os << "#include <random>\n";
  os << "#include <fstream>\n";
  os << "#include <functional>\n";
  os << "#include <array>\n\n";

  os << "using namespace tvm;\n";
  os << "using namespace tvm::runtime;\n";
  os << "using namespace tvm::runtime::vm;\n";
}

void VMAOTCompiler::GenerateHeaderFile(std::string header_file_name) {
  // Header guard
  hpp_stream_ << "#ifndef TVM_RELAY_BACKEND_VM_AOT_COMPILER_H_\n";
  hpp_stream_ << "#define TVM_RELAY_BACKEND_VM_AOT_COMPILER_H_\n";

  // Add includes
  this->EmitHeaderIncludes(hpp_stream_);

  hpp_stream_ << "\n\n";

  // Add MACRO definitions
  this->EmitMacros(hpp_stream_);

  // Forward declare all ADTs first
  for (auto var : mod_->GetGlobalTypeVars()) {
    this->DeclareADT(hpp_stream_, mod_->LookupTypeDef(var), false);
  }

  hpp_stream_ << "\n\n";

  // Then add definations for all ADTs
  for (auto var : mod_->GetGlobalTypeVars()) {
    this->DeclareADT(hpp_stream_, mod_->LookupTypeDef(var), true);
  }

  // Declare all the functions
  for (auto vm_func : exec_.functions) {
    Function relay_func = GetCompiledRelayFunction(compiled_functions_, mod_, vm_func.name);
    ICHECK(register_types_.count(vm_func.name)) << vm_func.name;
    auto function_register_types = register_types_.at(vm_func.name);
    auto function_invoke_type_vars = invoke_type_vars_.at(vm_func.name);
    auto function_get_field_tags = get_field_tags_.at(vm_func.name);
    auto function_call_attrs = call_attrs_.at(vm_func.name);
    auto function_if_offsets = if_offsets_.at(vm_func.name);
    VMAOTFunctionCompiler function_compiler(
        exec_, mod_, vm_func, relay_func, function_register_types, function_invoke_type_vars,
        compiled_functions_, function_get_field_tags, function_call_attrs, function_if_offsets,
        nullptr, hpp_stream_);
    function_compiler.GenerateCPPForFunction(false);
  }

  hpp_stream_ << "\n";

  EmitHarnessFunctionHeaders(hpp_stream_);

  EmitBatchedMainFunctionHeader(hpp_stream_);
  hpp_stream_ << ";\n";
  // Header guard
  hpp_stream_ << "#endif\n";
}

void VMAOTCompiler::Codegen() {
  std::string header_file_name = model_name_ + "_src.hpp";
  std::string cpp_file_name = model_name_ + "_src.cpp";
  GenerateCppFile(header_file_name);
  GenerateHeaderFile(header_file_name);

  // std::cout << "HEADER HEADER HEADER HEADER HEADER" << std::endl;
  // std::cout << hpp_stream_.str() << "\n\n" << std::endl;
  // std::cout << "SOURCE SOURCE SOURCE SOURCE SOURCE" << std::endl;
  // std::cout << cpp_stream_.str() << "\n\n" << std::endl;

  std::ofstream hpp_file_stream, cpp_file_stream;
  hpp_file_stream.open(output_directory_ + "/" + header_file_name);
  cpp_file_stream.open(output_directory_ + "/" + cpp_file_name);

  hpp_file_stream << hpp_stream_.str();
  cpp_file_stream << cpp_stream_.str();

  hpp_file_stream.close();
  cpp_file_stream.close();

  std::cout << "[AOT] Created files" << std::endl;
  std::cout << "[AOT]  " << output_directory_ + "/" + header_file_name << std::endl;
  std::cout << "[AOT]  " << output_directory_ + "/" + cpp_file_name << std::endl;
}

}  // namespace vm
}  // namespace relay
}  // namespace tvm
