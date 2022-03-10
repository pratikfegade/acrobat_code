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
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

std::string DTypeToStr(const DLDataType& dtype) {
  return "{" + std::to_string(dtype.code) + ", " + std::to_string(dtype.bits) + ", " +
         std::to_string(dtype.lanes) + "}";
}

void RelayTypeToCppStr(std::ostream& os, const Type& type) {
  if (auto tt = type.as<TensorTypeNode>()) {
    os << "NDArray";
  } else if (auto pt = type.as<PrimTypeNode>()) {
    os << pt->dtype;
  } else if (auto td = type.as<TypeDataNode>()) {
    if (td->header->name_hint == "Storage") {
      os << "Storage";
    } else {
      os << td->header->name_hint << "*";
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
    os << ")>";
  } else if (auto tv = type.as<TypeVarNode>()) {
    os << tv->name_hint;
  } else if (auto tt = type.as<TupleTypeNode>()) {
    os << "std::tuple<";
    for (size_t i = 0; i < tt->fields.size(); ++i) {
      RelayTypeToCppStr(os, tt->fields[i]);
      if (i < tt->fields.size() - 1) {
        os << ",";
      }
    }
    os << ">";
  } else if (auto tc = type.as<TypeCallNode>()) {
    auto type_func_gv = tc->func.as<GlobalTypeVarNode>();
    ICHECK(type_func_gv) << type;
    os << type_func_gv->name_hint;
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
    if (type_func_gv->name_hint != "Storage") {
      os << "*";
    }
  } else {
    std::cout << "DEFAULT " << type << std::endl;
    os << "DEFAULT";
  }
}

std::string GetModelMainFunctionName() { return "model_main"; }

std::string GetCppFunctionName(const std::string& name) {
  if (name == "main") {
    return GetModelMainFunctionName();
  }
  return name;
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

class VMAOTFunctionCompiler : SourcePrinter {
 public:
  VMAOTFunctionCompiler(const Executable& exec, const IRModule& mod, const VMFunction& vm_func,
                        const Function& relay_func,
                        const std::unordered_map<size_t, Type>& register_types,
                        const std::unordered_map<Index, Array<Type>>& invoke_type_vars,
                        const std::unordered_map<std::string, Function>& compiled_functions)
      : exec_(exec),
        mod_(mod),
        vm_func_(vm_func),
        relay_func_(relay_func),
        register_types_(register_types),
        invoke_type_vars_(invoke_type_vars),
        compiled_functions_(compiled_functions) {}

  void GenerateCPPForFunction() {
    std::cout << "[FUN] Visiting " << vm_func_.name << std::endl;
    Type function_type = relay_func_->checked_type_;
    CreateFunctionDeclaration(vm_func_, relay_func_);

    stream_ << " {\n";
    this->BeginScope();

    this->GenerateLocalDecls();

    this->VisitBytecode();

    this->EndScope();
    stream_ << "}\n";

    std::cout << "[FUN] Visited " << vm_func_.name << std::endl;
    std::cout << "[FUN]\n" << stream_.str() << std::endl;
  }

 private:
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
      stream_ << " " << GetVarForReg(i);
      if (i != function_type->arg_types.size() - 1) {
        stream_ << ", ";
      }
    }
    stream_ << ")";

    // std::cout << relay_func << "\n" << std::endl;
    // std::cout << stream_.str() << std::endl;
    // if (vm_func.params.size() != function_type->arg_types.size()) {
    //   std::cout << " VM Func Params" << std::endl;
    //   for (auto param : vm_func.params) {
    //     std::cout << "   " << param << std::endl;
    //   }

    //   std::cout << "  Relay Func Params" << std::endl;
    //   for (auto param : relay_func->params) {
    //     std::cout << "   " << param->name_hint() << std::endl;
    //   }
    // }
    // std::cout << "\n\n" << std::endl;
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

  void GenerateLocalDecls() {
    std::unordered_map<std::string, std::vector<std::string>> type2vars;
    for (size_t i = vm_func_.params.size(); i < vm_func_.register_file_size; ++i) {
      auto it = register_types_.find(i);
      ICHECK(it != register_types_.end()) << i;
      Type reg_type = it->second;
      if (reg_type.as<TensorTypeNode>() || reg_type.as<PrimTypeNode>() || IsStorageType(reg_type)) {
        std::stringstream ss;
        RelayTypeToCppStr(ss, reg_type);
        auto type_str = ss.str();
        type2vars[type_str].push_back(GetVarForReg(i));
      } else {
        this->PrintIndent();
        RelayTypeToCppStr(stream_, reg_type);
        stream_ << " " << GetVarForReg(i) << ";\n";
      }
    }

    for (auto kv : type2vars) {
      this->PrintIndent();
      stream_ << kv.first << " ";
      auto vars = kv.second;
      for (size_t i = 0; i < vars.size(); ++i) {
        stream_ << vars[i];
        if (i < vars.size() - 1) {
          stream_ << ", ";
        }
      }
      stream_ << ";\n";
    }
    stream_ << "\n";
  }

  void VisitBytecode() {
    // std::cout << "[BT] Visiting BT" << std::endl;
    std::unordered_map<Index, std::string> targets;
    int target_count = 0;

    // std::cout << "[BT]  Visiting targets" << std::endl;
    for (size_t i = 0; i < vm_func_.instructions.size(); ++i) {
      auto& instr = vm_func_.instructions[i];
      if (instr.op == Opcode::Goto) {
        targets[instr.pc_offset + i] = "target" + std::to_string(target_count++);
        // std::cout << "[BT]   Added goto target " << i << " " << instr.pc_offset + i << " "
        // << targets[instr.pc_offset + i] << std::endl;
      } else if (instr.op == Opcode::If) {
        targets[instr.if_op.true_offset + i] = "target" + std::to_string(target_count++);
        targets[instr.if_op.false_offset + i] = "target" + std::to_string(target_count++);
        // std::cout << "[BT]   Added if target " << i << " " << instr.if_op.true_offset + i << " "
        // << targets[instr.if_op.true_offset + i] << std::endl;
        // std::cout << "[BT]   Added else target " << i << " " << instr.if_op.false_offset + i << "
        // "
        // << targets[instr.if_op.false_offset + i] << std::endl;
      }
    }

    // std::cout << "[BT]  Visiting code" << std::endl;
    int tmp_var_counter = 0;
    for (size_t i = 0; i < vm_func_.instructions.size(); ++i) {
      auto& instr = vm_func_.instructions[i];
      // std::cout << "[BT]   " << instr << std::endl;

      auto it = targets.find(i);
      if (it != targets.end()) {
        this->PrintIndent(-1);
        stream_ << it->second << ":\n";
      }

      switch (instr.op) {
        case Opcode::Move: {
          auto src_var = GetVarForReg(instr.from);
          auto dst_var = GetVarForReg(instr.dst);
          this->PrintIndent();
          stream_ << dst_var << " = " << src_var << ";\n";
          break;
        }
        case Opcode::Fatal: {
          this->PrintIndent();
          stream_ << "throw std::runtime_error(\"Fatal error\");\n";
          break;
        }
        case Opcode::LoadConst: {
          auto dst_var = GetVarForReg(instr.dst);
          ICHECK(register_types_.at(instr.dst).as<TensorTypeNode>());
          this->PrintIndent();
          stream_ << "TVM_API_CALL(TVMDBLoadConstant(" << instr.const_index << ", &" << dst_var
                  << "));\n";
          break;
        }
        case Opcode::LoadConsti: {
          auto dst_var = GetVarForReg(instr.dst);
          this->PrintIndent();
          stream_ << dst_var << " = " << instr.load_consti.val << ";\n";
          break;
        }
        case Opcode::Invoke: {
          auto dst_var = GetVarForReg(instr.dst);
          auto callee_name = exec_.functions[instr.func_index].name;
          this->PrintIndent();
          stream_ << dst_var << " = " << callee_name;
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
          for (size_t i = 0; i < instr.num_args; ++i) {
            stream_ << GetVarForReg(instr.invoke_args_registers[i]);
            if (i < instr.num_args - 1) {
              stream_ << ",";
            }
          }
          stream_ << ");\n";
          break;
        }
        case Opcode::InvokePacked: {
          this->PrintIndent();
          auto args_vec = "args_tmp" + std::to_string(tmp_var_counter++);
          stream_ << "std::vector<ObjectRef> " << args_vec << " = {";
          for (int j = 0; j < instr.arity; ++j) {
            stream_ << GetVarForReg(instr.packed_args[j]);
            if (j < instr.arity - 1) {
              stream_ << ", ";
            }
          }
          stream_ << "};\n";
          this->PrintIndent();
          stream_ << "TVM_API_CALL(TVMDBInvokePacked(" << instr.packed_index << ", " << instr.arity
                  << ", " << instr.output_size << ", " << args_vec << ".data(), " << instr.arity
                  << "));\n";
          break;
        }
        case Opcode::InvokeClosure: {
          auto dst_var = GetVarForReg(instr.dst);
          auto callee_var = GetVarForReg(instr.closure);
          this->PrintIndent();
          stream_ << dst_var << " = " << callee_var;
          // auto it = invoke_type_vars_.find(i);
          // if (it != invoke_type_vars_.end() && it->second.size() > 0) {
          //   auto types = it->second;
          //   stream_ << "<";
          //   for (size_t j = 0; j < types.size(); ++j) {
          //     RelayTypeToCppStr(stream_, types[j]);
          //     if (j < types.size() - 1) {
          //       stream_ << ",";
          //     }
          //   }
          //   stream_ << ">";
          // }
          stream_ << "(";
          for (size_t i = 0; i < instr.num_closure_args; ++i) {
            stream_ << GetVarForReg(instr.closure_args[i]);
            if (i < instr.num_closure_args - 1) {
              stream_ << ",";
            }
          }
          stream_ << ");\n";
          break;
        }  //
        case Opcode::GetField: {
          auto object_var = GetVarForReg(instr.object);
          auto dst_var = GetVarForReg(instr.dst);
          this->PrintIndent();
          stream_ << dst_var << " = " << object_var << "->" << this->GetFieldName(instr.field_index)
                  << ";\n";
          break;
        }
        case Opcode::GetTag: {
          auto object_var = GetVarForReg(instr.object);
          auto dst_var = GetVarForReg(instr.dst);
          this->PrintIndent();
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
          this->PrintIndent();
          stream_ << "goto " << label << ";\n";
          break;
        }
        case Opcode::If: {
          Index true_target = i + instr.if_op.true_offset;
          auto true_label = targets.at(true_target);

          Index false_target = i + instr.if_op.false_offset;
          auto false_label = targets.at(false_target);

          auto test_var = GetVarForReg(instr.if_op.test);
          auto target_var = GetVarForReg(instr.if_op.target);

          this->PrintIndent();
          stream_ << "if (" << test_var << " = " << target_var << ") {\n";
          this->BeginScope();
          this->PrintIndent();
          stream_ << "goto " << true_label << ";\n";
          this->EndScope();
          this->PrintIndent();
          stream_ << "} else {\n";
          this->BeginScope();
          this->PrintIndent();
          stream_ << "goto " << false_label << ";\n";
          this->EndScope();
          this->PrintIndent();
          stream_ << "}\n";
          break;
        }
        case Opcode::AllocTensor: {
          auto dst_var = GetVarForReg(instr.dst);
          auto storage_var = GetVarForReg(instr.alloc_tensor.storage);
          auto offset_var = GetVarForReg(instr.alloc_tensor.offset);
          std::string dtype_str = DTypeToStr(instr.alloc_tensor.dtype);

          std::string shape_arr_name = "shape_arr" + std::to_string(tmp_var_counter++);
          this->PrintIndent();
          stream_ << "std::array<int64_t, " << instr.alloc_tensor.ndim << "> " << shape_arr_name
                  << " = {";
          for (size_t j = 0; j < instr.alloc_tensor.ndim; ++j) {
            stream_ << instr.alloc_tensor.shape[j];
            if (j < instr.alloc_tensor.ndim - 1) {
              stream_ << ", ";
            }
          }
          stream_ << "};\n";

          // TVM_DLL int TVMDBAllocateTensor(const tvm::runtime::vm::Storage& storage, int64_t
          // offset,
          //                                 uint32_t ndim, int64_t* shape, DLDataType dtype,
          //                                 tvm::runtime::NDArray* out);

          this->PrintIndent();
          stream_ << "TVM_API_CALL(TVMDBAllocateTensor(" << storage_var << ", " << offset_var
                  << ", " << instr.alloc_tensor.ndim << ", " << shape_arr_name << ".data(), "
                  << dtype_str << ", &" << dst_var << "));\n";
          break;
        }
        case Opcode::AllocTensorReg: {
          auto dst_var = GetVarForReg(instr.dst);
          auto storage_var = GetVarForReg(instr.alloc_tensor_reg.storage);
          auto offset_var = GetVarForReg(instr.alloc_tensor.offset);
          std::string dtype_str = DTypeToStr(instr.alloc_tensor_reg.dtype);

          std::string shape_var = GetVarForReg(instr.alloc_tensor_reg.shape_register);
          this->PrintIndent();
          stream_ << "TVM_API_CALL(TVMDBAllocateTensorReg(" << storage_var << ", " << offset_var
                  << ", " << shape_var << ", " << dtype_str << ", &" << dst_var << "));\n";
          break;
        }
        case Opcode::AllocADT: {
          auto dst_var = GetVarForReg(instr.dst);
          this->PrintIndent();
          if (instr.constructor_tag == 0) {
            stream_ << dst_var << " = allocate<std::tuple<";
            for (size_t j = 0; j < instr.num_fields; ++j) {
              ICHECK(register_types_.count(instr.datatype_fields[j])) << instr.datatype_fields[j];
              Type field_type = register_types_.at(instr.datatype_fields[j]);
              RelayTypeToCppStr(stream_, field_type);
              if (j < instr.num_fields - 1) {
                stream_ << ",";
              }
            }
            stream_ << ">>(";
            for (size_t j = 0; j < instr.num_fields; ++j) {
              ICHECK(register_types_.count(instr.datatype_fields[j])) << instr.datatype_fields[j];
              auto field_var = GetVarForReg(instr.datatype_fields[j]);
              stream_ << field_var;
              if (j < instr.num_fields - 1) {
                stream_ << ",";
              }
            }
            stream_ << ");\n";
          } else {
            auto constructor = mod_->LookupTag(instr.constructor_tag);
            std::string type_name = constructor->belong_to->name_hint;
            stream_ << dst_var << " = allocate<" << type_name;

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

            stream_ << ">();\n";
            this->PrintIndent();
            stream_ << dst_var << "->tag = " << instr.constructor_tag << ";\n";
            auto constructor_field_name = ToLowerCase(constructor->name_hint);
            for (size_t j = 0; j < instr.num_fields; ++j) {
              ICHECK(register_types_.count(instr.datatype_fields[j])) << instr.datatype_fields[j];
              auto field_var = GetVarForReg(instr.datatype_fields[j]);
              this->PrintIndent();
              stream_ << dst_var << "->" << constructor_field_name << "." << GetFieldName(j)
                      << " = " << field_var << ";\n";
            }
          }
          break;
        }
        case Opcode::AllocClosure: {
          std::cout << "[Closure] " << exec_.functions[instr.clo_index].name << " "
                    << exec_.functions[instr.clo_index].params.size() << " " << instr.num_freevar
                    << std::endl;
          auto& closure_func = exec_.functions[instr.clo_index];
          auto closure_relay_func =
              GetCompiledRelayFunction(compiled_functions_, mod_, closure_func.name);
          auto dst_var = GetVarForReg(instr.dst);
          this->PrintIndent();
          stream_ << dst_var << " = [";
          for (size_t i = 0; i < instr.num_freevar; ++i) {
            stream_ << GetVarForReg(instr.invoke_args_registers[i]);
            if (i < instr.num_freevar - 1) {
              stream_ << ",";
            }
          }
          stream_ << "](";
          for (size_t j = instr.num_freevar; j < closure_func.params.size(); ++j) {
            Type arg_type = closure_relay_func->params[j]->checked_type_;
            RelayTypeToCppStr(stream_, arg_type);
            stream_ << " " << GetTmpVarName(j);
            if (j < closure_func.params.size() - 1) {
              stream_ << ", ";
            }
          }
          stream_ << ") {\n";
          this->BeginScope();
          this->PrintIndent();
          stream_ << "return " << closure_func.name << "(";
          size_t j = 0;
          for (; j < instr.num_freevar; ++j) {
            stream_ << GetVarForReg(instr.invoke_args_registers[i]);
            if (j < closure_func.params.size() - 1) {
              stream_ << ", ";
            }
          }
          for (; j < closure_func.params.size(); ++j) {
            stream_ << GetTmpVarName(j);
            if (j < closure_func.params.size() - 1) {
              stream_ << ", ";
            }
          }
          stream_ << ");\n";
          this->EndScope();
          this->PrintIndent();
          stream_ << "}\n";
          break;
        }  //
        case Opcode::AllocStorage: {
          auto dst_var = GetVarForReg(instr.dst);
          auto allocation_size_var = GetVarForReg(instr.alloc_storage.allocation_size);
          auto dtype = instr.alloc_storage.dtype_hint;
          std::string dtype_str = "{" + std::to_string(dtype.code) + ", " +
                                  std::to_string(dtype.bits) + ", " + std::to_string(dtype.lanes) +
                                  "}";
          this->PrintIndent();
          stream_ << "TVM_API_CALL(TVMDBAllocateStorage(" << allocation_size_var << ", "
                  << instr.alloc_storage.alignment << ", " << dtype_str << ", "
                  << instr.alloc_storage.device_index << ", &" << dst_var << "));\n";
          break;
        }
        case Opcode::ShapeOf: {
          auto tensor_var = GetVarForReg(instr.reshape_tensor.tensor);
          auto dst_var = GetVarForReg(instr.dst);
          this->PrintIndent();
          stream_ << "TVM_API_CALL(TVMDBShapeOf(" << tensor_var << ", &" << dst_var << "));\n";
          break;
        }
        case Opcode::Ret: {
          auto result_var = GetVarForReg(instr.result);
          this->PrintIndent();
          stream_ << "return " << result_var << ";\n";
          break;
        }
        case Opcode::ReshapeTensor: {
          auto tensor_var = GetVarForReg(instr.reshape_tensor.tensor);
          auto shape_var = GetVarForReg(instr.reshape_tensor.newshape);
          auto dst_var = GetVarForReg(instr.dst);
          this->PrintIndent();
          stream_ << "TVM_API_CALL(TVMDBReshapeTensor(" << tensor_var << ", " << shape_var << ", &"
                  << dst_var << "));\n";
          break;
        }
        case Opcode::DeviceCopy: {
          auto src_var = GetVarForReg(instr.device_copy.src);
          auto dst_var = GetVarForReg(instr.dst);

          this->PrintIndent();
          stream_ << "TVM_API_CALL(TVMDBDeviceCopy(" << src_var << ", "
                  << instr.device_copy.src_device_index << ", "
                  << instr.device_copy.dst_device_index << ", &" << dst_var << "));\n";
          break;
        }
        default:
          LOG(FATAL) << "Unknown instruction opcode: " << int(instr.op);
          return;
      }
    }
    // std::cout << "[BT] Done visiting BT" << std::endl;
  }

  const Executable& exec_;
  const IRModule& mod_;
  const VMFunction& vm_func_;
  const Function& relay_func_;
  const std::unordered_map<size_t, Type>& register_types_;
  const std::unordered_map<Index, Array<Type>>& invoke_type_vars_;
  const std::unordered_map<std::string, Function>& compiled_functions_;
};

void VMAOTCompiler::EmitHeader() {
  const std::string api_call_check_macro =
      "#define TVM_API_CALL(call)                               \\"
      "  if (call != 0) {                                       \\"
      "    throw std::runtime_error(\"API call returned error\"); \\"
      "  }";
  stream_ << api_call_check_macro << "\n\n";
}

void VMAOTCompiler::DeclareADT(const TypeData& adt) {
  if (adt->type_vars.size() > 0) {
    stream_ << "template<";
    for (size_t i = 0; i < adt->type_vars.size(); ++i) {
      auto tvar = adt->type_vars[i];
      stream_ << "class " << tvar->name_hint;
      if (i != adt->type_vars.size() - 1) {
        stream_ << ", ";
      }
    }
    stream_ << ">\n";
  }

  stream_ << "struct " << adt->header->name_hint << " {\n";
  this->BeginScope();
  this->PrintIndent();
  stream_ << "int32_t tag;\n\n";
  this->PrintIndent();
  stream_ << "union {\n";
  this->BeginScope();

  for (auto constructor : adt->constructors) {
    this->PrintIndent();
    stream_ << "struct {\n";
    this->BeginScope();
    for (size_t i = 0; i < constructor->inputs.size(); ++i) {
      auto field = constructor->inputs[i];
      this->PrintIndent();
      RelayTypeToCppStr(stream_, field);
      stream_ << " " << GetFieldName(i) << ";\n";
    }
    this->EndScope();
    this->PrintIndent();
    stream_ << "} " << ToLowerCase(constructor->name_hint) << ";\n";
  }
  this->EndScope();
  this->PrintIndent();
  stream_ << "};\n";

  this->EndScope();
  this->PrintIndent();
  stream_ << "};\n\n";
}

void VMAOTCompiler::GenerateMainFunction() {
  stream_ << "int main(int argc, char *argv[]) {\n";
  this->BeginScope();

  this->EndScope();
  stream_ << "}\n";
}

void VMAOTCompiler::GenerateCPP() {
  for (auto var : mod_->GetGlobalTypeVars()) {
    auto type = mod_->LookupTypeDef(var);
    // std::cout << "[TYPES] " << var << " " << type << std::endl;
    this->DeclareADT(type);
  }
  std::cout << stream_.str() << std::endl;
  for (auto vm_func : exec_.functions) {
    Function relay_func = GetCompiledRelayFunction(compiled_functions_, mod_, vm_func.name);
    ICHECK(register_types_.count(vm_func.name)) << vm_func.name;
    auto function_register_types = register_types_.at(vm_func.name);
    auto function_invoke_type_vars = invoke_type_vars_.at(vm_func.name);
    VMAOTFunctionCompiler function_compiler(exec_, mod_, vm_func, relay_func,
                                            function_register_types, function_invoke_type_vars,
                                            compiled_functions_);
    function_compiler.GenerateCPPForFunction();
  }
}

}  // namespace vm
}  // namespace relay
}  // namespace tvm
