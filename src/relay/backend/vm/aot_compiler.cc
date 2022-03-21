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

void RelayTypeToCppStr(std::ostream& os, const Type& type, bool no_shared_ptr = false,
                       const std::string& replacement = "") {
  if (type.as<TensorTypeNode>()) {
    os << "NDArray";
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
                                    const std::string& replacement = "") {
  std::stringstream ss;
  RelayTypeToCppStr(ss, type, no_shared_ptr, replacement);
  return ss.str();
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
                        const std::unordered_map<std::string, Function>& compiled_functions,
                        const std::unordered_map<Index, int32_t>& get_field_tags,
                        std::ostream& stream)
      : exec_(exec),
        mod_(mod),
        vm_func_(vm_func),
        relay_func_(relay_func),
        register_types_(register_types),
        invoke_type_vars_(invoke_type_vars),
        compiled_functions_(compiled_functions),
        get_field_tags_(get_field_tags),
        stream_(stream) {}

  void GenerateCPPForFunction(bool definition) {
    // std::cout << "[FUN] Visiting " << vm_func_.name << std::endl;
    Type function_type = relay_func_->checked_type_;
    CreateFunctionDeclaration(vm_func_, relay_func_);

    if (definition) {
      stream_ << " {\n";
      this->BeginScope();

      this->VisitBytecode();

      this->EndScope();
      stream_ << "}\n";
    } else {
      stream_ << ";\n";
    }
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
      if (arg_type.as<TensorTypeNode>()) {
        stream_ << "&";
      }
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

  void GenerateLocalDecls(const std::vector<bool>& used_regs) {
    std::unordered_map<std::string, std::vector<std::string>> type2vars;
    for (int i = vm_func_.params.size(); i < vm_func_.register_file_size; ++i) {
      if (!used_regs[i]) {
        continue;
      }
      auto it = register_types_.find(i);
      ICHECK(it != register_types_.end()) << i;
      Type reg_type = it->second;
      if (reg_type.as<TensorTypeNode>() || reg_type.as<PrimTypeNode>() || IsStorageType(reg_type)) {
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

    for (auto kv : type2vars) {
      this->PrintIndent(stream_);
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
    std::vector<bool> used_regs(vm_func_.register_file_size, false);
    for (size_t i = 0; i < vm_func_.instructions.size(); ++i) {
      auto& instr = vm_func_.instructions[i];
      for (auto reg : Instruction::ReadRegisters(instr)) {
        used_regs[reg] = true;
      }
    }

    this->GenerateLocalDecls(used_regs);

    // std::cout << "[BT] Visiting BT" << std::endl;
    std::unordered_map<Index, std::string> targets;
    int target_count = 0;

    // std::cout << "[BT]  Visiting targets" << std::endl;
    for (size_t i = 0; i < vm_func_.instructions.size(); ++i) {
      auto& instr = vm_func_.instructions[i];
      if (instr.op == Opcode::Goto) {
        targets[instr.pc_offset + i] = "target" + std::to_string(target_count++);
      } else if (instr.op == Opcode::If) {
        targets[instr.if_op.true_offset + i] = "target" + std::to_string(target_count++);
        targets[instr.if_op.false_offset + i] = "target" + std::to_string(target_count++);
      }
    }

    // std::cout << "[BT]  Visiting code" << std::endl;
    int tmp_var_counter = 0;
    for (size_t i = 0; i < vm_func_.instructions.size(); ++i) {
      auto& instr = vm_func_.instructions[i];
      // std::cout << "[BT]   " << instr << std::endl;

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
          stream_ << dst_var << " = DynBatchRuntime::Current()->GetConstant(" << instr.const_index
                  << ");\n";
          break;
        }
        case Opcode::LoadConsti: {
          auto dst_var = GetVarForReg(instr.dst);
          this->PrintIndent(stream_);
          stream_ << dst_var << " = " << instr.load_consti.val << ";\n";
          break;
        }
        case Opcode::Invoke: {
          auto dst_var = GetVarForReg(instr.dst);
          auto callee_name = exec_.functions[instr.func_index].name;
          this->PrintIndent(stream_);
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
          for (int i = 0; i < instr.num_args; ++i) {
            stream_ << GetVarForReg(instr.invoke_args_registers[i]);
            if (i < instr.num_args - 1) {
              stream_ << ",";
            }
          }
          stream_ << ");\n";
          break;
        }
        case Opcode::InvokePacked: {
          this->PrintIndent(stream_);
          auto args_vec = "args_tmp" + std::to_string(tmp_var_counter++);
          std::vector<std::string> flattened_args;
          for (int j = 0; j < instr.arity; ++j) {
            if (auto tt = register_types_.at(instr.packed_args[j]).as<TupleTypeNode>()) {
              auto tuple_var = GetVarForReg(instr.packed_args[j]);
              for (size_t k = 0; k < tt->fields.size(); ++k) {
                flattened_args.push_back("std::get<" + std::to_string(k) + ">(*" + tuple_var + ")");
              }
            } else {
              flattened_args.push_back(GetVarForReg(instr.packed_args[j]));
            }
          }
          stream_ << "std::vector<NDArray> " << args_vec << " = {";

          for (size_t j = 0; j < flattened_args.size(); ++j) {
            stream_ << flattened_args[j];
            if (j < flattened_args.size() - 1) {
              stream_ << ", ";
            }
          }

          stream_ << "};\n";
          this->PrintIndent(stream_);
          stream_ << "DynBatchRuntime::Current()->InvokePacked(" << instr.packed_index << ", "
                  << instr.arity << ", " << instr.output_size << ", " << args_vec << ".data(), "
                  << flattened_args.size() << ");\n";
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
              stream_ << ",";
            }
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
          break;
        }
        case Opcode::AllocTensor: {
          auto dst_var = GetVarForReg(instr.dst);
          auto storage_var = GetVarForReg(instr.alloc_tensor.storage);
          auto offset_var = GetVarForReg(instr.alloc_tensor.offset);
          std::string dtype_str = DTypeToStr(instr.alloc_tensor.dtype);

          std::stringstream shape_arr;
          shape_arr << "{";
          for (size_t j = 0; j < instr.alloc_tensor.ndim; ++j) {
            shape_arr << instr.alloc_tensor.shape[j];
            if (j < instr.alloc_tensor.ndim - 1) {
              shape_arr << ", ";
            }
          }
          shape_arr << "}";

          this->PrintIndent(stream_);
          std::string offset_var_str = offset_var;
          if (register_types_.at(instr.alloc_tensor.offset).as<TensorTypeNode>()) {
            offset_var_str = "NDToInt64(" + offset_var_str + ")";
          }
          stream_ << dst_var << " = " << storage_var << "->AllocNDArray(" << offset_var_str << ", "
                  << shape_arr.str() << ", " << dtype_str << ");\n";

          break;
        }
        case Opcode::AllocTensorReg: {
          auto dst_var = GetVarForReg(instr.dst);
          auto storage_var = GetVarForReg(instr.alloc_tensor_reg.storage);
          auto offset_var = GetVarForReg(instr.alloc_tensor.offset);
          std::string dtype_str = DTypeToStr(instr.alloc_tensor_reg.dtype);

          std::string shape_var = GetVarForReg(instr.alloc_tensor_reg.shape_register);
          this->PrintIndent(stream_);
          stream_ << dst_var << " = DynBatchRuntime::Current()->AllocateTensorReg(" << storage_var
                  << ", " << offset_var << ", " << shape_var << ", " << dtype_str << ");\n";
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
                    << ">>(new std::tuple<" << types_str.str() << ">(" << args_str.str() << "));\n";
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
          for (int j = instr.num_freevar; j < closure_func.params.size(); ++j) {
            Type arg_type = closure_relay_func->params[j]->checked_type_;
            RelayTypeToCppStr(stream_, arg_type);
            stream_ << " " << GetTmpVarName(j);
            if (j < closure_func.params.size() - 1) {
              stream_ << ", ";
            }
          }
          stream_ << ") {\n";
          this->BeginScope();
          this->PrintIndent(stream_);
          stream_ << "return " << closure_func.name << "(";
          int j = 0;
          for (; j < instr.num_freevar; ++j) {
            stream_ << GetVarForReg(instr.invoke_args_registers[j]);
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
          this->PrintIndent(stream_);
          stream_ << "};\n";
          break;
        }
        case Opcode::AllocStorage: {
          auto dst_var = GetVarForReg(instr.dst);
          auto allocation_size_var = GetVarForReg(instr.alloc_storage.allocation_size);
          auto dtype = instr.alloc_storage.dtype_hint;
          std::string dtype_str = "{" + std::to_string(dtype.code) + ", " +
                                  std::to_string(dtype.bits) + ", " + std::to_string(dtype.lanes) +
                                  "}";
          this->PrintIndent(stream_);
          std::string allocation_size_str = allocation_size_var;
          if (register_types_.at(instr.alloc_storage.allocation_size).as<TensorTypeNode>()) {
            allocation_size_var = "NDToInt64(" + allocation_size_var + ")";
          }
          stream_ << dst_var << " = DynBatchRuntime::Current()->AllocateStorage("
                  << allocation_size_str << ", " << instr.alloc_storage.alignment << ", "
                  << dtype_str << ", " << instr.alloc_storage.device_index << ");\n";
          break;
        }
        case Opcode::ShapeOf: {
          auto tensor_var = GetVarForReg(instr.reshape_tensor.tensor);
          auto dst_var = GetVarForReg(instr.dst);
          this->PrintIndent(stream_);
          stream_ << dst_var << " = DynBatchRuntime::Current()->ShapeOf(" << tensor_var << ");\n";
          break;
        }
        case Opcode::Ret: {
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
          stream_ << dst_var << " = DynBatchRuntime::Current()->ReshapeTensor(" << tensor_var
                  << ", " << shape_var << ");\n";
          break;
        }
        case Opcode::DeviceCopy: {
          auto src_var = GetVarForReg(instr.device_copy.src);
          auto dst_var = GetVarForReg(instr.dst);

          this->PrintIndent(stream_);
          stream_ << dst_var << " = DynBatchRuntime::Current()->DeviceCopy(" << src_var << ", "
                  << instr.device_copy.src_device_index << ", "
                  << instr.device_copy.dst_device_index << ");\n";
          break;
        }
        default:
          LOG(FATAL) << "Unknown instruction opcode: " << int(instr.op);
          return;
      }
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

void VMAOTCompiler::EmitUtilFunctions(std::ostream& os) {
  auto nd_to_int64_func =
      "int64_t NDToInt64(const NDArray& nd) {\n"
      "  static auto int64_dtype = DataType::Int(64);\n"
      "  DLDevice cpu_ctx{kDLCPU, 0};\n"
      "  NDArray cpu_array = nd.CopyTo(cpu_ctx);\n"
      "  CHECK_EQ(DataType(cpu_array->dtype), int64_dtype);\n"
      "  return reinterpret_cast<int64_t*>(cpu_array->data)[0];\n"
      "}";
  os << nd_to_int64_func << "\n" << std::endl;
}

void VMAOTCompiler::EmitBatchedMainFunction(std::ostream& os) {
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
  os << "res.push_back(" << GetCppFunctionName("main") << "(";
  for (size_t i = 0; i < function_type->arg_types.size(); ++i) {
    auto arg_type = function_type->arg_types[i];
    os << " " << GetVarForReg(i) << "[b]";
    if (i != function_type->arg_types.size() - 1) {
      os << ", ";
    }
  }
  os << "));\n";
  this->EndScope();
  this->PrintIndent(os);
  os << "}\n";
  auto pass_ctx = transform::PassContext::Current();
  bool lazy_execution = pass_ctx->GetConfig<Bool>("relay.db_lazy_execution", Bool(false)).value();
  if (lazy_execution) {
    this->PrintIndent(os);
    os << "DynBatchRuntime::Current()->LazyExecute();\n";
  }
  this->PrintIndent(os);
  os << "return res;\n";
  this->EndScope();
  this->PrintIndent(os);
  os << "}\n";
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
    if (arg_type.as<TensorTypeNode>()) {
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
  os << "}\n";

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
  os << "  auto runtime = DynBatchRuntime::CreateRuntime();\n";

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
  os << "  invoke_model(devices);\n";
  os << "}\n\n";
}

void VMAOTCompiler::EmitHarnessFunctionHeaders(std::ostream& os) {
  os << "void invoke_model(std::vector<Device> devices)\n;";
  os << "std::pair<float, float> measure_time(std::function<std::pair<float, float>()> runner, "
        "bool profiling = false);\n";
}

void VMAOTCompiler::GenerateCppFile(std::string header_file_name) {
  cpp_stream_ << "#include \"" << header_file_name << "\"\n\n";

  EmitUtilFunctions(cpp_stream_);

  for (auto vm_func : exec_.functions) {
    Function relay_func = GetCompiledRelayFunction(compiled_functions_, mod_, vm_func.name);
    ICHECK(register_types_.count(vm_func.name)) << vm_func.name;
    auto function_register_types = register_types_.at(vm_func.name);
    auto function_invoke_type_vars = invoke_type_vars_.at(vm_func.name);
    auto function_get_field_tags = get_field_tags_.at(vm_func.name);
    VMAOTFunctionCompiler function_compiler(
        exec_, mod_, vm_func, relay_func, function_register_types, function_invoke_type_vars,
        compiled_functions_, function_get_field_tags, cpp_stream_);
    function_compiler.GenerateCPPForFunction(true);
    cpp_stream_ << "\n";
  }

  EmitBatchedMainFunction(cpp_stream_);
  EmitHarnessFunctions(cpp_stream_);
}

void VMAOTCompiler::EmitMacros(std::ostream& os) {}

void VMAOTCompiler::EmitHeaderIncludes(std::ostream& os) {
  os << "#include <dlpack/dlpack.h>\n";
  os << "#include <tvm/runtime/vm/vm_profiling.h>\n";
  os << "#include <tvm/runtime/vm/db_runtime.h>\n";
  os << "#include <tvm/runtime/vm/arena.h>\n";
  os << "#include <stdexcept>\n";
  os << "#include <vector>\n";
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
    VMAOTFunctionCompiler function_compiler(
        exec_, mod_, vm_func, relay_func, function_register_types, function_invoke_type_vars,
        compiled_functions_, function_get_field_tags, hpp_stream_);
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
