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

#ifndef TVM_RELAY_BACKEND_VM_COMPILER_H_
#define TVM_RELAY_BACKEND_VM_COMPILER_H_

#include "aot_compiler.h"

#include <tvm/ir/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/tir/function.h>

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace vm {

void VMAOTCompiler::HandleFunction(const VMFunction& vm_func, const Function& relay_func) {
  std::unordered_map<Index, std::string> targets;
  int target_count = 0;

  for (size_t i = 0; i < vm_func.instructions.size(); ++i) {
    auto& instr = vm_func.instructions[i];
    switch (instr.op) {
      case Opcode::Goto: {
        targets.insert({instr.pc_offset + i, "target" + std::to_string(target_count++)});
        break;
      }
      case Opcode::If: {
        targets.insert({instr.if_op.true_offset + i, , "target" + std::to_string(target_count++)});
        targets.insert({instr.if_op.false_offset + i, , "target" + std::to_string(target_count++)});
        break;
      }
    }
  }

  for (size_t i = 0; i < vm_func.instructions.size(); ++i) {
    auto& instr = vm_func.instructions[i];
    std::cout << instr << std::endl;

    auto it = targets.find(i);
    if (it != targets.end()) {
      this->PrintIndent(-1);
      os << it->second << ":\n";
    }

    switch (instr.op) {
      case Opcode::Move: {
        RegName src = instr.from;
        RegName dst = instr.dst;
        SrcVar src_var = GetVarForReg(src);
        SrcVar dst_var = GetVarForReg(dst);
        this->PrintIndent();
        os << dst_var << " = " << src_var << ";\n";
        break;
      }
      case Opcode::Fatal: {
        this->PrintIndent();
        os << "throw std::runtime_error(\"Fatal error\");\n";
        break;
      }
      case Opcode::LoadConst: {
      }
      case Opcode::LoadConsti:
      case Opcode::Invoke:
      case Opcode::InvokePacked:
      case Opcode::InvokeClosure:
      case Opcode::GetField:
      case Opcode::GetTag: {
        RegName object = instr.object;
        RegName dst = instr.dst;
        SrcVar object_var = GetVarForReg(object);
        SrcVar dst_var = GetVarForReg(dst);
        this->PrintIndent();
        os << dst_var << " = " << src_var << "->tag;\n";
        break;
      }
      case Opcode::Goto: {
        Index target = i + instr.pc_offset;
        auto label = targets.at(target);
        this->PrintIndent();
        os << "goto " << label << ";\n";
      }
      case Opcode::If: {
        RegName dst = instr.dst;
      }
      case Opcode::AllocTensor:
      case Opcode::AllocTensorReg:
      case Opcode::AllocADT:
      case Opcode::AllocClosure:
      case Opcode::AllocStorage:
      case Opcode::ShapeOf:
      case Opcode::Ret: {
        RegName result = instr.result;
        SrcVar result_var = GetVarForReg(result);
        this->PrintIndent();
        os << "return " << src_var << ";\n";
        break;
      }
      case Opcode::ReshapeTensor:
      case Opcode::DeviceCopy:
      default:
        LOG(FATAL) << "Unknown instruction opcode: " << int(instr.op);
        return false;
    }
  }
}

void VMAOTCompiler::GenerateCPP() {
  for (auto vm_func : exec_.functions) {
    std::cout << "[AOT] Function " << vm_func.name << " " << vm_func.register_file_size
              << std::endl;
    auto relay_func = mod_->Lookup(vm_func.name);
    HandleFunction(function);
    std::cout << std::endl;
  }
}

}  // namespace vm
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_VM_COMPILER_H_
