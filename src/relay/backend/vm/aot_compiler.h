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

#ifndef TVM_RELAY_BACKEND_VM_AOT_COMPILER_H_
#define TVM_RELAY_BACKEND_VM_AOT_COMPILER_H_

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

#include "../../../runtime/vm/naive_allocator.h"
#include "../../../runtime/vm/profiler/vm.h"
#include "../../transforms/pass_utils.h"
#include "../te_compiler.h"
#include "../te_compiler_cache.h"

namespace tvm {
namespace relay {
namespace vm {

using namespace tvm::runtime;
using namespace tvm::runtime::vm;
using namespace relay::transform;

class SourcePrinter {
 protected:
  inline void BeginScope() { indent_ += 2; }

  inline void EndScope() { indent_ -= 2; }

  inline void PrintIndent(int offset = 0) {
    for (int i = 0; i < indent_ + offset; ++i) {
      this->stream_ << ' ';
    }
  }

  inline std::string GetVarForReg(RegName reg) { return "local_" + std::to_string(reg); }

  inline std::string GetFieldName(size_t index) { return "field_" + std::to_string(index); }

  inline std::string GetTmpVarName(size_t index) { return "tmp_" + std::to_string(index); }

  int indent_ = 0;
  std::stringstream stream_;
};

class VMAOTCompiler : SourcePrinter {
 public:
  VMAOTCompiler(
      const Executable& exec, const IRModule& mod,
      std::unordered_map<std::string, std::unordered_map<size_t, Type>>& register_types,
      std::unordered_map<std::string, std::unordered_map<Index, Array<Type>>> invoke_type_vars,
      std::unordered_map<std::string, Function> compiled_functions)
      : exec_(exec),
        mod_(mod),
        register_types_(register_types),
        invoke_type_vars_(invoke_type_vars),
        compiled_functions_(compiled_functions) {}

  void GenerateCPP();

 private:
  void DeclareADT(const TypeData& adt);

  void EmitHeader();

  void GenerateMainFunction();

  const Executable& exec_;
  const IRModule& mod_;
  const std::unordered_map<std::string, std::unordered_map<size_t, Type>>& register_types_;
  const std::unordered_map<std::string, std::unordered_map<Index, Array<Type>>>& invoke_type_vars_;
  const std::unordered_map<std::string, Function>& compiled_functions_;
};

}  // namespace vm
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_VM_AOT_COMPILER_H_
