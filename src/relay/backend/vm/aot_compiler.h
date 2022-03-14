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

  inline void PrintIndent(std::ostream& stream, int offset = 0) {
    for (int i = 0; i < indent_ + offset; ++i) {
      stream << ' ';
    }
  }

  inline std::string GetVarForReg(RegName reg) { return "local_" + std::to_string(reg); }

  inline std::string GetFieldName(size_t index) { return "field_" + std::to_string(index); }

  inline std::string GetTmpVarName(size_t index) { return "tmp_" + std::to_string(index); }

  int indent_ = 0;
};

class VMAOTCompiler : SourcePrinter {
 public:
  VMAOTCompiler(
      const Executable& exec, const IRModule& mod,
      const std::unordered_map<std::string, std::unordered_map<size_t, Type>>& register_types,
      const std::unordered_map<std::string, std::unordered_map<Index, Array<Type>>>&
          invoke_type_vars,
      const std::unordered_map<std::string, Function>& compiled_functions,
      const std::unordered_map<std::string, std::unordered_map<Index, int32_t>>& get_field_tags,
      const std::string& output_directory, const std::string& model_name)
      : exec_(exec),
        mod_(mod),
        register_types_(register_types),
        invoke_type_vars_(invoke_type_vars),
        compiled_functions_(compiled_functions),
        get_field_tags_(get_field_tags),
        output_directory_(output_directory),
        model_name_(model_name) {}

  void Codegen();

 private:
  void DeclareADT(std::ostream& os, const TypeData& adt, bool include_definitions);

  void EmitUtilFunctions(std::ostream& os);

  void EmitBatchedMainFunction(std::ostream& os);

  void EmitBatchedMainFunctionHeader(std::ostream& os);

  void EmitHarnessFunctionHeaders(std::ostream& os);

  void EmitHarnessFunctions(std::ostream& os);

  void GenerateCppFile(std::string header_file_name);

  void EmitMacros(std::ostream& os);

  void EmitHeaderIncludes(std::ostream& os);

  void GenerateHeaderFile(std::string header_file_name);

  const Executable& exec_;
  const IRModule& mod_;
  const std::unordered_map<std::string, std::unordered_map<size_t, Type>>& register_types_;
  const std::unordered_map<std::string, std::unordered_map<Index, Array<Type>>>& invoke_type_vars_;
  const std::unordered_map<std::string, Function>& compiled_functions_;
  const std::unordered_map<std::string, std::unordered_map<Index, int32_t>>& get_field_tags_;
  const std::string& output_directory_;
  const std::string& model_name_;
  std::stringstream hpp_stream_;
  std::stringstream cpp_stream_;
};

}  // namespace vm
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_VM_AOT_COMPILER_H_
