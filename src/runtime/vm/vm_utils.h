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
 * \file src/runtime/vm/vm.cc
 * \brief The Relay virtual machine runtime.
 */

#include <dmlc/memory_io.h>
#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/vm/lazy_executor.h>
#include <tvm/runtime/vm/vm.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../file_utils.h"

using namespace tvm::runtime;

namespace tvm {
namespace runtime {
namespace vm {

/* Other utils */

/*!
 * \brief Convert NDArray to vector
 *
 * \param shape_tensor The source NDArray.
 *
 * \return The vector.
 */
std::vector<int64_t> ToShape(NDArray shape_tensor);

/*!
 * \brief Copy objects (arrays or lists of arrays) across devices
 *
 * \param src The source object.
 * \param dev The destinationd device.
 *
 * \return The copied object.
 */
ObjectRef CopyTo(ObjectRef src, const DLDevice& dev);

/* Invoking packed functions */

/*!
 * \brief Invoke a PackedFunction (refactored out to avoid code
 * duplication). This functions assumes that all ADT args have already
 * been unrolled into their constituent NDArrays
 *
 * \param func The PackedFunction to be invoked.
 * \param arg_count The number of arguments to the PackedFunction.
 * \param output_size The number of outputs of the PackedFunction.
 * \param args Arguments to the PackedFunction.
 *
 * \note The return value will be stored in the last output_size slots of args.
 */
void InvokePackedFnUnrolled(const size_t func_idx, const PackedFunc& func, Index output_size,
                            const NDArray* args, int num_args);

/*!
 * \brief Invoke a batch PackedFunction (refactored out to avoid code
 * duplication) on a batch of OpNodes . This functions assumes that
 * all ADT args have already been unrolled into their constituent
 * NDArrays
 *
 * \param func The PackedFunction to be invoked.
 * \param arg_count The number of arguments to the PackedFunction.
 * \param output_size The number of outputs of the PackedFunction.
 * \param arg_modes The argument modes of the PackedFunction.
 * \param nodes The OpNodes as a batch.
 *
 * \note The return value will be stored in the last output_size slots of args.
 */
void InvokePackedFnBatchedUnrolled(const size_t func_idx, const PackedFunc& func, Index arg_count,
                                   Index output_size,
                                   const std::vector<DBBatchedArgMode>& arg_modes,
                                   const std::vector<OpNode*>& nodes);

/*!
 * \brief Invoke a PackedFunction (refactored out to avoid code duplication)
 *
 * \param func The PackedFunction to be invoked.
 * \param arg_count The number of arguments to the PackedFunction.
 * \param output_size The number of outputs of the PackedFunction.
 * \param args Arguments to the PackedFunction.
 * \param num_args number of arguments to the PackedFunction.
 *
 * \note The return value will be stored in the last output_size slots of args.
 */
void InvokePackedFn(const PackedFunc& func, Index arg_count, Index output_size,
                    const ObjectRef* args, int64_t num_args,
                    const std::vector<DBBatchedArgMode>& arg_modes, bool batched = false,
                    bool scattered_kernels = false);
}  // namespace vm
}  // namespace runtime
}  // namespace tvm
