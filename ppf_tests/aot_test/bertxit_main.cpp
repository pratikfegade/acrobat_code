#include <tvm/runtime/module.h>
#include <tvm/runtime/vm/db_runtime.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/runtime/vm/vm_profiling.h>

#include <chrono>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <vector>

#include "bertxit_src.hpp"

using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::runtime::vm;

tvm::runtime::vm::FiberRuntime* tvm::runtime::vm::FiberRuntime::instance_{nullptr};

DLDevice dev{kDLCPU, 0};
DLDevice gpu_dev{kDLCUDA, 0};
DLDataType dtype{kDLFloat, 32, 1};

constexpr int seq_len = 128;
constexpr int model_size = 512;

template <typename TensorType>
TensorType GetRandomTensor(std::initializer_list<int64_t> shape, DLDevice device,
                           DLDataType dtype) {}

template <>
NDArray GetRandomTensor<NDArray>(std::initializer_list<int64_t> shape, DLDevice device,
                                 DLDataType dtype) {
  return NDArray::Empty(shape, dtype, device);
}

template <>
DLTensor* GetRandomTensor<DLTensor*>(std::initializer_list<int64_t> shape, DLDevice device,
                                     DLDataType dtype) {
  static std::vector<NDArray> arrays;
  auto array = GetRandomTensor<NDArray>(shape, device, dtype);
  arrays.push_back(array);
  return const_cast<DLTensor*>(array.operator->());
}

template <class TensorType>
std::vector<TensorType> create_vector(int length) {
  std::vector<TensorType> res;
  for (size_t i = 0; i < length; ++i) {
    res.push_back(GetRandomTensor<TensorType>({seq_len, model_size}, gpu_dev, dtype));
  }
  return res;
}

using ExecutorType = DepthTrackingExecutor;
// using ExecutorType = LazyExecutor<DLTensor*>;

template <typename TensorType>
void invoke_model(std::vector<Device> devices, int argc, char* argv[]) {
  int batch_size = atoi(argv[0]);
  int num_batches = 1;
  bool profile = false;
  bool debug = true;

  std::vector<TensorType> inputs = create_vector<TensorType>(batch_size);

  DeviceAPI::Get(gpu_dev)->StreamSync(gpu_dev, nullptr);
  if (debug) {
    batched_main(inputs);
    DynBatchRuntime<ExecutorType, TensorType>::Current()->LazyExecute();
  } else {
    if (profile) {
      VMDBProfiler::Init({dev, gpu_dev});
    }

    float all_gen_time_ms = 0.f;
    float all_exe_time_ms = 0.f;
    for (size_t j1 = 0; j1 < num_batches; ++j1) {
      auto runner = [&]() {
        DynBatchRuntime<ExecutorType, TensorType>::Current()->RecycleAllArenaMemory();

        auto start = std::chrono::system_clock::now();

        batched_main(inputs);

        auto mid = std::chrono::system_clock::now();

        DynBatchRuntime<ExecutorType, TensorType>::Current()->LazyExecute();

        auto end = std::chrono::system_clock::now();

        auto gen_fs = (mid - start);
        auto exe_fs = (end - mid);
        return std::make_pair(
            std::chrono::duration_cast<std::chrono::microseconds>(gen_fs).count(),
            std::chrono::duration_cast<std::chrono::microseconds>(exe_fs).count());
      };
      auto times = measure_time(runner, VMDBProfiler::DoProfile());

      float gen_time_ms = times.first / 1000.0;
      float exe_time_ms = times.second / 1000.0;
      all_gen_time_ms += gen_time_ms;
      all_exe_time_ms += exe_time_ms;
    }
    all_gen_time_ms /= num_batches;
    all_exe_time_ms /= num_batches;
    if (profile) {
      std::cout << VMDBProfiler::GetReport(100) << std::endl;
    }
    std::cout << "RESULTS," << all_gen_time_ms << "," << all_exe_time_ms << ","
              << (all_exe_time_ms + all_gen_time_ms) << std::endl;
  }
}

// template void invoke_model<NDArray>(std::vector<Device> devices);
template void invoke_model<DLTensor*>(std::vector<Device> devices, int argc, char* argv[]);
