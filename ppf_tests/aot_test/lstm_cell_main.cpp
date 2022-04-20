#include <tvm/runtime/module.h>
#include <tvm/runtime/vm/db_runtime.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/runtime/vm/vm_profiling.h>

#include "lstm_cell_src.hpp"

using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::runtime::vm;

DLDevice dev{kDLCPU, 0};
DLDevice gpu_dev{kDLCUDA, 0};
DLDataType dtype{kDLFloat, 32, 1};

constexpr int hsize = 32;
constexpr int bsize = 8;

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

template <typename TensorType>
std::vector<TensorType> input_vectors(int number) {
  std::vector<TensorType> ret;
  for (int i = 0; i < number; ++i) {
    ret.push_back(GetRandomTensor<TensorType>({1, hsize}, gpu_dev, dtype));
  }
  return ret;
}

template <typename TensorType>
void invoke_model(std::vector<Device> devices) {
  auto inputs = input_vectors<TensorType>(bsize);
  batched_main(inputs);
  DynBatchRuntime<DepthTrackingExecutor, TensorType>::Current()->LazyExecute();
  // auto runner = [&]() {
  // Arena::Current()->RecycleAll();
  // auto start = std::chrono::system_clock::now();

  // batched_main(inputs);

  // auto mid = std::chrono::system_clock::now();

  // DynBatchRuntime<DepthTrackingExecutor, TensorType>::Current()->LazyExecute();

  // auto end = std::chrono::system_clock::now();
  // auto gen_fs = (mid - start);
  // auto exe_fs = (end - mid);
  // return std::make_pair(std::chrono::duration_cast<std::chrono::microseconds>(gen_fs).count(),
  // std::chrono::duration_cast<std::chrono::microseconds>(exe_fs).count());
  // };
  // auto times = measure_time(runner, false);
  // float gen_time_ms = times.first / 1000.0;
  // float exe_time_ms = times.second / 1000.0;
  // std::cout << "RESULTS," << gen_time_ms << "," << exe_time_ms << "," << (exe_time_ms +
  // gen_time_ms)
  // << std::endl;
}

// template void invoke_model<NDArray>(std::vector<Device> devices);
template void invoke_model<DLTensor*>(std::vector<Device> devices);
