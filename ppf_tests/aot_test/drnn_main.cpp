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

#include "drnn_src.hpp"

using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::runtime::vm;

tvm::runtime::vm::FiberRuntime* tvm::runtime::vm::FiberRuntime::instance_{nullptr};

DLDevice dev{kDLCPU, 0};
DLDevice gpu_dev{kDLCUDA, 0};
DLDataType dtype{kDLFloat, 32, 1};

constexpr int hsize = 256;

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
    res.push_back(GetRandomTensor<TensorType>({1, hsize}, gpu_dev, dtype));
  }
  return res;
}

using ExecutorType = DepthTrackingExecutor;
// using ExecutorType = LazyExecutor<DLTensor*>;

template <typename TensorType>
std::pair<int, int> tree_stats(
    std::shared_ptr<Tree<std::shared_ptr<std::tuple<DLTensor*, DLTensor*>>>> tree_base) {
  auto tree =
      static_cast<Rose<std::shared_ptr<std::tuple<DLTensor*, DLTensor*>>>*>(tree_base.get());
  auto children_base = tree->field_1;

  int depth = 0;
  int nodes = 1;
  while (true) {
    if (children_base->tag == LIST_NIL_TAG) {
      break;
    }
    auto children = static_cast<
        Cons<std::shared_ptr<Tree<std::shared_ptr<std::tuple<DLTensor*, DLTensor*>>>>>*>(
        children_base.get());
    children_base = children->field_1;

    auto child_res = tree_stats<TensorType>(children->field_0);
    nodes += child_res.first;
    depth = 1 + std::max(depth, child_res.second);
  }
  return std::make_pair(nodes, depth);
}

template <typename TensorType>
void invoke_model(std::vector<Device> devices, int argc, char* argv[]) {
  int batch_size = atoi(argv[0]);
  int num_batches = 1;
  bool profile = true;
  bool debug = false;

  std::vector<TensorType> input1 = create_vector<TensorType>(batch_size);
  std::vector<TensorType> input2 = create_vector<TensorType>(batch_size);

  DeviceAPI::Get(gpu_dev)->StreamSync(gpu_dev, nullptr);
  if (debug) {
    auto res = batched_main(input1, input2);
    DynBatchRuntime<ExecutorType, TensorType>::Current()->LazyExecute();
    for (auto tree : res) {
      auto stats = tree_stats<TensorType>(tree);
      std::cout << "STATS " << stats.first << " " << stats.second << std::endl;
    }
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

        if (VMDBProfiler::DoProfile()) {
          VMDBProfiler::ProfileHostStartCall("graph_construction");
        }
        batched_main(input1, input2);
        if (VMDBProfiler::DoProfile()) {
          VMDBProfiler::ProfileHostStopCall();
        }

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
