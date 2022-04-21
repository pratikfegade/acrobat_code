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

#include "graph_gen_src.hpp"

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
std::shared_ptr<List<TensorType>> create_list(int length) {
  if (length == 0) {
    auto nil_node = std::static_pointer_cast<List<TensorType>>(std::make_shared<Nil<TensorType>>());
    nil_node->tag = LIST_NIL_TAG;
    return nil_node;
  } else {
    auto tail = create_list<TensorType>(length - 1);
    auto new_node =
        std::static_pointer_cast<List<TensorType>>(std::make_shared<Cons<TensorType>>());
    new_node->tag = LIST_CONS_TAG;
    static_cast<Cons<TensorType>*>(new_node.get())->field_0 =
        GetRandomTensor<TensorType>({1, hsize}, gpu_dev, dtype);
    static_cast<Cons<TensorType>*>(new_node.get())->field_1 = tail;
    return new_node;
  }
}

using ExecutorType = DepthTrackingExecutor;
// using ExecutorType = LazyExecutor<DLTensor*>;

template <typename TensorType>
void invoke_model(std::vector<Device> devices, int argc, char* argv[]) {
  int batch_size = atoi(argv[0]);
  int nodes_lo = atoi(argv[1]);
  int nodes_hi = atoi(argv[2]);
  int edges_lo = atoi(argv[3]);
  int edges_hi = atoi(argv[4]);
  int num_batches = 1;
  bool profile = false;
  bool debug = true;

  std::vector<std::shared_ptr<List<TensorType>>> inits;
  for (int i = 0; i < batch_size; ++i) {
    inits.push_back(create_list<TensorType>(GetRandom(nodes_lo, nodes_hi)));
  }

  DeviceAPI::Get(gpu_dev)->StreamSync(gpu_dev, nullptr);
  if (debug) {
    batched_main(inits);
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

        if (VMDBProfiler::DoProfile()) {
          VMDBProfiler::ProfileHostStartCall("graph_construction");
        }
        batched_main(inits);
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
      std::cout << VMDBProfiler::GetReport() << std::endl;
    }
    std::cout << "RESULTS," << all_gen_time_ms << "," << all_exe_time_ms << ","
              << (all_exe_time_ms + all_gen_time_ms) << std::endl;
  }
}

// template void invoke_model<NDArray>(std::vector<Device> devices);
template void invoke_model<DLTensor*>(std::vector<Device> devices, int argc, char* argv[]);
