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

#include "birnn_src.hpp"

using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::runtime::vm;

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

void read_lines(const std::string& filename, int num_lines, std::vector<int>* p_vec) {
  std::ifstream file(filename);
  if (!file) {
    throw std::runtime_error("Missing file");
  }
  std::string line;
  int ctr = 0;
  while (getline(file, line)) {
    p_vec->push_back(stoi(line));
    ctr++;
  }
  if (ctr < num_lines) {
    throw std::runtime_error("Insufficient lines in the file");
  }
}

using ExecutorType = DepthTrackingExecutor;
// using ExecutorType = LazyExecutor<DLTensor*>;

template <typename TensorType>
void invoke_model(std::vector<Device> devices, int argc, char* argv[]) {
  std::string dataset = argv[0];
  int batch_size = atoi(argv[1]);
  int num_batches = atoi(argv[2]);
  std::vector<int> lengths;
  if (dataset == "random") {
    int sent_lo = atoi(argv[3]);
    int sent_hi = atoi(argv[4]);
    for (int i = 0; i < batch_size * num_batches; ++i) {
      lengths.push_back(GetRandom(sent_lo, sent_hi));
    }
  } else {
    read_lines(dataset, batch_size * num_batches, &lengths);
  }
  bool profile = false;
  bool debug = false;

  std::vector<std::shared_ptr<List<TensorType>>> inits;
  DeviceAPI::Get(gpu_dev)->StreamSync(gpu_dev, nullptr);
  if (debug) {
    for (int i = 0; i < batch_size; ++i) {
      inits.push_back(create_list<TensorType>(lengths[i]));
    }
    batched_main(inits);
    DynBatchRuntime<ExecutorType, TensorType>::Current()->LazyExecute();
  } else {
    if (profile) {
      VMDBProfiler::Init({dev, gpu_dev});
    }

    float all_gen_time_ms = 0.f;
    float all_exe_time_ms = 0.f;
    for (size_t j1 = 0; j1 < num_batches; ++j1) {
      inits.clear();
      for (int j2 = 0; j2 < batch_size; ++j2) {
        inits.push_back(create_list<TensorType>(lengths[j1 * batch_size + j2]));
      }

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
      std::cout << VMDBProfiler::GetReport(dmlc::GetEnv("DB_WARM_UP_ITERATIONS", 1) +
                                           dmlc::GetEnv("DB_MEASURE_ITERATIONS", 1))
                << std::endl;
    }
    std::cout << "RESULTS," << all_gen_time_ms << "," << all_exe_time_ms << ","
              << (all_exe_time_ms + all_gen_time_ms) << std::endl;
  }
}

// template void invoke_model<NDArray>(std::vector<Device> devices);
template void invoke_model<DLTensor*>(std::vector<Device> devices, int argc, char* argv[]);
