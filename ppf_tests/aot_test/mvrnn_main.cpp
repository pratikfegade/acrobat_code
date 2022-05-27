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

#include "array_utils.hpp"

#if MODEL_SIZE == 1
#include "mvrnn_small_src.hpp"
#elif MODEL_SIZE == 2
#include "mvrnn_large_src.hpp"
#else
#error "Unsupported model size"
#endif

using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::runtime::vm;

DLDevice dev{kDLCPU, 0};
DLDevice gpu_dev{kDLCUDA, 0};
DLDataType dtype{kDLFloat, 32, 1};

#if MODEL_SIZE == 1
constexpr int hsize = 64;
#elif MODEL_SIZE == 2
constexpr int hsize = 128;
#else
#error "Unsupported model size"
#endif

using TreePtr = std::shared_ptr<MVTree>;

std::vector<std::string> tokenize_sst_sexpr(const std::string& s) {
  std::regex tokker(" +|[()]|[^ ()]+");
  std::vector<std::string> toks;
  for (auto it = std::sregex_iterator(s.begin(), s.end(), tokker); it != std::sregex_iterator();
       ++it) {
    std::string m = it->str();
    if (m != " ") toks.push_back(m);
  }
  return toks;
}

template <typename TensorType>
TreePtr within_bracket(std::vector<std::string>::const_iterator& tokit) {
  const std::string& label = *(tokit++);
  std::vector<TreePtr> children;
  while (true) {
    const std::string& tok = *(tokit++);
    if (tok == "(") {
      children.push_back(within_bracket<TensorType>(tokit));
    } else if (tok == ")") {
      if (children.size() <= 1) {
        auto leaf = std::static_pointer_cast<MVTree>(std::make_shared<LeafNode>());
        leaf->tag = MVTREE_LEAFNODE_TAG;
        static_cast<LeafNode*>(leaf.get())->field_0 =
            GetFillFloat32Tensor<TensorType>({hsize, hsize}, gpu_dev);
        static_cast<LeafNode*>(leaf.get())->field_1 =
            GetFillFloat32Tensor<TensorType>({1, hsize}, gpu_dev);
        return leaf;
      } else {
        auto inner = std::static_pointer_cast<MVTree>(std::make_shared<InnerNode>());
        inner->tag = MVTREE_INNERNODE_TAG;
        static_cast<InnerNode*>(inner.get())->field_0 = children[0];
        static_cast<InnerNode*>(inner.get())->field_1 = children[1];
        assert(children[0]);
        assert(children[1]);
        return inner;
      }
    } else {
      children.push_back(nullptr);
    }
  }
  throw std::runtime_error("Poorly structured tree");
}

template <typename TensorType>
TreePtr from_sst_sexpr(const std::string& str) {
  std::vector<std::string> toks = tokenize_sst_sexpr(str);
  std::vector<std::string>::const_iterator tokit = toks.begin();
  if (*(tokit++) != "(") throw std::runtime_error("Poorly structured tree");
  return within_bracket<TensorType>(tokit);
}

template <typename TensorType>
std::vector<TreePtr> parse_trees(std::vector<std::string> lines, int start, int size) {
  std::vector<TreePtr> ret;
  for (int i = start; i < start + size; ++i) {
    ret.push_back(from_sst_sexpr<TensorType>(lines[i]));
  }
  return ret;
}

std::vector<std::string> read_sst_dataset(const std::string& filename) {
  std::ifstream file(filename);
  if (!file) throw std::runtime_error("Missing file");
  std::string line;
  std::vector<std::string> ret;
  while (getline(file, line)) ret.push_back(line);
  return ret;
}

template <typename TensorType>
TreePtr complete_tree(int height) {
  if (height == 1) {
    auto leaf = std::static_pointer_cast<MVTree>(std::make_shared<LeafNode>());
    leaf->tag = MVTREE_LEAFNODE_TAG;
    static_cast<LeafNode*>(leaf.get())->field_0 =
        GetFillFloat32Tensor<TensorType>({hsize, hsize}, gpu_dev);

    static_cast<LeafNode*>(leaf.get())->field_1 =
        GetFillFloat32Tensor<TensorType>({1, hsize}, gpu_dev);
    return leaf;
  } else {
    auto inner = std::static_pointer_cast<MVTree>(std::make_shared<InnerNode>());
    inner->tag = MVTREE_INNERNODE_TAG;
    static_cast<InnerNode*>(inner.get())->field_0 = complete_tree<TensorType>(height - 1);
    static_cast<InnerNode*>(inner.get())->field_1 = complete_tree<TensorType>(height - 1);
    return inner;
  }
}

template <typename TensorType>
std::vector<TreePtr> complete_trees(int height, int number) {
  std::vector<TreePtr> ret;
  for (int i = 0; i < number; ++i) {
    ret.push_back(complete_tree<TensorType>(height));
  }
  return ret;
}

#if EXECUTOR_TYPE == 1
using ExecutorType = LazyExecutor<DLTensor*>;
#elif EXECUTOR_TYPE == 2
using ExecutorType = DepthTrackingExecutor;
#else
#error "Unsupported executor"
#endif

template <typename TensorType>
void invoke_model(std::vector<Device> devices, int argc, char* argv[]) {
  std::string tree_mode = argv[0];
  int batch_size = atoi(argv[1]);
  int height = 6;
  int num_batches = 1;
  std::vector<std::string> lines;
  if (tree_mode == "complete") {
    height = atoi(argv[2]);
  } else {
    std::string data_file = tree_mode;
    num_batches = atoi(argv[2]);
    lines = read_sst_dataset(data_file);
    num_batches = std::min((int)(lines.size() / batch_size), num_batches);
  }
  bool profile = false;
  bool debug = true;

  DeviceAPI::Get(gpu_dev)->StreamSync(gpu_dev, nullptr);
  if (debug) {
    std::vector<TreePtr> trees;
    if (tree_mode == "complete") {
      trees = complete_trees<TensorType>(height, batch_size);
    } else {
      trees = parse_trees<TensorType>(lines, 0, batch_size);
    }
    auto res = batched_main(trees);
    DynBatchRuntime<ExecutorType, TensorType>::Current()->LazyExecute();

    for (auto p : res) {
      std::cout << GetTensorMean({hsize, hsize}, std::get<0>(*p)) << " "
                << GetTensorMean({1, hsize}, std::get<1>(*p)) << std::endl;
    }
  } else {
    if (profile) {
      VMDBProfiler::Init({dev, gpu_dev});
    }

    float all_gen_time_ms = 0.f;
    float all_exe_time_ms = 0.f;
    std::vector<TreePtr> trees;
    for (size_t j1 = 0; j1 < num_batches; ++j1) {
      if (tree_mode == "complete") {
        trees = complete_trees<TensorType>(height, batch_size);
      } else {
        trees = parse_trees<TensorType>(lines, j1 * batch_size, batch_size);
      }

      auto runner = [&]() {
        DynBatchRuntime<ExecutorType, TensorType>::Current()->RecycleAllArenaMemory();

        auto start = std::chrono::system_clock::now();

        if (VMDBProfiler::DoProfile()) {
          VMDBProfiler::ProfileHostStartCall("graph_construction");
        }
        batched_main(trees);
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
