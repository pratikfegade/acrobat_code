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

#include "treelstm_src.hpp"

using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::runtime::vm;

DLDevice dev{kDLCPU, 0};
DLDevice gpu_dev{kDLCUDA, 0};
DLDataType dtype{kDLFloat, 32, 1};

constexpr int hsize = 256;
constexpr int bsize = 8;
constexpr int theight = 6;

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
std::shared_ptr<Tree<TensorType>> within_bracket(std::vector<std::string>::const_iterator& tokit) {
  const std::string& label = *(tokit++);
  std::vector<std::shared_ptr<Tree<TensorType>>> children;
  while (true) {
    const std::string& tok = *(tokit++);
    if (tok == "(") {
      children.push_back(within_bracket<TensorType>(tokit));
    } else if (tok == ")") {
      auto ret = std::static_pointer_cast<Tree<TensorType>>(std::make_shared<Rose<TensorType>>());
      ret->tag = TREE_ROSE_TAG;
      static_cast<Rose<TensorType>*>(ret.get())->field_0 =
          GetRandomTensor<TensorType>({1, hsize}, gpu_dev, dtype);

      auto nil_node = std::static_pointer_cast<List<std::shared_ptr<Tree<TensorType>>>>(
          std::make_shared<Nil<std::shared_ptr<Tree<TensorType>>>>());
      nil_node->tag = LIST_NIL_TAG;
      auto new_list_head = nil_node;
      auto new_list_tail = nil_node;

      std::cout << "[NC] " << children.size() << std::endl;
      for (size_t i = 0; i < children.size(); ++i) {
        auto new_node = std::static_pointer_cast<List<std::shared_ptr<Tree<TensorType>>>>(
            std::make_shared<Cons<std::shared_ptr<Tree<TensorType>>>>());
        new_node->tag = LIST_CONS_TAG;
        static_cast<Cons<std::shared_ptr<Tree<TensorType>>>*>(new_node.get())->field_0 =
            children[i];
        if (new_list_tail->tag != LIST_NIL_TAG) {
          static_cast<Cons<std::shared_ptr<Tree<TensorType>>>*>(new_list_tail.get())->field_1 =
              new_node;
        } else {
          new_list_head = new_node;
        }
        static_cast<Cons<std::shared_ptr<Tree<TensorType>>>*>(new_node.get())->field_1 = nil_node;
        new_list_tail = new_node;
      }

      static_cast<Rose<TensorType>*>(ret.get())->field_1 = new_list_head;
      return ret;
    } else {
      auto ret = std::static_pointer_cast<Tree<TensorType>>(std::make_shared<Rose<TensorType>>());
      ret->tag = TREE_ROSE_TAG;
      static_cast<Rose<TensorType>*>(ret.get())->field_0 =
          GetRandomTensor<TensorType>({1, hsize}, gpu_dev, dtype);
      auto nil = std::static_pointer_cast<List<std::shared_ptr<Tree<TensorType>>>>(
          std::make_shared<Nil<std::shared_ptr<Tree<TensorType>>>>());
      nil->tag = LIST_NIL_TAG;
      static_cast<Rose<TensorType>*>(ret.get())->field_1 = nil;

      children.push_back(ret);
    }
  }
  throw std::runtime_error("Poorly structured tree");
}

template <typename TensorType>
std::shared_ptr<Tree<TensorType>> from_sst_sexpr(const std::string& str) {
  std::vector<std::string> toks = tokenize_sst_sexpr(str);
  std::vector<std::string>::const_iterator tokit = toks.begin();
  if (*(tokit++) != "(") throw std::runtime_error("Poorly structured tree");
  return within_bracket<TensorType>(tokit);
}

template <typename TensorType>
std::vector<std::shared_ptr<Tree<TensorType>>> parse_trees(std::vector<std::string> lines,
                                                           int start, int size) {
  std::vector<std::shared_ptr<Tree<TensorType>>> ret;
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
std::shared_ptr<Tree<TensorType>> complete_tree(int height) {
  auto ret = std::static_pointer_cast<Tree<TensorType>>(std::make_shared<Rose<TensorType>>());
  ret->tag = TREE_ROSE_TAG;
  static_cast<Rose<TensorType>*>(ret.get())->field_0 =
      GetRandomTensor<TensorType>({1, hsize}, gpu_dev, dtype);
  if (height == 1) {
    auto nil = std::static_pointer_cast<List<std::shared_ptr<Tree<TensorType>>>>(
        std::make_shared<Nil<std::shared_ptr<Tree<TensorType>>>>());
    nil->tag = LIST_NIL_TAG;
    static_cast<Rose<TensorType>*>(ret.get())->field_1 = nil;
  } else {
    auto node1 = std::static_pointer_cast<List<std::shared_ptr<Tree<TensorType>>>>(
        std::make_shared<Cons<std::shared_ptr<Tree<TensorType>>>>());
    auto node2 = std::static_pointer_cast<List<std::shared_ptr<Tree<TensorType>>>>(
        std::make_shared<Cons<std::shared_ptr<Tree<TensorType>>>>());
    auto tail = std::static_pointer_cast<List<std::shared_ptr<Tree<TensorType>>>>(
        std::make_shared<Nil<std::shared_ptr<Tree<TensorType>>>>());
    node1->tag = LIST_CONS_TAG;
    node2->tag = LIST_CONS_TAG;
    tail->tag = LIST_NIL_TAG;

    static_cast<Cons<std::shared_ptr<Tree<TensorType>>>*>(node1.get())->field_0 =
        complete_tree<TensorType>(height - 1);
    static_cast<Cons<std::shared_ptr<Tree<TensorType>>>*>(node2.get())->field_0 =
        complete_tree<TensorType>(height - 1);
    static_cast<Cons<std::shared_ptr<Tree<TensorType>>>*>(node1.get())->field_1 = node2;
    static_cast<Cons<std::shared_ptr<Tree<TensorType>>>*>(node2.get())->field_1 = tail;
    static_cast<Rose<TensorType>*>(ret.get())->field_1 = node1;
  }
  return ret;
}

template <typename TensorType>
std::vector<std::shared_ptr<Tree<TensorType>>> complete_trees(int height, int number) {
  std::vector<std::shared_ptr<Tree<TensorType>>> ret;
  for (int i = 0; i < number; ++i) {
    ret.push_back(complete_tree<TensorType>(height));
  }
  return ret;
}

using ExecutorType = DepthTrackingExecutor;
// using ExecutorType = LazyExecutor<DLTensor*>;

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

  auto trees = complete_trees<TensorType>(theight, bsize);
  if (debug) {
    std::vector<std::shared_ptr<Tree<TensorType>>> trees;
    if (tree_mode == "complete") {
      trees = complete_trees<TensorType>(height, batch_size);
    } else {
      trees = parse_trees<TensorType>(lines, 0, batch_size);
    }
    batched_main(trees);
    // DynBatchRuntime<ExecutorType, TensorType>::Current()->LazyExecute();
  } else {
    if (profile) {
      VMDBProfiler::Init({dev, gpu_dev});
    }
    float all_gen_time_ms = 0.f;
    float all_exe_time_ms = 0.f;
    std::vector<std::shared_ptr<Tree<TensorType>>> trees;
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
