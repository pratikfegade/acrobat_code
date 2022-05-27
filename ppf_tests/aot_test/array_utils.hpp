#include <tvm/runtime/ndarray.h>

#include <random>

using namespace tvm;
using namespace tvm::runtime;

int64_t GetTotalSize(const std::initializer_list<int64_t>& shape) {
  int64_t res = 1;
  for (auto d : shape) {
    res *= d;
  }
  return res;
}

float* CreateRandomFloat32HostArray(int64_t total_size) {
  std::random_device rd;
  std::mt19937 e2(rd());
  std::normal_distribution<> dist(0, 1);

  float* arr = static_cast<float*>(malloc(total_size * sizeof(float)));
  for (int64_t i = 0; i < total_size; ++i) {
    arr[i] = dist(e2);
  }
  return arr;
}

float* CreateFillFloat32HostArray(int64_t total_size, float fill_value) {
  float* arr = static_cast<float*>(malloc(total_size * sizeof(float)));
  for (int64_t i = 0; i < total_size; ++i) {
    arr[i] = fill_value;
  }
  return arr;
}

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
TensorType GetRandomFloat32Tensor(std::initializer_list<int64_t> shape, DLDevice device) {}

template <>
NDArray GetRandomFloat32Tensor<NDArray>(std::initializer_list<int64_t> shape, DLDevice device) {
  auto destination = NDArray::Empty(shape, {kDLFloat, 32, 1}, device);
  auto total_size = GetTotalSize(shape);
  auto src = CreateRandomFloat32HostArray(total_size);
  destination.CopyFromBytes(src, total_size * sizeof(float));
  delete src;
  return destination;
}

template <>
DLTensor* GetRandomFloat32Tensor<DLTensor*>(std::initializer_list<int64_t> shape, DLDevice device) {
  static std::vector<NDArray> arrays;
  auto array = GetRandomFloat32Tensor<NDArray>(shape, device);
  arrays.push_back(array);
  return const_cast<DLTensor*>(array.operator->());
}

template <typename TensorType>
TensorType GetFillFloat32Tensor(std::initializer_list<int64_t> shape, DLDevice device,
                                float fill_value = 0.1) {}

template <>
NDArray GetFillFloat32Tensor<NDArray>(std::initializer_list<int64_t> shape, DLDevice device,
                                      float fill_value) {
  auto destination = NDArray::Empty(shape, {kDLFloat, 32, 1}, device);
  auto total_size = GetTotalSize(shape);
  auto src = CreateFillFloat32HostArray(total_size, fill_value);
  destination.CopyFromBytes(src, total_size * sizeof(float));
  delete src;
  return destination;
}

template <>
DLTensor* GetFillFloat32Tensor<DLTensor*>(std::initializer_list<int64_t> shape, DLDevice device,
                                          float fill_value) {
  static std::vector<NDArray> arrays;
  auto array = GetFillFloat32Tensor<NDArray>(shape, device);
  arrays.push_back(array);
  return const_cast<DLTensor*>(array.operator->());
}

float GetTensorMean(std::initializer_list<int64_t> shape, DLTensor* tensor) {
  auto destination = NDArray::Empty(shape, {kDLFloat, 32, 1}, {kDLCPU, 0});
  destination.CopyFrom(tensor);
  auto arr = static_cast<float*>(destination.operator->()->data);
  float sum = 0;
  auto total_size = GetTotalSize(shape);
  for (int64_t i = 0; i < total_size; ++i) {
    sum += arr[i];
  }
  return sum / total_size;
}
