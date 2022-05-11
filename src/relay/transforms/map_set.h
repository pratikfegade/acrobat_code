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

#include <tvm/ir/expr.h>
#include <tvm/runtime/container/map.h>

namespace tvm {
namespace relay {

class MapSet {
 public:
  template <typename T>
  static Map<T, Bool> Create(std::initializer_list<T> values) {
    Map<T, Bool> res;
    for (auto value : values) {
      res.Set(value, bool_true);
    }
    return res;
  }

  template <typename T>
  static void Insert(Map<T, Bool>& map, T value) {
    map.Set(value, bool_true);
  }

  template <typename T>
  static void Remove(Map<T, Bool>& map, T value) {
    map.erase(value);
  }

  template <typename T>
  static bool Contains(Map<T, Bool>& map, T value) {
    return map.count(value);
  }

  template <typename T>
  static Map<T, Bool> Merge(const Map<T, Bool>& map1, const Map<T, Bool>& map2) {
    Map<T, Bool> res;
    for (auto kv : map1) {
      res.Set(kv.first, kv.second);
    }
    for (auto kv : map2) {
      res.Set(kv.first, kv.second);
    }
    return res;
  }

  template <typename T>
  static Map<T, Bool> Merge(const Array<Map<T, Bool>>& maps) {
    Map<T, Bool> res;
    for (auto& map : maps) {
      for (auto& kv : map) {
        res.Set(kv.first, kv.second);
      }
    }
    return res;
  }

  static Bool bool_true;
};

}  // namespace relay
}  // namespace tvm
