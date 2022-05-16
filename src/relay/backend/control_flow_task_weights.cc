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
 *
 * \file src/relay/transforms/fuse_ops.cc
 *
 * \brief This is a backend-aware optimization pass.
 *   Fuse necessary ops into a single one.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include <limits>

#include "../op/annotation/annotation.h"
#include "../op/memory/memory.h"
#include "../op/vm/vm.h"
#include "../transforms/function_pointer_analysis.h"
#include "../transforms/pass_utils.h"

namespace tvm {
namespace relay {
namespace tec {

using NodeT = const FunctionNode*;

namespace {
using UFunctionSet = std::unordered_set<NodeT>;

class TarjansAlgorithm {
 public:
  TarjansAlgorithm(PreciseCallGraph& call_graph,
                   std::unordered_map<const BaseFuncNode*, std::string>& func_name_map)
      : call_graph_(call_graph), func_name_map_(func_name_map) {}

  void findComponent(NodeT u) {
    static int time = 0;
    disc[u] = lowLink[u] = ++time;
    stk.push(u);
    stkItem[u] = true;

    OrderedFunctionSet callees;
    auto it = call_graph_.find(u);
    if (it != call_graph_.end()) {
      callees = it->second;
    }

    for (auto v : callees) {
      if (disc[v] == -1) {
        findComponent(v);
        lowLink[u] = std::min(lowLink[u], lowLink[v]);
      } else if (stkItem[v]) {
        lowLink[u] = std::min(lowLink[u], disc[v]);
      }
    }

    NodeT poppedItem = nullptr;
    if (lowLink[u] == disc[u]) {
      UFunctionSet scc;
      while (stk.top() != u) {
        poppedItem = stk.top();
        // std::cout << func_name_map_[poppedItem] << " " << poppedItem << " ";
        scc.insert(poppedItem);
        stkItem[poppedItem] = false;
        stk.pop();
      }
      poppedItem = stk.top();
      // std::cout << func_name_map_[poppedItem] << " " << poppedItem << std::endl;
      scc.insert(poppedItem);
      stkItem[poppedItem] = false;
      stk.pop();

      scc_set_.push_back(scc);
    }
  }

  std::vector<UFunctionSet> Run() {
    std::unordered_set<NodeT> all_nodes;
    for (auto kv : call_graph_) {
      all_nodes.insert(kv.first);
      all_nodes.insert(kv.second.begin(), kv.second.end());
    }

    for (auto node : all_nodes) {
      disc[node] = lowLink[node] = -1;
      stkItem[node] = false;
    }

    for (auto node : all_nodes) {
      if (disc[node] == -1) {
        findComponent(node);
      }
    }
    return scc_set_;
  }

  PreciseCallGraph& call_graph_;
  std::unordered_map<const BaseFuncNode*, std::string>& func_name_map_;
  std::unordered_map<NodeT, int> disc;
  std::unordered_map<NodeT, int> lowLink;
  std::stack<NodeT> stk;
  std::unordered_map<NodeT, bool> stkItem;
  std::vector<UFunctionSet> scc_set_;
};

void AssignLevelsSCC(std::vector<std::unordered_set<int>>& scc_graph, std::vector<bool>& visited,
                     std::vector<int>& weights, int scc) {
  if (visited[scc]) {
    return;
  }
  visited[scc] = true;
  int weight = weights[scc];
  for (auto callee_scc : scc_graph[scc]) {
    weights[callee_scc] = std::max(weights[callee_scc], weight + 1);
    AssignLevelsSCC(scc_graph, visited, weights, callee_scc);
  }
}

}  // namespace

IRModule InferTaskWeights(IRModule& mod) {
  // std::cout << "Determining weights" << std::endl;
  std::unordered_map<const BaseFuncNode*, std::string> func_name_map;
  for (auto kv : mod->functions) {
    func_name_map[kv.second.get()] = kv.first->name_hint;
  }

  auto call_graph = GetPreciseCallGraph(mod);

  // std::cout << "[SCC] CALL GRAPH" << std::endl;
  // for (auto kv : call_graph) {
  //   std::cout << "[SCC] " << func_name_map[kv.first] << std::endl;
  //   for (auto fn : kv.second) {
  //     std::cout << "[SCC]   " << func_name_map[fn] << std::endl;
  //   }
  // }

  TarjansAlgorithm scc_creator(call_graph, func_name_map);
  scc_creator.Run();
  auto sccs = scc_creator.Run();

  std::unordered_map<NodeT, int> node2scc;
  for (size_t i = 0; i < sccs.size(); ++i) {
    for (auto fn : sccs[i]) {
      node2scc[fn] = i;
    }
  }

  std::vector<std::unordered_set<int>> scc_graph(sccs.size());

  for (auto kv : call_graph) {
    auto u = kv.first;
    for (auto v : kv.second) {
      auto uscc = node2scc[u];
      auto vscc = node2scc[v];
      scc_graph[uscc].insert(vscc);
    }
  }

  // std::cout << "[SCC] SCC GRAPH" << std::endl;
  // for (size_t i = 0; i < sccs.size(); ++i) {
  //   std::cout << "[SCC] " << i << ": "
  //             << support::PrintVector(std::vector<int>(scc_graph[i].begin(), scc_graph[i].end()))
  //             << std::endl;
  // }

  int entry_scc = node2scc[mod->Lookup("main").as<FunctionNode>()];
  std::vector<bool> visited(sccs.size(), false);
  std::vector<int> weights(sccs.size(), 0);
  AssignLevelsSCC(scc_graph, visited, weights, entry_scc);

  // std::cout << "[SCC] SCC WEIGHTS" << std::endl;
  // for (size_t i = 0; i < sccs.size(); ++i) {
  //   std::cout << "[SCC] " << i << " SCC " << weights[i] << std::endl;
  //   for (auto fn : sccs[i]) {
  //     std::cout << "[SCC]     " << func_name_map[fn] << std::endl;
  //   }
  // }

  std::unordered_map<NodeT, int> prim_func_weights;
  for (size_t i = 0; i < sccs.size(); ++i) {
    for (auto fn : sccs[i]) {
      OrderedFunctionSet callees;
      auto it = call_graph.find(fn);
      if (it != call_graph.end()) {
        callees = it->second;
      }
      for (auto callee : callees) {
        prim_func_weights[callee] += weights[i];
      }
    }
  }

  int recursive_multiplier =
      transform::PassContext::Current()
          ->GetConfig<Integer>("relay.db_autosched_recursive_weight_multiplier", Integer(5))
          .value();

  Map<Function, Integer> res;
  for (auto kv : prim_func_weights) {
    res.Set(GetRef<Function>(kv.first), Integer(kv.second * recursive_multiplier + 1));
  }

  return AddFunctionTaints(res, mod, tir::attr::kDBStaticAutoschedTaskWeight);
}

}  // namespace tec
}  // namespace relay
}  // namespace tvm
