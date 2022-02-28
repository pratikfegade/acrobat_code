import os

import numpy as np
import tvm
from tvm import te, auto_scheduler

@auto_scheduler.register_workload
def matmul_add(N, L, M, dtype):
    print(N, L, M, dtype)
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    C = te.placeholder((N, M), name="C", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute(
        (N, M),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )
    out = te.compute((N, M), lambda i, j: matmul[i, j] + C[i, j], name="out")

    return [A, B, C, out]

target = tvm.target.Target("llvm")
N = L = M = 1024
# V = te.var('Vavavoom')
N =  7865
# task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(N, L, M, "float32"), target=target)
compute_dag = tvm.auto_scheduler.ComputeDAG(matmul_add(N, L, M, "float32"))
task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(N, L, M, "float32"),
                                     compute_dag=compute_dag, target=target)
# ctask = task.make_concrete({V: 7865})
# print(ctask.compute_dag)
# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

log_file = "/home/ppf/Documents/projects/projects/dyn_batch/tvm/build/matmul.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=1,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=1,
)

task.tune(tune_option)

# ####################################################
# # V = te.var('V')
# V = N*2
# task2 = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(V, L*2, M*2, "float32"), target=target)
# ####################################################

# sch, args = task.apply_best_from_other_task(log_file, task)
sch, args = task.apply_best(log_file)

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))

print("Equivalent python schedule:")
print(task.print_best(log_file))
