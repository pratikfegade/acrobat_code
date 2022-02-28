import os

import numpy as np
import tvm
from tvm import te, auto_scheduler

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul_add(N, L, M, dtype):
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
# N = L = M = 512
L = M = 512
N = te.var('N')
task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(N, L, M, "float32"), target=target)
concrete_task = task.make_concrete({N: 512})

# Inspect the computational graph
print("Computational DAGs:")
print(task.compute_dag)
print(task.compute_dag.tensors)
print(concrete_task.compute_dag)
print(concrete_task.compute_dag.tensors)
auto_scheduler.register_workload_tensors(concrete_task.workload_key, concrete_task.compute_dag.tensors)
# exit(0)
log_file = "/home/ppf/Documents/projects/projects/dyn_batch/tvm/build/matmul.json"
if os.path.exists(log_file):
    os.remove(log_file)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=2,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    num_measures_per_round=32,
    verbose=2,
)

# Run auto-tuning (search)
concrete_task.tune(tune_option)
# task.tune(tune_option)
# Apply the best schedule
sch, args = task.apply_best_from_other_task(log_file, concrete_task)

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))

print("Equivalent python schedule:")
print(concrete_task.print_best(log_file))












# AssertionError('Traceback (most recent call last):
#   File "/home/ppf/Documents/projects/projects/dyn_batch/tvm/python/tvm/exec/popen_worker.py", line 87, in main
#     result = fn(*args, **kwargs)
#   File "/home/ppf/Documents/projects/projects/dyn_batch/tvm/python/tvm/auto_scheduler/measure.py", line 662, in local_build_worker
#     return _local_build_worker(inp, build_func, verbose)
#   File "/home/ppf/Documents/projects/projects/dyn_batch/tvm/python/tvm/auto_scheduler/measure.py", line 601, in _local_build_worker
#     inp = MeasureInput.deserialize(inp_serialized)
#   File "/home/ppf/Documents/projects/projects/dyn_batch/tvm/python/tvm/auto_scheduler/measure.py", line 149, in deserialize
#     return recover_measure_input(inp)
#   File "/home/ppf/Documents/projects/projects/dyn_batch/tvm/python/tvm/auto_scheduler/measure.py", line 230, in recover_measure_input
#     new_task = SearchTask(
#   File "/home/ppf/Documents/projects/projects/dyn_batch/tvm/python/tvm/auto_scheduler/search_task.py", line 449, in __init__
#     assert (
# AssertionError: Either a workload generation function or a comppppute dag should be provided
# '), all_cost:15.00, Tstamp:1646085118.47)
