import os

import numpy as np
import tvm
from tvm import te, auto_scheduler

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def softmax(N, M, dtype):
    scale = 0.125
    A = te.placeholder((N, M), name='A')
    k = te.reduce_axis((0, M), name="k")
    Amax = te.compute((N,), lambda b: te.max(A[b, k], axis=k), name = 'Amax')
    Aexp = te.compute((N, M), lambda b, i: te.exp((A[b, i] - Amax[b]) * scale), name = 'Aexp')
    k = te.reduce_axis((0, M), name="k")
    Asum = te.compute((N,), lambda b: te.sum(Aexp[b, k], axis=k), name = 'Asum')
    O = te.compute((N, M), lambda b, i: Aexp[b, i] / Asum[b], name = 'O')
    return [A, Amax, Aexp, Asum, O]

target = tvm.target.Target("cuda")
M = 1024
N = 449
task = tvm.auto_scheduler.SearchTask(func=softmax, args=(N, M, "float32"), target=target)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

log_file = "softmax.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=500,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=1,
)

# Run auto-tuning (search)
task.tune(tune_option)

# Apply the best schedule
sch, args = task.apply_best(log_file)

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))
print(task.print_best(log_file))
