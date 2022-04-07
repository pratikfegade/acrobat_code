import os

import numpy as np
import tvm
from tvm import te, auto_scheduler

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def ewise(N, M, dtype):
    A = te.placeholder((N, M), name="A", dtype=dtype)
    B = te.placeholder((N, M), name="B", dtype=dtype)
    C = te.compute((N, M), lambda i, j: A[i, j] * B[i, j], name="C")
    return [A, B, C]

MAX_BS=128
target = tvm.target.Target("cuda")
M = 1024
N = 449
task = tvm.auto_scheduler.SearchTask(func=ewise, args=(N, M, "float32"), target=target)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

log_file = "ewise.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=200,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# Run auto-tuning (search)
task.tune(tune_option)

# Apply the best schedule
sch, args = task.apply_best(log_file)

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))
print(task.print_best(log_file))
