import os
os.environ["AUTOSCHEDULER_STAND_ALONE"] = "1"

import numpy as np
import tvm
from tvm import te, auto_scheduler

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul_add(X, N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B0 = te.placeholder((X, L, M), name="B0", dtype=dtype)
    B1 = te.placeholder((X, L, M), name="B1", dtype=dtype)
    B2 = te.placeholder((X, L, M), name="B2", dtype=dtype)

    B = te.compute((X, 3, L, M), lambda b, n, i, j:
                   te.if_then_else(n == 0, B0[b, i, j], te.if_then_else(n == 1, B1[b, i, j], B2[b, i, j])),
                   name="B")

    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute(
        (X, 3, N, M),
        lambda b, n, i, j: te.sum(A[i, k] * B[b, n, k, j], axis=k),
        name="matmul",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )

    C0 = te.compute((X, N, M), lambda b, i, j: matmul[b, 0, i, j], name="C0")
    C1 = te.compute((X, N, M), lambda b, i, j: te.tanh(matmul[b, 1, i, j]), name="C1")
    C2 = te.compute((X, N, M), lambda b, i, j: te.sigmoid(matmul[b, 2, i, j]), name="C2")

    return [A, B0, B1, B2, C0, C1, C2]

target = tvm.target.Target("cuda")
N = L = M = 512
X = 8
task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(X, N, L, M, "float32"), target=target)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)


log_file = "matmul.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
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
