import os
os.environ["DIETCODE_CODEGEN_OPT"] = "1"
os.environ["DIETCODE_DO_LOCAL_PADDING"] = "1"
os.environ["DIETCODE_DO_LOOP_PARTITIONING"] = "1"

import numpy as np
import tvm
from tvm import te, auto_scheduler

M = 1024
N = te.var("N")
dtype = "float32"
target = "cuda"

A = te.placeholder((N, M), name="A", dtype=dtype)
B = te.placeholder((N, M), name="B", dtype=dtype)
C = te.compute((N, M), lambda i, j: A[i, j] * B[i, j], name="C")

s = te.create_schedule([C.op])


C_i, C_j = tuple(C.op.axis) + tuple(C.op.reduce_axis)
C_i_j_fused = s[C].fuse(C_i, C_j)
C_i_j_fused_o, C_i_j_fused_i = s[C].split(C_i_j_fused, factor=64)
s[C].bind(C_i_j_fused_o, te.thread_axis("blockIdx.x"))
s[C].bind(C_i_j_fused_i, te.thread_axis("threadIdx.x"))

args = [N, A, B, C]
print(tvm.lower(s, args, simple_mode=True))
built = tvm.build(s, args=args, target=target)

N = 449

ctx = tvm.gpu(0)
Ai = tvm.nd.empty((N, M), dtype="float32", device=ctx)
Bi = tvm.nd.empty((N, M), dtype="float32", device=ctx)
Ci = tvm.nd.empty((N, M), dtype="float32", device=ctx)
inputs = [N, Ai, Bi, Ci]

evaluator = built.time_evaluator(built.entry_name, ctx, repeat=5, number=100)
eval_result = evaluator(*inputs)
def mean(l): return sum(l) / len(l)
print(mean(list(eval_result.results)[1:]) * 1000)