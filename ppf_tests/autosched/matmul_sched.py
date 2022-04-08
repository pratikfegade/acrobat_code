import os
os.environ["DIETCODE_CODEGEN_OPT"] = "1"
os.environ["DIETCODE_DO_LOCAL_PADDING"] = "1"
os.environ["DIETCODE_DO_LOOP_PARTITIONING"] = "1"

os.environ["DIETCODE_SCHED_OPT_PARTITION_BLOCKIDX"] = "1"
os.environ["DIETCODE_SCHED_OPT_PARTITION_CONST_LOOPS"] = "1"
os.environ["DIETCODE_SCHED_OPT"] = "1"


import numpy as np
import tvm
from tvm import te, auto_scheduler

N = te.var("N")
L = M = 1024
dtype = "float32"
target = "cuda"

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

s = te.create_schedule([out.op])

matmul_i, matmul_j, matmul_k = tuple(matmul.op.axis) + tuple(matmul.op.reduce_axis)
out_i, out_j = tuple(out.op.axis) + tuple(out.op.reduce_axis)
matmul_i_o_i, matmul_i_i = s[matmul].split(matmul_i, factor=1)
matmul_i_o_o_i, matmul_i_o_i = s[matmul].split(matmul_i_o_i, factor=1)
matmul_i_o_o_o_i, matmul_i_o_o_i = s[matmul].split(matmul_i_o_o_i, factor=449)
matmul_i_o_o_o_o, matmul_i_o_o_o_i = s[matmul].split(matmul_i_o_o_o_i, factor=1)
matmul_j_o_i, matmul_j_i = s[matmul].split(matmul_j, factor=4)
matmul_j_o_o_i, matmul_j_o_i = s[matmul].split(matmul_j_o_i, factor=2)
matmul_j_o_o_o_i, matmul_j_o_o_i = s[matmul].split(matmul_j_o_o_i, factor=2)
matmul_j_o_o_o_o, matmul_j_o_o_o_i = s[matmul].split(matmul_j_o_o_o_i, factor=2)
matmul_k_o_i, matmul_k_i = s[matmul].split(matmul_k, factor=16)
matmul_k_o_o, matmul_k_o_i = s[matmul].split(matmul_k_o_i, factor=1)
s[matmul].reorder(matmul_i_o_o_o_o, matmul_j_o_o_o_o, matmul_i_o_o_o_i, matmul_j_o_o_o_i, matmul_i_o_o_i, matmul_j_o_o_i, matmul_k_o_o, matmul_k_o_i, matmul_i_o_i, matmul_j_o_i, matmul_k_i, matmul_i_i, matmul_j_i)
out_i_o_i, out_i_i = s[out].split(out_i, factor=1)
out_i_o_o_i, out_i_o_i = s[out].split(out_i_o_i, factor=449)
out_i_o_o_o, out_i_o_o_i = s[out].split(out_i_o_o_i, factor=1)
out_j_o_i, out_j_i = s[out].split(out_j, factor=8)
out_j_o_o_i, out_j_o_i = s[out].split(out_j_o_i, factor=2)
out_j_o_o_o, out_j_o_o_i = s[out].split(out_j_o_o_i, factor=2)
s[out].reorder(out_i_o_o_o, out_j_o_o_o, out_i_o_o_i, out_j_o_o_i, out_i_o_i, out_j_o_i, out_i_i, out_j_i)
s[matmul].compute_at(s[out], out_j_o_i)
B_shared = s.cache_read(B, "shared", [matmul])
B_shared_ax0, B_shared_ax1 = tuple(B_shared.op.axis)
s[B_shared].compute_at(s[matmul], matmul_k_o_o)
A_shared = s.cache_read(A, "shared", [matmul])
A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
s[A_shared].compute_at(s[matmul], matmul_k_o_o)
out_i_o_o_o_j_o_o_o_fused = s[out].fuse(out_i_o_o_o, out_j_o_o_o)
s[out].bind(out_i_o_o_o_j_o_o_o_fused, te.thread_axis("blockIdx.x"))
out_i_o_o_i_j_o_o_i_fused = s[out].fuse(out_i_o_o_i, out_j_o_o_i)
s[out].bind(out_i_o_o_i_j_o_o_i_fused, te.thread_axis("vthread"))
out_i_o_i_j_o_i_fused = s[out].fuse(out_i_o_i, out_j_o_i)
s[out].bind(out_i_o_i_j_o_i_fused, te.thread_axis("threadIdx.x"))
B_shared_ax0_ax1_fused = s[B_shared].fuse(B_shared_ax0, B_shared_ax1)
B_shared_ax0_ax1_fused_o, B_shared_ax0_ax1_fused_i = s[B_shared].split(B_shared_ax0_ax1_fused, factor=2)
s[B_shared].vectorize(B_shared_ax0_ax1_fused_i)
B_shared_ax0_ax1_fused_o_o, B_shared_ax0_ax1_fused_o_i = s[B_shared].split(B_shared_ax0_ax1_fused_o, factor=898)
s[B_shared].bind(B_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))
A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=1)
s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=898)
s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))
# s[matmul].pragma(matmul_i_o_o_o_o, "auto_unroll_max_step", 256)
# s[matmul].pragma(matmul_i_o_o_o_o, "unroll_explicit", True)

print_after_passes = [
    # "tir.HoistIfThenElse",
    # "tir.RewriteUnsafeSelect",
]

args = [N, A, B, C, out]
# lowered = tvm.lower(s, args, simple_mode=True, print_after_passes=print_after_passes)
# print(lowered)
built = tvm.build(s, args=args, target=target)
print(built.imported_modules[0].get_source())

N = 449

ctx = tvm.gpu(0)
Ai = tvm.nd.empty((N, L), dtype="float32", device=ctx)
Bi = tvm.nd.empty((L, M), dtype="float32", device=ctx)
Ci = tvm.nd.empty((N, M), dtype="float32", device=ctx)
outi = tvm.nd.empty((N, M), dtype="float32", device=ctx)
inputs = [N, Ai, Bi, Ci, outi]

evaluator = built.time_evaluator(built.entry_name, ctx, repeat=5, number=100)
eval_result = evaluator(*inputs)
def mean(l): return sum(l) / len(l)
print(mean(list(eval_result.results)[1:]) * 1000)
