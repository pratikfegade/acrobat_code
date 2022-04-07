import os
os.environ["DIETCODE_CODEGEN_OPT"] = "1"
os.environ["DIETCODE_DO_LOCAL_PADDING"] = "1"
os.environ["DIETCODE_DO_LOOP_PARTITIONING"] = "1"

import numpy as np
import tvm
from tvm import te, auto_scheduler

N = te.var("N")
M = 1024
dtype = "float32"

scale = 0.125
A = te.placeholder((N, M), name='A')
k = te.reduce_axis((0, M), name="k")
Amax = te.compute((N,), lambda b: te.max(A[b, k], axis=k), name = 'Amax')
Aexp = te.compute((N, M), lambda b, i: te.exp((A[b, i] - Amax[b]) * scale), name = 'Aexp')
k = te.reduce_axis((0, M), name="k")
Asum = te.compute((N,), lambda b: te.sum(Aexp[b, k], axis=k), name = 'Asum')
O = te.compute((N, M), lambda b, i: Aexp[b, i] / Asum[b], name = 'O')

s = te.create_schedule([O.op])


Amax_b, Amax_k = tuple(Amax.op.axis) + tuple(Amax.op.reduce_axis)
Aexp_b, Aexp_i = tuple(Aexp.op.axis) + tuple(Aexp.op.reduce_axis)
Asum_b, Asum_k = tuple(Asum.op.axis) + tuple(Asum.op.reduce_axis)
O_b, O_i = tuple(O.op.axis) + tuple(O.op.reduce_axis)
s[Aexp].compute_inline()
Amax_k_o, Amax_k_i = s[Amax].split(Amax_k, factor=32)
s[Amax].bind(Amax_k_i, te.thread_axis("threadIdx.x"))
O_b_i_fused = s[O].fuse(O_b, O_i)
O_b_i_fused_o, O_b_i_fused_i = s[O].split(O_b_i_fused, factor=64)
s[O].bind(O_b_i_fused_o, te.thread_axis("blockIdx.x"))
s[O].bind(O_b_i_fused_i, te.thread_axis("threadIdx.x"))
Asum_b_o, Asum_b_i = s[Asum].split(Asum_b, factor=1)
s[Asum].bind(Asum_b_o, te.thread_axis("blockIdx.x"))
s[Asum].bind(Asum_b_i, te.thread_axis("threadIdx.x"))
s[Amax].bind(Amax_b, te.thread_axis("blockIdx.x"))
s[Amax].pragma(Amax_b, "auto_unroll_max_step", 512)
s[Amax].pragma(Amax_b, "unroll_explicit", True)
s[Asum].pragma(Asum_b_o, "auto_unroll_max_step", 64)
s[Asum].pragma(Asum_b_o, "unroll_explicit", True)

args = [N, A, O]
print(tvm.lower(s, args, simple_mode=True))
