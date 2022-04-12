import os
# os.environ["DIETCODE_CODEGEN_OPT"] = "1"
# os.environ["DIETCODE_DO_LOCAL_PADDING"] = "1"
# os.environ["DIETCODE_DO_LOOP_PARTITIONING"] = "1"

from tvm import te
import argparse
import numpy as np
import tvm

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--batch-size', dest='batch_size', default=1, type=int)
parser.add_argument('--peel-loops', dest='peel_loops', default=False, action='store_true')
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--const-bs', dest='const_bs', default=False, action='store_true')
parser.add_argument('--debug-code', dest='debug_code', nargs='?', default='none')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--hidden-size', dest='hidden_size', default=256, type=int)
args = parser.parse_args()

peel_loops = args.peel_loops
float_dtype = args.dtype
num_gates = 4
hidden_size = args.hidden_size

##################### Model placeholders #####################
batch_size = 128 if args.const_bs else te.var("BS")

W = te.placeholder((hidden_size, hidden_size), name = 'W', dtype = float_dtype)
I = te.placeholder((batch_size, hidden_size), name = 'I', dtype = float_dtype)

##################### Computation #####################
k = te.reduce_axis((0, hidden_size), name = 'i_kh2h')
Ol = te.compute((batch_size, hidden_size),
                lambda n, i: te.sum(W[i, k] * I[n, k], axis = k),
                name = 'Ol')

O = te.compute((batch_size, hidden_size), lambda n, i: 2*Ol[n, k], name = 'O')

##################### Scheduling #####################
s = te.create_schedule([O.op])

if args.target == "cuda":
    thread_x = lambda: te.thread_axis("threadIdx.x")
    thread_y = lambda: te.thread_axis("threadIdx.y")
    block_x = lambda: te.thread_axis("blockIdx.x")
    block_y = lambda: te.thread_axis("blockIdx.y")

    ############# Scheduling for internal nodes #############
    hS = s.cache_read(I, 'shared', [Ol])
    wS = s.cache_read(W, 'shared', [Ol])

    s[Ol].set_scope('local')

    x, y = s[O].leaf_iter_vars[0:2]
    xo, xi = s[O].split(x, nparts = 4)
    yo, yi = s[O].split(y, nparts = 4)

    s[O].reorder(xo, yo, xi, yi)

    s[O].bind(xo, block_y())
    s[O].bind(yo, block_x())
    yio, yii = s[O].split(yi, factor = 16)
    s[O].reorder(yio, xi, yii)
    s[O].bind(xi, thread_y())
    s[O].bind(yio, te.thread_axis("vthread"))
    s[O].bind(yii, thread_x())

    s[Ol].compute_at(s[O], yii)
    x, y, k = s[Ol].leaf_iter_vars
    k = s[Ol].op.reduce_axis[0]
    ko, ki = s[Ol].split(k, nparts = 16)
    s[Ol].reorder(ko, x, y, ki)
    s[hS].compute_at(s[Ol], ko)
    s[wS].compute_at(s[Ol], ko)

    s[hS].bind(s[hS].leaf_iter_vars[0], thread_y())
    s[hS].bind(s[hS].leaf_iter_vars[1], thread_x())

    xo, xi = s[wS].split(s[wS].leaf_iter_vars[0], factor = 4)
    s[wS].bind(xi, thread_y())
    s[wS].bind(s[wS].leaf_iter_vars[2], thread_x())

print_after_passes = [
    # "tir.NarrowDataType",
    # "tir.InjectPrefetch",
    # "tir.LowerThreadAllreduce",
    # "tir.LowerThreadAllreduce",
    # "tir.SplitHostDevice"
    # "tir.CombineContextCall",
    # "tir.LowerScatterLoadsAndStores"
]

with tvm.transform.PassContext(config={ "tir.detect_global_barrier": False }):
    inputs = [W, I, O] if args.const_bs else [batch_size, W, I, O]

    if args.debug_code == "ir":
        lowered = tvm.lower(s, inputs, simple_mode=False,
                            print_after_passes=print_after_passes)
        print(lowered)
    elif args.debug_code == "code":
        fadd = tvm.build(s, inputs, args.target,
                         print_after_passes=print_after_passes)
        if args.target == 'cuda':
            print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
        else:
            print('-----CPU code-----\n' + fadd.get_source())
    else:
        fadd = tvm.build(s, inputs, args.target)
        fadd.export_library('lstm.so')
