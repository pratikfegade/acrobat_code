from tvm import te
import argparse
import numpy as np
import os
import tvm

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--batch-size', dest='batch_size', default=1, type=int)
parser.add_argument('--peel-loops', dest='peel_loops', default=False, action='store_true')
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--debug-code', dest='debug_code', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--hidden-size', dest='hidden_size', default=256, type=int)
args = parser.parse_args()

peel_loops = args.peel_loops
float_dtype = args.dtype
num_gates = 4
hidden_size = args.hidden_size

##################### Model placeholders #####################
batch_size = 128

W = te.placeholder((hidden_size, hidden_size), name = 'W', dtype = float_dtype)
I = te.placeholder((batch_size, hidden_size), name = 'I', dtype = float_dtype)

##################### Computation #####################
h_idx, c_idx = 0, 1
k = te.reduce_axis((0, hidden_size), name = 'i_kh2h')
O = te.compute((batch_size, hidden_size),
               lambda n, i: te.sum(W[i, k] * I[n, k], axis = k),
               name = 'O')

##################### Scheduling #####################
s = te.create_schedule([O.op])

if True:
    x, y = s[O].leaf_iter_vars[0:2]
    s[O].parallel(x)

print_after_passes = [
    # "tir.VerifyMemory",
    # "tir.ThreadSync",
    # "tir.LowerThreadAllreduce",
    # "tir.SplitHostDevice"
    "tir.CombineContextCall"
]

with tvm.transform.PassContext(config={ "tir.detect_global_barrier": False }):
    inputs = [W, I, O]

    if (args.debug_code):
        # lowered = tvm.lower(s, inputs, simple_mode=False,
                            # print_after_passes=print_after_passes)
        # print(lowered)
        fadd = tvm.build(s, inputs, args.target,
                         print_after_passes=print_after_passes)
        if args.target == 'cuda':
            # print('-----GPU code-----\n' + fadd.get_source())
            print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
        else:
            print('-----CPU code-----\n' + fadd.get_source())
    else:
        fadd = tvm.build(s, inputs, args.target)
        fadd.export_library('lstm.so')
