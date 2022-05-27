import os
os.environ["DIETCODE_CODEGEN_OPT"] = "1"
os.environ["DIETCODE_DO_LOCAL_PADDING"] = "1"
os.environ["DIETCODE_DO_LOOP_PARTITIONING"] = "1"
TVM_HOME = os.environ["TVM_HOME"]

import sys
import timeit
import numpy as np
import tvm
from tvm import relay
from tvm import auto_scheduler
from treelstm import TreeLSTM
from network import copy_var
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from utils import get_ansor_log_file, get_random_tensor, pgo_and_auto_schedule, get_cmd_parser
from tree_utils import generate_complete_treelstm_trees

def initialize_tlstm(input_size, memory_size):
    tlstm = TreeLSTM(input_size=input_size, memory_size=memory_size, name="treelstm")
    mod = tlstm.mod
    tlstm_func = mod[tlstm.f]
    tlstm_gv = tlstm.f
    gv = relay.GlobalVar("main")
    main_params = [copy_var(v) for v in tlstm_func.params]
    mod[gv] = relay.Function(main_params, tlstm_gv(*main_params), tlstm_func.ret_type)
    return tlstm, mod, tlstm.p

# target = "llvm -mcpu=core-avx2"
target = "cuda"
if target.startswith("cuda"): device = tvm.runtime.device("cuda", 0)
else: device = tvm.runtime.device("cpu")

args = get_cmd_parser().parse_args()

hidden_sizes = {"small": 256, "large": 512}
hidden_size = hidden_sizes[args.hidden]
batch_size = 8
tree_height = 6

batched_execution=True
model_name="treelstm_" + args.hidden

lazy_execution=args.lazy
coarsened_execution=args.coarsened
scattered_kernels=args.scattered
concurrent_execution=args.concurrent
use_autoscheduler=args.autosched
use_depth_tracking=args.depth_tracking
perform_static_scheduling=args.static_scheduling
aot_output_directory=args.aot_out_dir
generate_aot_code=args.aot_code
dynamic_batch_size_estimate=args.bs_estimate

tlstm, mod, prelude = initialize_tlstm(hidden_size, hidden_size)
mod = tvm.relay.transform.RemoveUnusedFunctions(batched_execution=batched_execution)(mod)
weight_vars = tlstm.all_weights()

trees = generate_complete_treelstm_trees(tree_height, batch_size, (1, hidden_size), mod)

weights_list = []
weights_dict = {}
for weight in weight_vars:
    tensor = get_random_tensor(tuple([int(i) for i in weight.type_annotation.shape]))
    weights_list.append(tensor)
    weights_dict[weight.name_hint] = tensor

pass_context, execution_options = relay.backend.vm.create_workflow_configs(
    lazy_execution=lazy_execution,
    coarsened_execution=coarsened_execution,
    batched_execution=batched_execution,
    scattered_kernels=scattered_kernels,
    concurrent_execution=concurrent_execution,
    use_depth_tracking=use_depth_tracking,
    perform_static_scheduling=perform_static_scheduling,
    use_autoscheduler=use_autoscheduler,
    dynamic_batch_size_estimate=dynamic_batch_size_estimate,
    batch_size=batch_size,
    aot_output_directory=aot_output_directory,
    model_name=model_name,
    generate_aot_code=generate_aot_code,
    opt_level=3)

def print_time(time):
    print(
        lazy_execution,
        coarsened_execution,
        batched_execution,
        scattered_kernels,
        concurrent_execution,
        use_autoscheduler,
        dynamic_batch_size_estimate,
        batch_size,
        time
    )

log_file = get_ansor_log_file(model_name, [hidden_size], pass_context, target)
def auto_schedule():
    pgo_and_auto_schedule(mod, weights_dict, trees, batch_size, log_file,
                          target, pass_context, execution_options, fin_iterations=50)

def execute():
    with tvm.auto_scheduler.ApplyHistoryBest(log_file):
        with pass_context:
            executor = relay.backend.vm.VMExecutor(mod, device, target)
            if generate_aot_code:
                executable = executor.compile(params=weights_dict)
            else:
                fin_executor = executor._make_executor(execution_options=execution_options, params=weights_dict)
                params_list = []
                for tree in trees: params_list += [tree]

                executor.vm.set_input("main", batch_size, *params_list)
                # exit()
                # fin_executor()
                iters = 100
                timeit.timeit(fin_executor, number=50)
                print_time(timeit.timeit(fin_executor, number=iters)*1000/iters)

if use_autoscheduler: auto_schedule()
print("===============================================================================", flush=True)
execute()
