import os
os.environ["DIETCODE_CODEGEN_OPT"] = "1"
os.environ["DIETCODE_DO_LOCAL_PADDING"] = "1"
os.environ["DIETCODE_DO_LOOP_PARTITIONING"] = "1"
TVM_HOME = os.environ["TVM_HOME"]

import timeit
import numpy as np
import tvm
import tvm.runtime as trt
from tvm import relay
from tvm import auto_scheduler
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from utils import get_ansor_log_file, get_random_tensor, pgo_and_auto_schedule, get_cmd_parser
from tree_utils import generate_complete_mvrnn_trees

parser = get_cmd_parser().parse_args()

batch_size=8
hidden_sizes = {"small": 64, "large": 128}
hidden_size=hidden_sizes[args.hidden]
tree_height=6
target = "cuda"
device = tvm.runtime.device(target)

mod = tvm.IRModule()
model_name="mvrnn_" + args.hidden
mod._import(TVM_HOME + "/ppf_tests/mvrnn/" + model_name + ".rly")

mvrnn_func = mod["mvrnn"]

trees = generate_complete_mvrnn_trees(tree_height, batch_size, hidden_size, mod)

vweight = get_random_tensor((hidden_size, 2*hidden_size))
vbias = get_random_tensor((1, hidden_size))
mweight = get_random_tensor((hidden_size, 2*hidden_size))
mbias = get_random_tensor((1, hidden_size))

weights_list = [vweight, vbias, mweight, mbias]
weights_dict = {
    mvrnn_func.params[0].name_hint: vweight,
    mvrnn_func.params[1].name_hint: vbias,
    mvrnn_func.params[2].name_hint: mweight,
    mvrnn_func.params[3].name_hint: mbias
}

batched_execution=True

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
                          target, pass_context, execution_options, fin_iterations=20)

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
                iters = 100
                timeit.timeit(fin_executor, number=50)
                print_time(timeit.timeit(fin_executor, number=iters)*1000/iters)

if use_autoscheduler: auto_schedule()
print("===============================================================================", flush=True)
execute()
