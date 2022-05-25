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
from utils import get_ansor_log_file, get_random_tensor, pgo_and_auto_schedule
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

hidden_size = 256
batch_size = 8
tree_height = 6

lazy_execution=True
coarsened_execution=True
batched_execution=True
scattered_kernels=True
concurrent_execution=True
use_autoscheduler=True
use_depth_tracking=True
perform_static_scheduling=False
aot_output_directory=TVM_HOME + "/ppf_tests/aot_test"
model_name="treelstm"
generate_aot_code=True
dynamic_batch_size_estimate=256

mod = tvm.IRModule()
mod.import_from_std("prelude.rly")
mod._import(TVM_HOME + "/ppf_tests/treelstm/treelstm.rly")

mod = tvm.relay.transform.RemoveUnusedFunctions(batched_execution=batched_execution)(mod)
main_func = mod["main"]

weights_list = []
weights_dict = {
    main_func.params[0].name_hint: get_random_tensor((hidden_size, hidden_size)),
    main_func.params[1].name_hint: get_random_tensor((hidden_size, hidden_size)),
    main_func.params[2].name_hint: get_random_tensor((3*hidden_size, hidden_size)),
    main_func.params[3].name_hint: get_random_tensor((1, 3*hidden_size)),
    main_func.params[4].name_hint: get_random_tensor((1, 3*hidden_size)),
    main_func.params[5].name_hint: get_random_tensor((3*hidden_size, hidden_size)),
    main_func.params[6].name_hint: get_random_tensor((1, hidden_size)),
    main_func.params[7].name_hint: get_random_tensor((1, hidden_size)),
}
for i in range(len(weights_dict)):
    weights_list.append(weights_dict[main_func.params[i].name_hint])

trees = generate_complete_treelstm_trees(tree_height, batch_size, (1, hidden_size), mod)

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
