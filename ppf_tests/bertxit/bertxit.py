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
from utils import get_ansor_log_file, get_random_tensor, pgo_and_auto_schedule

mod = tvm.IRModule()
mod._import(TVM_HOME + "/ppf_tests/bertxit/bertxit.rly")
mod = tvm.relay.transform.RemoveUnusedFunctions(batched_execution=True)(mod)

main_func = mod["main"]

num_heads=8
head_size=64
model_size=head_size*num_heads
ff_size=2048
seq_len=128
batch_size=4
target = "cuda"
device = tvm.runtime.device(target)

weights_list = []
weights_dict = {
    main_func.params[0].name_hint: get_random_tensor((model_size, model_size)),
    main_func.params[1].name_hint: get_random_tensor((model_size, model_size)),
    main_func.params[2].name_hint: get_random_tensor((model_size, model_size)),
    main_func.params[3].name_hint: get_random_tensor((1, model_size)),
    main_func.params[4].name_hint: get_random_tensor((1, model_size)),
    main_func.params[5].name_hint: get_random_tensor((1, model_size)),
    main_func.params[6].name_hint: get_random_tensor((model_size,)),
    main_func.params[7].name_hint: get_random_tensor((model_size,)),
    main_func.params[8].name_hint: get_random_tensor((model_size, model_size)),
    main_func.params[9].name_hint: get_random_tensor((1, model_size)),
    main_func.params[10].name_hint: get_random_tensor((ff_size, model_size)),
    main_func.params[11].name_hint: get_random_tensor((1, ff_size)),
    main_func.params[12].name_hint: get_random_tensor((model_size, ff_size)),
    main_func.params[13].name_hint: get_random_tensor((1, model_size)),
    main_func.params[14].name_hint: get_random_tensor((model_size,)),
    main_func.params[15].name_hint: get_random_tensor((model_size,)),
    main_func.params[16].name_hint: get_random_tensor((1, 512)),
    main_func.params[17].name_hint: get_random_tensor((1, 1)),
    main_func.params[18].name_hint: get_random_tensor((16, 512)),
    main_func.params[19].name_hint: get_random_tensor((1, 16))
}
for i in range(len(weights_dict)):
    weights_list.append(weights_dict[main_func.params[i].name_hint])

inputs = []
for i in range(batch_size):
    inputs.append(get_random_tensor((seq_len, model_size)))

lazy_execution=True
coarsened_execution=False
batched_execution=True
scattered_kernels=True
concurrent_execution=True
use_autoscheduler=True
use_depth_tracking=True
perform_static_scheduling=False
aot_output_directory=TVM_HOME + "/ppf_tests/aot_test/"
model_name="bertxit"
generate_aot_code=True
dynamic_batch_size_estimate=8
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

log_file = get_ansor_log_file(model_name, [ff_size, num_heads, head_size, model_size], pass_context, target)
def auto_schedule(tune):
    pgo_and_auto_schedule(mod, weights_dict, inputs, batch_size, log_file,
                          target, pass_context, execution_options)

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
                exit()
                # fin_executor()
                # iters = 100
                # timeit.timeit(fin_executor, number=50)
                # print_time(timeit.timeit(fin_executor, number=iters)*1000/iters)

if use_autoscheduler: auto_schedule((not os.path.exists(log_file)))
# print("===============================================================================", flush=True)
# execute()
