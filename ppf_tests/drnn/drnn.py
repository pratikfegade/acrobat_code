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
from utils import get_ansor_log_file, get_random_tensor

mod = tvm.IRModule()
mod.import_from_std("prelude.rly")
mod._import(TVM_HOME + "/ppf_tests/drnn/drnn.rly")
mod = tvm.relay.transform.RemoveUnusedFunctions(batched_execution=True)(mod)

main_func = mod["main"]

batch_size=8
hidden_size=64
target = "cuda"
device = tvm.runtime.device(target)

weights_list = []
weights_dict = {
    main_func.params[0].name_hint: get_random_tensor((256, 512)),
    main_func.params[1].name_hint: get_random_tensor((1, 256)),
    main_func.params[2].name_hint: get_random_tensor((256, 512)),
    main_func.params[3].name_hint: get_random_tensor((1, 256)),
    main_func.params[4].name_hint: get_random_tensor((256, 512)),
    main_func.params[5].name_hint: get_random_tensor((256, 256)),
    main_func.params[6].name_hint: get_random_tensor((1, 256)),
    main_func.params[7].name_hint: get_random_tensor((1, 256)),
    main_func.params[8].name_hint: get_random_tensor((2, 256)),
}
for i in range(len(weights_dict)):
    weights_list.append(weights_dict[main_func.params[i].name_hint])

lazy_execution=True
coarsened_execution=True
batched_execution=True
scattered_kernels=True
concurrent_execution=True
use_autoscheduler=False
use_depth_tracking=True
perform_static_scheduling=False
aot_output_directory=TVM_HOME + "/ppf_tests/aot_test/"
model_name="drnn"
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

log_file = get_ansor_log_file(model_name, [hidden_size], pass_context, target)
def auto_schedule(tune):
    with pass_context:
        tasks, task_weights = auto_scheduler.extract_tasks(mod, weights_dict, target, pass_context,
                                                           include_simple_tasks=True)

        if tune:
            measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=100)
            tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=20000,
                runner=measure_ctx.runner,
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            )
            tuner.tune(tune_option)

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
print("===============================================================================", flush=True)
execute()
