import os
# os.environ["DIETCODE_CODEGEN_OPT"] = "1"
# os.environ["DIETCODE_DO_LOCAL_PADDING"] = "1"
# os.environ["DIETCODE_DO_LOOP_PARTITIONING"] = "1"

import sys
import timeit
import numpy as np
import tvm
from tvm import relay
from tvm import auto_scheduler
from converter import initialize_tlstm, generate_random_trees, get_random_tensor
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
from utils import get_ansor_log_file

device = tvm.runtime.device("cuda")

hidden_size = 256
batch_size = 8
num_nodes = 6

# target = "llvm -mcpu=core-avx2"
target = "cuda"

lazy_execution=True
coarsened_execution=True
batched_execution=True
scattered_kernels=True
concurrent_execution=False
use_autoscheduler=True
aot_output_directory="/home/ppf/dyn_batch/tvm/ppf_tests/aot_test"
# aot_output_directory="/home/ppf/data/ppf/projects/projects/dyn_batch/tvm/ppf_tests/aot_test"
model_name="treelstm"
generate_aot_code=True
dynamic_batch_size_estimate=997

tlstm, mod, prelude = initialize_tlstm(hidden_size, hidden_size)
mod = tvm.relay.transform.RemoveUnusedFunctions(batched_execution=batched_execution)(mod)
weight_vars = tlstm.all_weights()

trees = generate_random_trees(num_nodes, batch_size, (1, hidden_size), prelude)

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
                num_measure_trials=20,
                runner=measure_ctx.runner,
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            )
            tuner.tune(tune_option)

        # for task in tasks:
        #     try:
        #         print("YOLO", task.print_best(log_file))
        #     except Exception:
        #         pass

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
                iters = 100
                timeit.timeit(fin_executor, number=50)
                print_time(timeit.timeit(fin_executor, number=iters)*1000/iters)

auto_schedule((not os.path.exists(log_file)))
print("===============================================================================", flush=True)
execute()
