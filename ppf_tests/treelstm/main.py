import os
import timeit
import numpy as np
import tvm
from tvm import relay
from tvm import auto_scheduler
from converter import initialize_tlstm, generate_random_trees, get_random_tensor

device = tvm.runtime.device("cpu")

hidden_size = 1
batch_size = 8
num_nodes = 6

target = "llvm"
lazy_execution=False
coarsened_execution=True
batched_execution=False
scattered_kernels=False
concurrent_execution=False
use_autoscheduler=False
aot_output_directory="/home/ppf/data/ppf/projects/projects/dyn_batch/tvm/ppf_tests/aot_test"
model_name="treelstm"
dynamic_batch_size_estimate=64

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
    opt_level=3)

def get_ansor_log_file(model_name, parameters, pass_context, target):
    batched_execution = pass_context.config["relay.db_batched_execution"]
    dynamic_batch_size_estimate = pass_context.config["relay.db_dynamic_batch_size_estimate"]
    config_str = ("%d_%d_%s") % (batched_execution, dynamic_batch_size_estimate, target)
    model_str = model_name + "_" + "_".join([str(i) for i in parameters])
    file_name = model_str + "_" + config_str + ".log"
    print(file_name)
    log_dir = "ansor_logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir + file_name

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

log_file = get_ansor_log_file("treelstm", [hidden_size], pass_context, target)
def auto_schedule(tune):
    with pass_context:
        tasks, task_weights = auto_scheduler.extract_tasks(mod, weights_dict, target, pass_context,
                                                           include_simple_tasks=True)

        # for idx, task in enumerate(tasks):
            # print("========== Task %d  (workload key: %s, weight: %s) ==========" %
                  # (idx, task.workload_key, task_weights[idx]))
            # print(task.compute_dag)
        if tune:
            measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=100)
            tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=1000,  # change this to 20000 to achieve the best performance
                runner=measure_ctx.runner,
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                # layout_rewrite_option=auto_scheduler.LayoutRewriteOption.NO_REWRITE,
            )
            tuner.tune(tune_option)

def execute():
    with tvm.auto_scheduler.ApplyHistoryBest(log_file):
        with pass_context:
            executor = relay.backend.vm.VMExecutor(mod, device, target)
            executable = executor.compile(params=weights_dict)
            executable.save_to_file(aot_output_directory + "/treelstm.ro",
                                    aot_output_directory + "/treelstm_lib.so")
            exit(0)
            # fin_executor = executor._make_executor(execution_options=execution_options)
            # params_list = []
            # if use_autoscheduler:
                # for tree in trees: params_list += [tree]
            # else:
                # for tree in trees: params_list += weights_list + [tree]

            # executor.vm.set_input("main", batch_size, *params_list)

            fin_executor()
            # iters = 20
            # timeit.timeit(fin_executor, number=iters)
            # print_time(timeit.timeit(fin_executor, number=iters)*1000/iters)

# auto_schedule((not os.path.exists(log_file)))
print("===============================================================================", flush=True)
execute()
