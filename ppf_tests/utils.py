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

def get_ansor_log_file(model_name, parameters, pass_context, target):
    target = target.split(' ')[0]
    batched_execution = pass_context.config["relay.db_batched_execution"]
    dynamic_batch_size_estimate = pass_context.config["relay.db_dynamic_batch_size_estimate"]
    config_str = ("%d_%d_%s") % (batched_execution, dynamic_batch_size_estimate, target)
    model_str = model_name + "_" + "_".join([str(i) for i in parameters])
    file_name = model_str + "_" + config_str + ".log"
    log_dir = TVM_HOME + "/ansor_logs/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir + file_name

def get_random_tensor(shape):
    return relay.const(np.random.normal(size=tuple(shape)), dtype='float32').data


def pgo_and_auto_schedule(mod, weights_dict, inputs, batch_size, log_file,
                          target, pass_context, execution_options):
    with pass_context:
        def tune_fn(_tasks, _task_weights, _num_iterations):
            measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=100)
            tuner = auto_scheduler.TaskScheduler(_tasks, _task_weights, load_log_file=log_file)
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=_num_iterations,
                runner=measure_ctx.runner,
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            )
            tuner.tune(tune_option)

        # Extract tasks
        tasks, task_weights = auto_scheduler.extract_tasks(mod, weights_dict, target, pass_context,
                                                           include_simple_tasks=True,
                                                           execution_options=None)

        # Build and execute on the CPU for PGO stats
        tasks, task_weights, executors = auto_scheduler.extract_tasks(mod, weights_dict, target, pass_context,
                                                                      include_simple_tasks=True,
                                                                      execution_options=execution_options)

        dynamic_batch_sizes = [0] * len(tasks)
        executor, fin_executor = executors
        params_list = inputs
        executor.vm.set_input("main", batch_size, *params_list)
        fin_executor()
        stats = executor.vm.get_pgo_stats()
        for i in range(len(tasks)):
            key = tasks[i].compute_dag.workload_key()
            this_stats = stats.get(key, {})
            task_weights[i] = int(this_stats.get("exe_count", 1))
            dynamic_batch_sizes[i] = float(this_stats.get("batch_size", 1.0))

        print(task_weights)
        print(dynamic_batch_sizes)

        # Finally tune ops with updated weights
        tune_fn(tasks, task_weights, 20000)
