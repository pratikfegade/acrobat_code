import timeit
import numpy as np
import tvm
import tvm.runtime as trt
from tvm import relay
from tvm import auto_scheduler
from utils import get_ansor_log_file

device = tvm.runtime.device("cpu")
target = "llvm"

hidden_size = 256
iterations = 1
batch_size = 20

mod, params = relay.testing.lstm.get_workload(iterations, hidden_size)
print(mod)

lazy_execution=True
coarsened_execution=False
batched_execution=True
scattered_kernels=True
concurrent_execution=False
dynamic_batch_size_estimate=64
use_autoscheduler=False
pass_context, execution_options = relay.backend.vm.create_workflow_configs(
    lazy_execution=lazy_execution,
    coarsened_execution=coarsened_execution,
    batched_execution=batched_execution,
    scattered_kernels=scattered_kernels,
    concurrent_execution=concurrent_execution,
    use_autoscheduler=use_autoscheduler,
    batch_size=batch_size,
    dynamic_batch_size_estimate=dynamic_batch_size_estimate,
    opt_level=3)

log_file = get_ansor_log_file(model_name, [hidden_size], pass_context, target)
def auto_schedule(tune):
    with pass_context:
        # print("extracting task")
        tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target, pass_context,
                                                           include_simple_tasks=True)
        # print("extracting task done")

        for idx, task in enumerate(tasks):
            print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
            print(task.compute_dag)
        # exit(0)
        if tune:
            measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=100)
            tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
            tune_option = auto_scheduler.TuningOptions(
                num_measure_trials=20,  # change this to 20000 to achieve the best performance
                runner=measure_ctx.runner,
                measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
                # layout_rewrite_option=auto_scheduler.LayoutRewriteOption.NO_REWRITE,
            )
            tuner.tune(tune_option)

def execute():
    with tvm.auto_scheduler.ApplyHistoryBest(log_file):
        with pass_context:
            executor = relay.backend.vm.VMExecutor(mod, device, target)
            fin_executor = executor._make_executor(execution_options=execution_options)

            params_list = []
            for i in range(batch_size):
                datas = []
                for i in range(iterations):
                    datas.append(tvm.nd.array(np.zeros((1, hidden_size)).astype("float32"), device=tvm.cpu(0)))
                params_list += [params["i2h_weight"], params["i2h_bias"], params["h2h_weight"], params["h2h_bias"]]
                params_list += datas

            executor.vm.set_input("main", batch_size, *params_list)
            # exit(0)
            # print("Executing")
            # fin_executor()
            iters = 1000
            print(timeit.timeit(fin_executor, number=iters)*1000/iters)

auto_schedule(False)
print("===============================================================================", flush=True)
execute()
