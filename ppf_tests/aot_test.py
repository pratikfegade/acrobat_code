import aot
import timeit
import numpy as np
import tvm
import tvm.runtime as trt
from tvm import relay
from tvm import auto_scheduler

device = tvm.runtime.device("cpu")
target = "llvm"

hidden_size = 256
iterations = 1
batch_size = 1

mod, params = relay.testing.lstm.get_workload(iterations, hidden_size)

lazy_execution=True
coarsened_execution=True
batched_execution=False
scattered_kernels=False
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

def execute():
    with pass_context:
        executor = relay.backend.vm.VMExecutor(mod, device, target)
        fin_executor = executor._make_executor(execution_options=execution_options)
        exit(0)

        # params_list = []
        # for i in range(batch_size):
        #     datas = []
        #     for i in range(iterations):
        #         datas.append(tvm.nd.array(np.zeros((1, hidden_size)).astype("float32"), device=tvm.cpu(0)))
        #     params_list += [params["i2h_weight"], params["i2h_bias"], params["h2h_weight"], params["h2h_bias"]]
        #     params_list += datas

        # executor.vm.set_input("main", batch_size, *params_list)
        # # exit(0)
        # # print("Executing")
        # # fin_executor()
        # iters = 1000
        # print(timeit.timeit(fin_executor, number=iters)*1000/iters)

print("===============================================================================", flush=True)
execute()
