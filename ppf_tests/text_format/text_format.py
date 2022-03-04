import timeit
import numpy as np
import tvm
import tvm.runtime as trt
from tvm import relay
from tvm import auto_scheduler

mod = tvm.IRModule()
p = relay.Prelude(mod)
mod._import("/home/ppf/Documents/projects/projects/dyn_batch/tvm/ppf_tests/text_format/text_format.rly")

device = tvm.runtime.device("cpu")
target = "llvm"

weight = tvm.nd.array(np.zeros((256, 256)).astype("float32"), device=tvm.cpu(0))

lazy_execution=False
coarsened_execution=False
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
    batch_size=1,
    dynamic_batch_size_estimate=1,
    opt_level=3)

def execute():
    with pass_context:
        executor = relay.backend.vm.VMExecutor(mod, device, target)
        fin_executor = executor._make_executor(execution_options=execution_options)

        exit(0)
        params_list = [weight]
        params_list += datas

        executor.vm.set_input("main", batch_size, *params_list)
        iters = 1000
        print(timeit.timeit(fin_executor, number=iters)*1000/iters)

print("===============================================================================", flush=True)
execute()
