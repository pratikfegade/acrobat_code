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

weight = tvm.nd.array(np.zeros((256, 256)).astype("float32"), device=tvm.cpu(0))

target = "llvm"
batch_size = 8
lazy_execution=True
coarsened_execution=False
batched_execution=True
scattered_kernels=False
concurrent_execution=False
use_autoscheduler=False
aot_output_directory="/home/ppf/data/ppf/projects/projects/dyn_batch/tvm/ppf_tests/aot_test"
model_name="tdc"
generate_aot_code=True
dynamic_batch_size_estimate=64
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
