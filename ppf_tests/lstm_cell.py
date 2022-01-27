import timeit
import numpy as np
import tvm
import tvm.runtime as trt
from tvm import relay

device = tvm.runtime.device("cpu")
target = "llvm"

hidden_size = 32
iterations = 1
batch_size = 2

mod, params = relay.testing.lstm.get_workload(iterations, hidden_size)

lazy_execution=False
coarsened_execution=False
batched_execution=False
scattered_kernels=False
concurrent_execution=True
pass_context, execution_options = relay.backend.vm.create_workflow_configs(lazy_execution=lazy_execution,
                                                                           coarsened_execution=coarsened_execution,
                                                                           batched_execution=batched_execution,
                                                                           scattered_kernels=scattered_kernels,
                                                                           concurrent_execution=concurrent_execution,
                                                                           batch_size=batch_size,
                                                                           opt_level=3)
with pass_context:
    executor = relay.backend.vm.VMExecutor(mod, device, target)
    fin_executor = executor._make_executor(execution_options=execution_options)

    if concurrent_execution:
        params_list = [4 + iterations]
        for i in range(batch_size):
            datas = []
            for i in range(iterations):
                datas.append(tvm.nd.array(np.zeros((1, hidden_size)).astype("float32"), device=tvm.cpu(0)))
            params_list += [params["i2h_weight"], params["i2h_bias"], params["h2h_weight"], params["h2h_bias"]] + datas
        executor.vm.set_input("main", *params_list)
    else:
        datas = []
        for i in range(batch_size * iterations):
            datas.append(tvm.nd.array(np.zeros((1, hidden_size)).astype("float32"), device=tvm.cpu(0)))
        params_list = [params["i2h_weight"], params["i2h_bias"], params["h2h_weight"], params["h2h_bias"]] + datas
        executor.vm.set_input("main", *params_list)

    iters = 1000
    print(timeit.timeit(fin_executor, number=iters)*1000/iters)
