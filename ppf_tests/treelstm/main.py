import timeit
import numpy as np
import tvm
from tvm import relay
from tvm import auto_scheduler
from converter import initialize_tlstm, generate_random_trees, get_random_tensor

device = tvm.runtime.device("cpu")
target = "llvm"

hidden_size = 32
batch_size = 5
num_nodes = 6
lazy_execution=True
coarsened_execution=True
batched_execution=True
scattered_kernels=True
concurrent_execution=False
use_autoscheduler=False

tlstm, mod, prelude = initialize_tlstm(hidden_size, hidden_size)
mod = tvm.relay.transform.RemoveUnusedFunctions(batched_execution=batched_execution)(mod)
params = tlstm.all_weights()

trees = generate_random_trees(num_nodes, batch_size, (1, hidden_size), prelude)
# exit(0)

param_tensors = [get_random_tensor(tuple([int(i) for i in param.type_annotation.shape])) for param in params]

pass_context, execution_options = relay.backend.vm.create_workflow_configs(
    lazy_execution=lazy_execution,
    coarsened_execution=coarsened_execution,
    batched_execution=batched_execution,
    scattered_kernels=scattered_kernels,
    concurrent_execution=concurrent_execution,
    use_autoscheduler=use_autoscheduler,
    batch_size=batch_size,
    opt_level=3)
with pass_context:
    executor = relay.backend.vm.VMExecutor(mod, device, target)
    fin_executor = executor._make_executor(execution_options=execution_options)

    params_list = []
    for tree in trees: params_list += param_tensors + [tree]

    executor.vm.set_input("main", batch_size, *params_list)

    fin_executor()
    # iters = 1000
    # print(timeit.timeit(fin_executor, number=iters)*1000/iters)
