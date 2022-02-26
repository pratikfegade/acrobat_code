import timeit
import numpy as np
import tvm
import tvm.runtime as trt
from tvm import relay
from tvm import auto_scheduler

device = tvm.runtime.device("cpu")
target = "llvm"

hidden_size = 32
iterations = 1
batch_size = 20

mod, params = relay.testing.lstm.get_workload(iterations, hidden_size)

lazy_execution=True
coarsened_execution=False
batched_execution=True
scattered_kernels=True
concurrent_execution=False
use_autoscheduler=True
pass_context, execution_options = relay.backend.vm.create_workflow_configs(
    lazy_execution=lazy_execution,
    coarsened_execution=coarsened_execution,
    batched_execution=batched_execution,
    scattered_kernels=scattered_kernels,
    concurrent_execution=concurrent_execution,
    use_autoscheduler=use_autoscheduler,
    batch_size=batch_size,
    opt_level=3)

log_file = "logs/maskrcnn_rtx3070.log"
def auto_schedule():
    with pass_context:
        print("extracting task")
        tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target, pass_context)
        print("extracting task done")

        for idx, task in enumerate(tasks):
            print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
            print(task.compute_dag)

        exit(0)
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=100)

        # tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=2,  # change this to 20000 to achieve the best performance
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

        tuner.tune(tune_option)
auto_schedule()

# with tvm.auto_scheduler.ApplyHistoryBest(log_file):
# with pass_context:
#     executor = relay.backend.vm.VMExecutor(mod, device, target)
#     fin_executor = executor._make_executor(execution_options=execution_options)

#     params_list = []
#     for i in range(batch_size):
#         datas = []
#         for i in range(iterations):
#             datas.append(tvm.nd.array(np.zeros((1, hidden_size)).astype("float32"), device=tvm.cpu(0)))
#         params_list += [params["i2h_weight"], params["i2h_bias"], params["h2h_weight"], params["h2h_bias"]]
#         params_list += datas

#     executor.vm.set_input("main", batch_size, *params_list)

#     iters = 1000
#     print(timeit.timeit(fin_executor, number=iters)*1000/iters)




























# KeyError: 'Traceback (most recent call last):
#   22: TVMFuncCall
#         at /home/ppf/Documents/projects/projects/dyn_batch/tvm/src/runtime/c_runtime_api.cc:475
#   21: tvm::runtime::PackedFunc::CallPacked(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
#         at /home/ppf/Documents/projects/projects/dyn_batch/tvm/include/tvm/runtime/packed_func.h:1151
#   20: std::function<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
#         at /usr/include/c++/11.1.0/bits/std_function.h:560
#   19: _M_invoke
#         at /usr/include/c++/11.1.0/bits/std_function.h:291
#   18: __invoke_r<void, tvm::runtime::TypedPackedFunc<tvm::runtime::Array<tvm::runtime::ObjectRef>(tvm::auto_scheduler::SearchPolicy, int, tvm::auto_scheduler::ProgramMeasurer)>::AssignTypedLambda<tvm::auto_scheduler::<lambda(tvm::auto_scheduler::SearchPolicy, int, tvm::auto_scheduler::ProgramMeasurer)> >(tvm::auto_scheduler::<lambda(tvm::auto_scheduler::SearchPolicy, int, tvm::auto_scheduler::ProgramMeasurer)>, std::string)::<lambda(const tvm::runtime::TVMArgs&, tvm::runtime::TVMRetValue*)>&, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*>
#         at /usr/include/c++/11.1.0/bits/invoke.h:154
#   17: __invoke_impl<void, tvm::runtime::TypedPackedFunc<tvm::runtime::Array<tvm::runtime::ObjectRef>(tvm::auto_scheduler::SearchPolicy, int, tvm::auto_scheduler::ProgramMeasurer)>::AssignTypedLambda<tvm::auto_scheduler::<lambda(tvm::auto_scheduler::SearchPolicy, int, tvm::auto_scheduler::ProgramMeasurer)> >(tvm::auto_scheduler::<lambda(tvm::auto_scheduler::SearchPolicy, int, tvm::auto_scheduler::ProgramMeasurer)>, std::string)::<lambda(const tvm::runtime::TVMArgs&, tvm::runtime::TVMRetValue*)>&, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*>
#         at /usr/include/c++/11.1.0/bits/invoke.h:61
#   16: operator()
#         at /home/ppf/Documents/projects/projects/dyn_batch/tvm/include/tvm/runtime/packed_func.h:1480
#   15: unpack_call<tvm::runtime::Array<tvm::runtime::ObjectRef>, 3, tvm::auto_scheduler::<lambda(tvm::auto_scheduler::SearchPolicy, int, tvm::auto_scheduler::ProgramMeasurer)> >
#         at /home/ppf/Documents/projects/projects/dyn_batch/tvm/include/tvm/runtime/packed_func.h:1421
#   14: run<>
#         at /home/ppf/Documents/projects/projects/dyn_batch/tvm/include/tvm/runtime/packed_func.h:1382
#   13: run<tvm::runtime::TVMMovableArgValueWithContext_>
#         at /home/ppf/Documents/projects/projects/dyn_batch/tvm/include/tvm/runtime/packed_func.h:1382
#   12: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
#         at /home/ppf/Documents/projects/projects/dyn_batch/tvm/include/tvm/runtime/packed_func.h:1382
#   11: run<tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_, tvm::runtime::TVMMovableArgValueWithContext_>
#         at /home/ppf/Documents/projects/projects/dyn_batch/tvm/include/tvm/runtime/packed_func.h:1397
#   10: operator()
#         at /home/ppf/Documents/projects/projects/dyn_batch/tvm/src/auto_scheduler/search_policy/search_policy.cc:111
#   9: tvm::auto_scheduler::SketchPolicyNode::ContinueSearchOneRound(int, tvm::auto_scheduler::ProgramMeasurer)
#         at /home/ppf/Documents/projects/projects/dyn_batch/tvm/src/auto_scheduler/search_policy/sketch_policy.cc:265
#   8: tvm::auto_scheduler::ProgramMeasurerNode::Measure(tvm::auto_scheduler::SearchTask const&, tvm::auto_scheduler::SearchPolicy const&, tvm::runtime::Array<tvm::auto_scheduler::MeasureInput, void> const&, int)
#         at /home/ppf/Documents/projects/projects/dyn_batch/tvm/src/auto_scheduler/measure.cc:255
#   7: tvm::auto_scheduler::ProgramMeasurerNode::SilentMeasure(tvm::auto_scheduler::SearchTask const&, tvm::runtime::Array<tvm::auto_scheduler::MeasureInput, void> const&, tvm::runtime::Array<tvm::auto_scheduler::MeasureResult, void>*)
#         at /home/ppf/Documents/projects/projects/dyn_batch/tvm/src/auto_scheduler/measure.cc:319
#   6: tvm::auto_scheduler::RPCRunnerNode::Run(tvm::runtime::Array<tvm::auto_scheduler::MeasureInput, void> const&, tvm::runtime::Array<tvm::auto_scheduler::BuildResult, void> const&, int)
#         at /home/ppf/Documents/projects/projects/dyn_batch/tvm/src/auto_scheduler/measure.cc:178
#   5: tvm::runtime::TVMRetValue tvm::runtime::PackedFunc::operator()<tvm::runtime::Array<tvm::auto_scheduler::MeasureInput, void> const&, tvm::runtime::Array<tvm::auto_scheduler::BuildResult, void> const&, tvm::runtime::String&, tvm::runtime::String&, int&, int&, int&, int&, int&, int&, int&, double&, bool&, int&>(tvm::runtime::Array<tvm::auto_scheduler::MeasureInput, void> const&, tvm::runtime::Array<tvm::auto_scheduler::BuildResult, void> const&, tvm::runtime::String&, tvm::runtime::String&, int&, int&, int&, int&, int&, int&, int&, double&, bool&, int&) const
#   4: std::function<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const
#         at /usr/include/c++/11.1.0/bits/std_function.h:560
#   3: _M_invoke
#         at /usr/include/c++/11.1.0/bits/std_function.h:291
#   2: __invoke_r<void, TVMFuncCreateFromCFunc(TVMPackedCFunc, void*, TVMPackedCFuncFinalizer, void**)::<lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>&, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*>
#         at /usr/include/c++/11.1.0/bits/invoke.h:154
#   1: __invoke_impl<void, TVMFuncCreateFromCFunc(TVMPackedCFunc, void*, TVMPackedCFuncFinalizer, void**)::<lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)>&, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*>
#         at /usr/include/c++/11.1.0/bits/invoke.h:61
#   0: operator()
#         at /home/ppf/Documents/projects/projects/dyn_batch/tvm/src/runtime/c_runtime_api.cc:526
#   File "/home/ppf/Documents/projects/projects/dyn_batch/tvm/python/tvm/_ffi/_ctypes/packed_func.py", line 81, in cfun
#     rv = local_pyfunc(*pyargs)
#   File "/home/ppf/Documents/projects/projects/dyn_batch/tvm/python/tvm/auto_scheduler/measure.py", line 1271, in rpc_runner_run
#     [
#   File "/home/ppf/Documents/projects/projects/dyn_batch/tvm/python/tvm/auto_scheduler/measure.py", line 1275, in <listcomp>
#     prepare_runner_args(inp, build_res),
#   File "/home/ppf/Documents/projects/projects/dyn_batch/tvm/python/tvm/auto_scheduler/measure.py", line 848, in prepare_runner_args
#     task_input_buffer = get_task_input_buffer(inp.task.workload_key, tensor_name)
#   File "/home/ppf/Documents/projects/projects/dyn_batch/tvm/python/tvm/auto_scheduler/search_task.py", line 357, in get_task_input_buffer
#     input_table = TASK_INPUT_BUFFER_TABLE["default"]
# KeyError: \'default\''
