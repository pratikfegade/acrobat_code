import tvm
from tvm import te
from tvm import relay
from tvm.relay import GlobalVar
from tvm.relay.transform import gradient
from tvm.relay.prelude import Prelude

def test_global_function():
    m = tvm.IRModule()
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.Var("x", t)
    d = GlobalVar("double")
    m[d] = relay.Function([x], relay.sin(x), t)
    y = relay.Var("y", t)
    q = GlobalVar("q")
    m[q] = relay.Function([y], d(y))
    g = GlobalVar("grad")
    m = tvm.relay.transform.InferType()(m)
    m[g] = tvm.relay.transform.gradient(q, m)
    m = tvm.relay.transform.InferType()(m)
    m = tvm.relay.transform.PartialEvaluate()(m)
    m = tvm.relay.transform.DeadCodeElimination(inline_once=False)(m)

    # back_func = m[g]
    # print(g, back_func)
    print('YUMMA')
    print(m)

test_global_function()
exit(0)


mod = tvm.IRModule()
p = Prelude(mod)

shape = (256,)
dtype = "float32"
t = relay.TensorType(shape, dtype)

# rnn_cell_body
hidden_var = relay.Var("hidden", t)
input_var = relay.Var("input", t)
rnn_cell = relay.Function([hidden_var, input_var], relay.tanh(hidden_var + input_var))
rnn_cell_var = GlobalVar("rnn_cell")
mod[rnn_cell_var] = rnn_cell


lt = mod.get_type('List')[0](t)
init_var = relay.Var("x", t)
sentence_var = relay.Var("x", lt)
foldl = mod.get_global_var('foldl')
rnn_body = foldl(rnn_cell_var, init_var, sentence_var)
rnn_func = relay.Function([init_var, sentence_var], rnn_body)
rnn_func_var = GlobalVar("rnn_func")
mod[rnn_func_var] = rnn_func

rnn_grad_var = GlobalVar("rnn_grad")

mod = tvm.relay.transform.InferType()(mod)

mod[rnn_grad_var] = tvm.relay.transform.gradient(rnn_func_var, mod)
mod = tvm.relay.transform.InferType()(mod)

rnn_grad = mod[rnn_grad_var]
print(rnn_grad)
