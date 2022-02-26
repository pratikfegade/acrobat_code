import tvm
from tvm import relay
from tvm.relay import op, var, Var, Function, Clause, PatternConstructor, PatternVar, Match
from tvm.relay import TupleGetItem, Tuple, TensorType, TupleType

def build_ir(input_size, output_size, dtype="float32"):
    ret_type = TensorType(shape=(1, output_size,), dtype=dtype)
    inp = var("input", shape=(1, input_size), dtype=dtype)
    weight = var("weight", shape=(output_size, input_size), dtype=dtype)
    linear = op.nn.dense(inp, weight)
    mean = op.mean(linear)
    output = relay.If(relay.equal(mean, relay.const(0.0, dtype)), op.sigmoid(linear), linear)
    func = relay.Function([weight, inp], output, ret_type)

    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    return mod

print(build_ir(32, 32))
