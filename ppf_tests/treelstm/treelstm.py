from network import Network
from tvm import relay, ir
from tvm.relay import op, var, Var, Function, Clause, PatternConstructor, PatternVar, Match
from tvm.relay import TupleGetItem, Tuple, TensorType, TupleType, Let

def unique_var(name, shape=None, dtype=None):
    unique_var.ctr += 1
    var_ = var(name + str(unique_var.ctr), shape=shape, dtype=dtype)
    return var_
unique_var.ctr = 0

class Linear(Network):
    def initialize(self, input_size, output_size, dtype="float32"):
        self.w = self.weight(unique_var("linear_weight", shape=(output_size, input_size), dtype=dtype))
        self.b = self.weight(unique_var("linear_bias", shape=(1, output_size,), dtype=dtype))

    def build_impl(self, input_size, output_size, dtype="float32"):
        self.ret_type = TensorType(shape=(1, output_size,), dtype=dtype)
        x = self.input(unique_var("linear_input", shape=(1, input_size), dtype=dtype))
        return op.add(op.nn.dense(x, self.w), self.b)

def lam(names, func):
    args = [unique_var(name) for name in names]
    return Function(args, func(*args))

def fuse_ops(body_fn, body_type, arg_list):
    param_list = []
    tmp_list = []
    for i in range(len(arg_list)):
        param_list.append(relay.Var("param" + str(i)))
        tmp_list.append(relay.Var("tmp" + str(i)))
        i += 1
    body = body_fn(*param_list)
    attr_dict = { "Primitive": 1 }
    attrs = ir.make_node("DictAttrs", **attr_dict)
    func = relay.Function(param_list, body, body_type, attrs=attrs)
    return func(*arg_list)


class RNNCell(Network):
    def initialize(self, input_size, memory_size, dtype="float32"):
        self.ilinear = self.create_sub_network(Linear(input_size=input_size,
                                                      output_size=memory_size, name="ilinear"))
        self.hlinear = self.create_sub_network(Linear(input_size=memory_size,
                                                      output_size=memory_size, name="hlinear"))

    def build_impl(self, input_size, memory_size, dtype="float32"):
        t = TensorType(shape=(1, memory_size), dtype=dtype)
        i = self.input(unique_var("lstmcell_input", shape=(1, input_size), dtype=dtype))
        h = self.input(unique_var("lstmcell_input", shape=(1, input_size), dtype=dtype))

        ilinear = self.ilinear(self, False, i)
        hlinear = self.hlinear(self, False, h)

        return op.sigmoid(ilinear + hlinear)

class LSTMCellMultipleChildren(Network):
    def initialize(self, input_size, memory_size, dtype="float32"):
        self.ilinear = self.create_sub_network(Linear(input_size=input_size,
                                                      output_size=memory_size * 3, name="ilinear"))
        self.hlinear = self.create_sub_network(Linear(input_size=memory_size,
                                                      output_size=memory_size * 3, name="hlinear"))
        self.fhlinear = self.create_sub_network(Linear(input_size=memory_size, output_size=memory_size))

        self.fxlinear_w = self.weight(unique_var("linear_weight", shape=(memory_size, input_size), dtype=dtype))
        self.fxlinear_b = self.weight(unique_var("linear_bias", shape=(1, memory_size,), dtype=dtype))


    def build_impl(self, input_size, memory_size, dtype="float32"):
        t = TensorType(shape=(1, memory_size), dtype=dtype)
        i = self.input(unique_var("lstmcell_input", shape=(1, input_size), dtype=dtype))
        c = self.input(Var("lstmcell_children", self.l(TupleType([t, t]))))
        sum = lam(["x", "y"], lambda x, y: x + y)
        child_h_sum = self.p.foldl(sum,
                                   op.zeros(shape=(1, memory_size), dtype=dtype),
                                   self.p.map(lam(["z"], lambda z: TupleGetItem(z, 1)), c))
        ioux = self.ilinear(self, False, i)
        iouh = self.hlinear(self, False, child_h_sum)
        iou = ioux + iouh

        #####################
        iu = fuse_ops(
            lambda _iou: (op.sigmoid(TupleGetItem(op.split(_iou, 3, axis=1).astuple(), 0)) *
                          op.tanh(TupleGetItem(op.split(_iou, 3, axis=1).astuple(), 2))),
            TensorType(shape=(1, memory_size), dtype=dtype),
            [iou]
        )
        #####################




        #####################
        fx = relay.Var("fx")
        fx_value = fuse_ops(
            lambda _iou, _w, _b: op.add(op.nn.dense(op.sigmoid(TupleGetItem(
                op.split(_iou, 3, axis=1).astuple(), 0)), _w), _b),
            TensorType(shape=(1, memory_size), dtype=dtype),
            [iou, self.fxlinear_w, self.fxlinear_b]
        )
        #####################

        fh = self.fhlinear
        def foreach_children(children):
            f = op.sigmoid(fh(self, False, TupleGetItem(children, 1)) + fx)
            return f * TupleGetItem(children, 0)
        # c = Let(fx, fx_value, self.p.foldl(sum, iu, self.p.map(lam(["z"], foreach_children), c)))
        c = self.p.foldl(sum, iu, self.p.map(lam(["z"], foreach_children), c))



        #####################
        h = fuse_ops(
            lambda _iou, _c: op.sigmoid(TupleGetItem(op.split(_iou, 3, axis=1).astuple(), 1)) * op.tanh(_c),
            TensorType(shape=(1, memory_size), dtype=dtype),
            [iou, c]
        )
        #####################

        return Let(fx, fx_value, Tuple([c, h]))

class LSTMCellOneChild(Network):
    def initialize(self, input_size, memory_size, dtype="float32"):
        self.ilinear = self.create_sub_network(Linear(input_size=input_size,
                                                      output_size=memory_size * 3, name="ilinear"))
        self.hlinear = self.create_sub_network(Linear(input_size=memory_size,
                                                      output_size=memory_size * 3, name="hlinear"))
        self.fhlinear = self.create_sub_network(Linear(input_size=memory_size, output_size=memory_size))

        self.fxlinear_w = self.weight(unique_var("linear_weight", shape=(memory_size, input_size), dtype=dtype))
        self.fxlinear_b = self.weight(unique_var("linear_bias", shape=(1, memory_size,), dtype=dtype))


    def build_impl(self, input_size, memory_size, dtype="float32"):
        t = TensorType(shape=(1, memory_size), dtype=dtype)
        i = self.input(unique_var("lstmcell_input", shape=(1, input_size), dtype=dtype))
        c = self.input(Var("lstmcell_child", TupleType([t, t])))
        child_h_sum = TupleGetItem(c, 1)
        ioux = self.ilinear(self, False, i)
        iouh = self.hlinear(self, False, child_h_sum)
        iou = ioux + iouh

        #####################
        iu = fuse_ops(
            lambda _iou: (op.sigmoid(TupleGetItem(op.split(_iou, 3, axis=1).astuple(), 0)) *
                          op.tanh(TupleGetItem(op.split(_iou, 3, axis=1).astuple(), 2))),
            TensorType(shape=(1, memory_size), dtype=dtype),
            [iou]
        )
        #####################

        #####################
        fx = relay.Var("fx")
        fx_value = fuse_ops(
            lambda _iou, _w, _b: op.add(op.nn.dense(op.sigmoid(TupleGetItem(
                op.split(_iou, 3, axis=1).astuple(), 0)), _w), _b),
            TensorType(shape=(1, memory_size), dtype=dtype),
            [iou, self.fxlinear_w, self.fxlinear_b]
        )
        #####################

        fh = self.fhlinear
        def foreach_children(children):
            f = op.sigmoid(fh(self, False, TupleGetItem(children, 1)) + fx)
            return f * TupleGetItem(children, 0)
        c = iu + foreach_children(c)

        #####################
        h = fuse_ops(
            lambda _iou, _c: op.sigmoid(TupleGetItem(op.split(_iou, 3, axis=1).astuple(), 1)) * op.tanh(_c),
            TensorType(shape=(1, memory_size), dtype=dtype),
            [iou, c]
        )
        #####################

        # return Tuple([c, o * op.tanh(c)])
        return Tuple([c, h])

class LSTMEncoder(Network):
    def build_impl(self, input_size, memory_size, dtype="float32"):
        l = self.input(Var("l", self.l(TensorType(shape=(1, input_size), dtype=dtype))))
        cell = LSTMCell(input_size=input_size, memory_size=memory_size, dtype=dtype)
        return self.p.foldl(lam(["c", "x"], lambda c, x: cell(x, False, self.p.cons(c, self.p.nil()))),
                            Tuple([op.zeros(shape=(1, memory_size), dtype=dtype),
                                   op.zeros(shape=(1, memory_size), dtype=dtype)]), l)

class LSTMTransformer(Network):
    def build_impl(self, input_size, memory_size, dtype="float32"):
        l = self.input(Var("l", self.l(TensorType(shape=(1, input_size), dtype=dtype))))
        def f(c, x):
            cell = LSTMCell(input_size=input_size, memory_size=memory_size, dtype=dtype)
            o = cell(x, self.p.cons(c, self.p.nil()))
            return Tuple([o, TupleGetItem(o, 1)])
        res = self.p.map_accuml(lam(["c", "x"], f),
                                 Tuple([op.zeros(shape=(1, memory_size), dtype=dtype),
                                        op.zeros(shape=(1, memory_size), dtype=dtype)]),
                                 l)
        return Tuple([TupleGetItem(TupleGetItem(res, 0), 1), TupleGetItem(res, 1)])

class TreeLSTM(Network):
    def initialize(self, input_size, memory_size, dtype="float32"):
        self.lstm_cell = self.create_sub_network(
            LSTMCellMultipleChildren(input_size=input_size,
                                     memory_size=memory_size,
                                     dtype=dtype, name="lstm_cell"))

    def build_impl(self, input_size, memory_size, dtype="float32"):
        t = TensorType(shape=(1, memory_size), dtype=dtype)
        self.ret_type = TupleType([t, t])
        tree_type = self.tree(TensorType(shape=(1, input_size), dtype=dtype))
        t = self.input(Var("tlstm_input", tree_type))
        i = Var("i", TensorType(shape=(1, input_size), dtype=dtype))
        c = Var("c", self.l(tree_type))
        children_tensors = self.p.map(lam(["x"], lambda *inputs: self(self, True, *inputs)), c)
        rose_case = Clause(PatternConstructor(self.rose(), [PatternVar(i), PatternVar(c)]),
                           self.lstm_cell(self, False, i, children_tensors))
        return Match(t, [rose_case])

class BiLSTM(Network):
    def build_impl(self, input_size, memory_size, dtype="float32"):
        l = self.input(Var("l", self.l(TensorType(shape=(1, input_size), dtype=dtype))))
        def LSTM(l):
            return LSTMTransformer(input_size=input_size,
                                   memory_size=memory_size,
                                   dtype=dtype)(l)
        fwd = LSTM(l)
        rev = LSTM(self.p.rev(l))
        lhs = op.concatenate([TupleGetItem(fwd, 0), TupleGetItem(rev, 0)], axis=1)
        t = TensorType(shape=(1, memory_size), dtype=dtype)
        x = Var("x", TupleType([t, t])) # cannot infer here
        rhs = self.p.map(Function([x], op.concatenate([TupleGetItem(x, 0),
                                                       TupleGetItem(x, 1)],
                                                      axis=1)),
                         self.p.zip(TupleGetItem(fwd, 1), TupleGetItem(rev, 1)))
        return Tuple([lhs, rhs])
