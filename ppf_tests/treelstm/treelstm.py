from network import Network
from tvm import relay
from tvm.relay import op, var, Var, Function, Clause, PatternConstructor, PatternVar, Match
from tvm.relay import TupleGetItem, Tuple, TensorType, TupleType

class Linear(Network):
    def build_impl(self, input_size, output_size, dtype="float32"):
        self.ret_type = TensorType(shape=(1, output_size,), dtype=dtype)
        x = self.input(var("linear_input", shape=(1, input_size), dtype=dtype))
        w = self.weight(var("linear_weight", shape=(output_size, input_size), dtype=dtype))
        b = self.weight(var("linear_bias", shape=(1, output_size,), dtype=dtype))
        return op.add(op.nn.dense(x, w), b)

def lam(names, func):
    args = [Var(name) for name in names]
    return Function(args, func(*args))

class LSTMCell(Network):
    def build_impl(self, input_size, memory_size, dtype="float32"):
        t = TensorType(shape=(1, memory_size), dtype=dtype)
        i = self.input(var("lstmcell_input", shape=(1, input_size), dtype=dtype))
        c = self.input(Var("lstmcell_children", self.l(TupleType([t, t]))))
        sum = lam(["x", "y"], lambda x, y: x + y)
        child_h_sum = self.p.foldl(sum,
                                   op.zeros(shape=(1, memory_size), dtype=dtype),
                                   self.p.map(lam(["z"], lambda z: TupleGetItem(z, 1)), c))
        ioux = Linear(input_size=input_size, output_size=memory_size * 3, name="ilinear")(self, i)
        iouh = Linear(input_size=memory_size, output_size=memory_size * 3, name="hlinear")(self, child_h_sum)
        iou = ioux + iouh
        i, o, u = op.split(iou, 3, axis=1)
        i, o, u = op.sigmoid(i), op.sigmoid(o), op.tanh(u)

        fx = Linear(input_size=input_size, output_size=memory_size)(self, i)
        fh = Linear(input_size=memory_size, output_size=memory_size)

        def foreach_children(children):
            f = op.sigmoid(fh(self, TupleGetItem(children, 1)) + fx)
            return f * TupleGetItem(children, 0)
        c = self.p.foldl(sum, i * u, self.p.map(lam(["z"], foreach_children), c))
        return Tuple([c, o * op.tanh(c)])

    # def build_impl(self, input_size, memory_size, dtype="float32"):
        # t = TensorType(shape=(1, memory_size), dtype=dtype)
        # i = self.input(var("lstmcell_input", shape=(1, input_size), dtype=dtype))
        # c = self.input(Var("lstmcell_children", self.l(TupleType([t, t]))))
        # sum = lam(["x", "y"], lambda x, y: x + y)
        # child_h_sum = self.p.foldl(sum,
                                   # op.zeros(shape=(1, memory_size), dtype=dtype),
                                   # self.p.map(lam(["z"], lambda z: TupleGetItem(z, 1)), c))
        # iou = Linear(input_size=memory_size, output_size=memory_size * 3, name="hlinear")(self, child_h_sum)
        # i, o, u = op.split(iou, 3, axis=1)
        # fh = Linear(input_size=memory_size, output_size=memory_size)

        # def foreach_children(children):
            # f = op.sigmoid(fh(self, TupleGetItem(children, 1)))
            # return f * TupleGetItem(children, 0)
        # c = self.p.foldl(sum, i * u, self.p.map(lam(["z"], foreach_children), c))
        # return Tuple([c, o])

class LSTMEncoder(Network):
    def build_impl(self, input_size, memory_size, dtype="float32"):
        l = self.input(Var("l", self.l(TensorType(shape=(1, input_size), dtype=dtype))))
        cell = LSTMCell(input_size=input_size, memory_size=memory_size, dtype=dtype)
        return self.p.foldl(lam(["c", "x"], lambda c, x: cell(x, self.p.cons(c, self.p.nil()))),
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
    def build_impl(self, input_size, memory_size, dtype="float32"):
        t = TensorType(shape=(1, memory_size), dtype=dtype)
        self.ret_type = TupleType([t, t])
        tree_type = self.tree(TensorType(shape=(1, input_size), dtype=dtype))
        t = self.input(Var("tlstm_input", tree_type))
        i = Var("i", TensorType(shape=(1, input_size), dtype=dtype))
        c = Var("c", self.l(tree_type))
        cell = LSTMCell(input_size=input_size, memory_size=memory_size, dtype=dtype, name="lstm_cell")
        rose_case = Clause(PatternConstructor(self.rose(), [PatternVar(i), PatternVar(c)]),
                           cell(self, i, self.p.map(lam(["x"], lambda *inputs: self(self, *inputs)), c)))
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
