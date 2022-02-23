import numpy as np
from random import randrange
import tvm
from tvm import relay
from tvm.runtime.container import ADT
from tvm.relay.prelude import Prelude

from treelstm import TreeLSTM, LSTMCell, Linear
from network import copy_var

class RoseTree:
    def __init__(self, head, children):
        self.head = head
        self.children = children

    def __str__(self):
        return "Tree(" + str(self.head) + ", " + str(self.children) + ")"

    def __repr__(self):
        return self.__str__()

    def fmap(self, f):
        return RoseTree(f(self.head), [x.fmap(f) for x in self.children])

    def size(self):
        return 1 + sum([x.size() for x in self.children])

# creates relay list from a list
def from_list(p, l, t):
    if len(l) == 0:
        return ADT(p.mod.get_type("List")[2].tag, [])
    else:
        return ADT(p.mod.get_type("List")[1].tag, [l[0], from_list(p, l[1:], t)])

def from_tree(p, rt, t):
    return ADT(p.mod.get_type("Tree")[1].tag,
               [rt.head,
                from_list(p, [from_tree(p, x, t) for x in rt.children], t)])


def forward(tree, inputs):
    children = [forward(x, inputs) for x in tree.children]
    return RoseTree(inputs[tree.idx], children)


def initialize_tlstm(input_size, memory_size):
    tlstm = TreeLSTM(input_size=input_size, memory_size=memory_size, name="treelstm")
    mod = tlstm.mod
    tlstm_func = mod[tlstm.f]
    tlstm_gv = tlstm.f
    gv = relay.GlobalVar("main")
    main_params = [copy_var(v) for v in tlstm_func.params]
    mod[gv] = relay.Function(main_params, tlstm_gv(*main_params), tlstm_func.ret_type)
    return tlstm, mod, tlstm.p

def get_random_tensor(shape):
    return relay.const(np.random.normal(size=tuple(shape)), dtype='float32').data

def generate_random_tree(num_nodes, tensor_shape):
    assert num_nodes > 0
    if num_nodes == 1:
        return RoseTree(get_random_tensor(tensor_shape), [])

    num_nodes -= 1
    l_nodes = randrange(num_nodes)
    r_nodes = num_nodes - l_nodes
    if l_nodes == 0 or r_nodes == 0:
        return RoseTree(get_random_tensor(tensor_shape),
                        [generate_random_tree(num_nodes, tensor_shape)])
    else:
        return RoseTree(get_random_tensor(tensor_shape),
                        [generate_random_tree(l_nodes, tensor_shape),
                         generate_random_tree(r_nodes, tensor_shape)])

def generate_random_trees(num_nodes, batch_size, tensor_shape, prelude):
    trees = [generate_random_tree(num_nodes, tensor_shape) for i in range(batch_size)]
    return [from_tree(prelude, tree,
                      relay.TensorType(tensor_shape, dtype='float32')) for tree in trees]
