import numpy as np
import random
from random import randrange
random.seed(0)
import tvm
from tvm import relay
from tvm.runtime.container import ADT
from tvm.relay.prelude import Prelude
from utils import get_random_tensor

class RoseTree:
    idx = 0

    def __init__(self, head, children):
        self.head = head
        self.children = children
        self.idx = RoseTree.idx
        RoseTree.idx += 1

    def __str__(self):
        return "Tree(" + str(self.idx) + ", " + str(self.children) + ")"

    def __repr__(self):
        return self.__str__()

    def fmap(self, f):
        return RoseTree(f(self.head), [x.fmap(f) for x in self.children])

    def size(self):
        return 1 + sum([x.size() for x in self.children])

    def printTree(self, level=0):
        if len(self.children) == 0:
            print(' ' * 4 * level + '-> ' + str(self.idx))
        elif len(self.children) == 1:
            print(' ' * 4 * level + '-> ' + str(self.idx) + ' -> ' + str(self.children[0].idx))
        else:
            self.children[1].printTree(level + 1)
            print(' ' * 4 * level + '-> ' + str(self.idx))
            self.children[0].printTree(level + 1)

# creates relay list from a list
def from_list(mod, l):
    if len(l) == 0:
        return ADT(mod.get_type("List")[2].tag, [])
    else:
        return ADT(mod.get_type("List")[1].tag, [l[0], from_list(mod, l[1:])])

def from_tree_treelstm(mod, rt, t):
    return ADT(mod.get_type("Tree")[1].tag,
               [rt.head,
                from_list(mod, [from_tree_treelstm(mod, x, t) for x in rt.children])])

def from_tree_mvrnn(mod, rt):
    if len(rt.children) == 0:
        return ADT(mod.get_type("MVTree")[2].tag, [rt.head[0], rt.head[1]])
    else:
        return ADT(mod.get_type("MVTree")[1].tag,
                   [from_tree_mvrnn(mod, rt.children[0]),
                    from_tree_mvrnn(mod, rt.children[1])])

def forward(tree, inputs):
    children = [forward(x, inputs) for x in tree.children]
    return RoseTree(inputs[tree.idx], children)

def generate_complete_tree(height, data_fn):
    assert height > 0
    if height == 1:
        return RoseTree(data_fn(height), [])

    return RoseTree(data_fn(height),
                    [generate_complete_tree(height - 1, data_fn),
                     generate_complete_tree(height - 1, data_fn)])

def generate_complete_treelstm_trees(tree_height, batch_size, tensor_shape, mod):
    def data_fn(height):
        return get_random_tensor(tensor_shape)
    trees = [generate_complete_tree(tree_height, data_fn) for i in range(batch_size)]
    return [from_tree_treelstm(mod, tree, relay.TensorType(tensor_shape, dtype='float32'))
            for tree in trees]

def generate_complete_mvrnn_trees(tree_height, batch_size, hidden_size, mod):
    def data_fn(height):
        if height == 1: return [get_random_tensor((hidden_size, hidden_size)),
                                get_random_tensor((1, hidden_size))]
        else: return []
    trees = [generate_complete_tree(tree_height, data_fn) for i in range(batch_size)]
    return [from_tree_mvrnn(mod, tree) for tree in trees]

def generate_random_tensor_lists(batch_size, tensor_shape, mod):
    lists = [[get_random_tensor(tensor_shape) for j in range(random.randrange(0, 20))] for i in range(batch_size)]
    return [from_list(mod, l) for l in lists]
