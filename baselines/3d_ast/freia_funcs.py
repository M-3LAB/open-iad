'''This Code is based on the FrEIA Framework, source: https://github.com/VLL-HD/FrEIA
It is a assembly of the necessary modules/functions from FrEIA that are needed for our purposes.'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable

import config as c
from utils import *

VERBOSE = False


class dummy_data:
    def __init__(self, *dims):
        self.dims = dims

    @property
    def shape(self):
        return self.dims


class permute_layer(nn.Module):
    '''permutes input vector in a random but fixed way'''

    def __init__(self, dims_in, seed):
        super(permute_layer, self).__init__()
        self.in_channels = dims_in[0][0]

        np.random.seed(seed)
        self.perm = np.random.permutation(self.in_channels)
        np.random.seed()

        self.perm_inv = np.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i

        self.perm = torch.LongTensor(self.perm)
        self.perm_inv = torch.LongTensor(self.perm_inv)

    def forward(self, x, rev=False):
        if not rev:
            return [x[0][:, self.perm]]
        else:
            return [x[0][:, self.perm_inv]]

    def jacobian(self, x, rev=False):
        return 0.

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims


class F_conv(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=c.channels_hidden_teacher,
                 kernel_size=3, leaky_slope=0.1,
                 batch_norm=False):
        super(F_conv, self).__init__()

        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        pad_mode = ['zeros', 'replicate'][1]
        self.leaky_slope = leaky_slope
        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(in_channels, channels_hidden,
                               kernel_size=kernel_size, padding=pad, padding_mode=pad_mode,
                               bias=not batch_norm)
        self.conv2 = nn.Conv2d(channels_hidden, channels,
                               kernel_size=kernel_size, padding=pad, padding_mode=pad_mode,
                               bias=not batch_norm)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if c.use_gamma:
            out = out * self.gamma
            return out
        return out


class glow_coupling_layer_cond(nn.Module):
    def __init__(self, dims_in, F_class=F_conv, F_args={},
                 clamp=5., cond_dim=0, cat_dim=1, split_len=None):
        super(glow_coupling_layer_cond, self).__init__()
        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.cat_dim = cat_dim
        if split_len is None:
            self.split_len1 = channels // 2
            self.split_len2 = channels - channels // 2
        else:
            self.split_len1 = split_len
            self.split_len2 = channels - split_len

        self.clamp = clamp
        self.cond_dim = cond_dim
        self.use_cond = (self.cond_dim > 0)

        self.s1 = F_class(self.split_len1 + self.cond_dim, self.split_len2 * 2, **F_args)
        self.s2 = F_class(self.split_len2 + self.cond_dim, self.split_len1 * 2, **F_args)

    def e(self, s):
        return torch.exp(self.log_e(s))

    def log_e(self, s):
        if c.clamp > 0:
            return self.clamp * 0.636 * torch.atan(s / self.clamp)
        else:
            return s

    def forward(self, x, rev=False):
        x1, x2 = (x[0].narrow(self.cat_dim, 0, self.split_len1),
                  x[0].narrow(self.cat_dim, self.split_len1, self.split_len2))
        if self.use_cond:
            cond = x[1]

        if not rev:
            r2 = self.s2(torch.cat([x2, cond], dim=self.cat_dim) if self.use_cond else x2)
            s2, t2 = torch.split(r2, r2.shape[self.cat_dim] // 2, self.cat_dim)
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(torch.cat([y1, cond], dim=self.cat_dim) if self.use_cond else y1)
            s1, t1 = torch.split(r1, r1.shape[self.cat_dim] // 2, self.cat_dim)
            y2 = self.e(s1) * x2 + t1

        else:  # names of x and y are swapped!
            r1 = self.s1(torch.cat([x1, cond], dim=self.cat_dim) if self.use_cond else x1)
            s1, t1 = torch.split(r1, r1.shape[self.cat_dim] // 2, self.cat_dim)
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(torch.cat([y2, cond], dim=self.cat_dim) if self.use_cond else y2)
            s2, t2 = torch.split(r2, r2.shape[self.cat_dim] // 2, self.cat_dim)
            y1 = (x1 - t2) / self.e(s2)

        y = torch.cat((y1, y2), self.cat_dim)
        y = torch.clamp(y, -1e6, 1e6)

        jac = torch.sum(self.log_e(s1), dim=self.cat_dim) + torch.sum(self.log_e(s2), dim=self.cat_dim)
        self.last_jac = jac
        return [y]

    def jacobian(self, x, rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims


class Node:
    '''The Node class represents one transformation in the graph, with an
    arbitrary number of in- and outputs.'''

    def __init__(self, inputs, module_type, module_args, name=None):
        self.inputs = inputs
        self.outputs = []
        self.module_type = module_type
        self.module_args = module_args

        self.input_dims, self.module = None, None
        self.computed = None
        self.computed_rev = None
        self.id = None

        if name:
            self.name = name
        else:
            self.name = hex(id(self))[-6:]
        for i in range(255):
            exec('self.out{0} = (self, {0})'.format(i))

    def build_modules(self, verbose=VERBOSE):
        ''' Returns a list with the dimension of each output of this node,
        recursively calling build_modules of the nodes connected to the input.
        Use this information to initialize the pytorch nn.Module of this node.
        '''

        if not self.input_dims:  # Only do it if this hasn't been computed yet
            self.input_dims = [n.build_modules(verbose=verbose)[c]
                               for n, c in self.inputs]
            try:
                self.module = self.module_type(self.input_dims,
                                               **self.module_args)
            except Exception as e:
                print('Error in node %s' % (self.name))
                raise e

            if verbose:
                print("Node %s has following input dimensions:" % (self.name))
                for d, (n, c) in zip(self.input_dims, self.inputs):
                    print("\t Output #%i of node %s:" % (c, n.name), d)
                print()

            self.output_dims = self.module.output_dims(self.input_dims)
            self.n_outputs = len(self.output_dims)

        return self.output_dims

    def run_forward(self, op_list):
        '''Determine the order of operations needed to reach this node. Calls
        run_forward of parent nodes recursively. Each operation is appended to
        the global list op_list, in the form (node ID, input variable IDs,
        output variable IDs)'''

        if not self.computed:

            # Compute all nodes which provide inputs, filter out the
            # channels you need
            self.input_vars = []
            for i, (n, c) in enumerate(self.inputs):
                self.input_vars.append(n.run_forward(op_list)[c])
                # Register youself as an output in the input node
                n.outputs.append((self, i))

            # All outputs could now be computed
            self.computed = [(self.id, i) for i in range(self.n_outputs)]
            op_list.append((self.id, self.input_vars, self.computed))

        # Return the variables you have computed (this happens mulitple times
        # without recomputing if called repeatedly)
        return self.computed

    def run_backward(self, op_list):
        '''See run_forward, this is the same, only for the reverse computation.
        Need to call run_forward first, otherwise this function will not
        work'''

        assert len(self.outputs) > 0, "Call run_forward first"
        if not self.computed_rev:

            # These are the input variables that must be computed first
            output_vars = [(self.id, i) for i in range(self.n_outputs)]

            # Recursively compute these
            for n, c in self.outputs:
                n.run_backward(op_list)

            # The variables that this node computes are the input variables
            # from the forward pass
            self.computed_rev = self.input_vars
            op_list.append((self.id, output_vars, self.computed_rev))

        return self.computed_rev


class InputNode(Node):
    '''Special type of node that represents the input data of the whole net (or
    ouput when running reverse)'''

    def __init__(self, *dims, name='node'):
        self.name = name
        self.data = dummy_data(*dims)
        self.outputs = []
        self.module = None
        self.computed_rev = None
        self.n_outputs = 1
        self.input_vars = []
        self.out0 = (self, 0)

    def build_modules(self, verbose=VERBOSE):
        return [self.data.shape]

    def run_forward(self, op_list):
        return [(self.id, 0)]


class OutputNode(Node):
    '''Special type of node that represents the output of the whole net (of the
    input when running in reverse)'''

    class dummy(nn.Module):

        def __init__(self, *args):
            super(OutputNode.dummy, self).__init__()

        def __call__(*args):
            return args

        def output_dims(*args):
            return args

    def __init__(self, inputs, name='node'):
        self.module_type, self.module_args = self.dummy, {}
        self.output_dims = []
        self.inputs = inputs
        self.input_dims, self.module = None, None
        self.computed = None
        self.id = None
        self.name = name

        for c, inp in enumerate(self.inputs):
            inp[0].outputs.append((self, c))

    def run_backward(self, op_list):
        return [(self.id, 0)]


class ReversibleGraphNet(nn.Module):
    '''This class represents the invertible net itself. It is a subclass of
    torch.nn.Module and supports the same methods. The forward method has an
    additional option 'rev', whith which the net can be computed in reverse.'''

    def __init__(self, node_list, ind_in=None, ind_out=None, verbose=False, n_jac=1):
        '''node_list should be a list of all nodes involved, and ind_in,
        ind_out are the indexes of the special nodes InputNode and OutputNode
        in this list.'''
        super(ReversibleGraphNet, self).__init__()

        # Gather lists of input and output nodes
        if ind_in is not None:
            if isinstance(ind_in, int):
                self.ind_in = list([ind_in])
            else:
                self.ind_in = ind_in
        else:
            self.ind_in = [i for i in range(len(node_list))
                           if isinstance(node_list[i], InputNode)]
            assert len(self.ind_in) > 0, "No input nodes specified."
        if ind_out is not None:
            if isinstance(ind_out, int):
                self.ind_out = list([ind_out])
            else:
                self.ind_out = ind_out
        else:
            self.ind_out = [i for i in range(len(node_list))
                            if isinstance(node_list[i], OutputNode)]
            assert len(self.ind_out) > 0, "No output nodes specified."

        self.return_vars = []
        self.input_vars = []

        # Assign each node a unique ID
        self.node_list = node_list
        for i, n in enumerate(node_list):
            n.id = i

        # Recursively build the nodes nn.Modules and determine order of
        # operations
        ops = []
        for i in self.ind_out:
            node_list[i].build_modules(verbose=verbose)
            node_list[i].run_forward(ops)

        # create list of Pytorch variables that are used
        variables = set()
        for o in ops:
            variables = variables.union(set(o[1] + o[2]))
        self.variables_ind = list(variables)

        self.indexed_ops = self.ops_to_indexed(ops)

        self.module_list = nn.ModuleList([n.module for n in node_list])
        self.variable_list = [Variable(requires_grad=True) for v in variables]

        # Find out the order of operations for reverse calculations
        ops_rev = []
        for i in self.ind_in:
            node_list[i].run_backward(ops_rev)
        self.indexed_ops_rev = self.ops_to_indexed(ops_rev)
        self.n_jac = n_jac

    def ops_to_indexed(self, ops):
        '''Helper function to translate the list of variables (origin ID, channel),
        to variable IDs.'''
        result = []

        for o in ops:
            try:
                vars_in = [self.variables_ind.index(v) for v in o[1]]
            except ValueError:
                vars_in = -1

            vars_out = [self.variables_ind.index(v) for v in o[2]]

            # Collect input/output nodes in separate lists, but don't add to
            # indexed ops
            if o[0] in self.ind_out:
                self.return_vars.append(self.variables_ind.index(o[1][0]))
                continue
            if o[0] in self.ind_in:
                self.input_vars.append(self.variables_ind.index(o[1][0]))
                continue

            result.append((o[0], vars_in, vars_out))

        # Sort input/output variables so they correspond to initial node list
        # order
        self.return_vars.sort(key=lambda i: self.variables_ind[i][0])
        self.input_vars.sort(key=lambda i: self.variables_ind[i][0])

        return result

    def forward(self, x, rev=False):
        '''Forward or backward computation of the whole net.'''
        if rev:
            use_list = self.indexed_ops_rev
            input_vars, output_vars = self.return_vars, self.input_vars
        else:
            use_list = self.indexed_ops
            input_vars, output_vars = self.input_vars, self.return_vars

        if isinstance(x, (list, tuple)):
            assert len(x) == len(input_vars), (
                f"Got list of {len(x)} input tensors for "
                f"{'inverse' if rev else 'forward'} pass, but expected "
                f"{len(input_vars)}."
            )
            for i in range(len(input_vars)):
                self.variable_list[input_vars[i]] = x[i]
        else:
            assert len(input_vars) == 1, (f"Got single input tensor for "
                                          f"{'inverse' if rev else 'forward'} "
                                          f"pass, but expected list of "
                                          f"{len(input_vars)}.")
            self.variable_list[input_vars[0]] = x

        for o in use_list:
            try:
                results = self.module_list[o[0]]([self.variable_list[i]
                                                  for i in o[1]], rev=rev)
            except TypeError:
                raise RuntimeError("Are you sure all used Nodes are in the "
                                   "Node list?")
            for i, r in zip(o[2], results):
                self.variable_list[i] = r
            # self.variable_list[o[2][0]] = self.variable_list[o[1][0]]

        out = [self.variable_list[output_vars[i]]
               for i in range(len(output_vars))]
        if len(out) == 1:
            return out[0]
        else:
            return out

    def jacobian(self, x=None, rev=False, run_forward=True):
        '''Compute the jacobian determinant of the whole net.'''
        jacobian = [0.] * self.n_jac

        if rev:
            use_list = self.indexed_ops_rev
        else:
            use_list = self.indexed_ops

        if run_forward:
            if x is None:
                raise RuntimeError("You need to provide an input if you want "
                                   "to run a forward pass")
            self.forward(x, rev=rev)

        for o in use_list:
            try:
                node_jac = self.module_list[o[0]].jacobian(
                    [self.variable_list[i] for i in o[1]], rev=rev
                )
                node_jac = [node_jac] if not isinstance(node_jac, list) else node_jac
                for i_j, jac in enumerate(node_jac):
                    jacobian[i_j] += jac
            except TypeError:
                raise RuntimeError("Are you sure all used Nodes are in the "
                                   "Node list?")
        return jacobian
