import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from math import exp

VERBOSE = False

class dummy_data:
    def __init__(self, *dims):
        self.dims = dims

    @property
    def shape(self):
        return self.dims

class ParallelPermute(nn.Module):
    '''permutes input vector in a random but fixed way'''

    def __init__(self, dims_in, seed):
        super(ParallelPermute, self).__init__()
        # print('dims in', dims_in)
        # exit()
        self.n_inputs = len(dims_in)
        self.in_channels = [dims_in[i][0] for i in range(self.n_inputs)]

        np.random.seed(seed)
        perm, perm_inv = self.get_random_perm(0)
        self.perm = [perm]
        self.perm_inv = [perm_inv]

        for i in range(1, self.n_inputs):
            perm, perm_inv = self.get_random_perm(i)
            self.perm.append(perm)
            self.perm_inv.append(perm_inv)

    def get_random_perm(self, i):
        perm = np.random.permutation(self.in_channels[i])
        perm_inv = np.zeros_like(perm)
        for i, p in enumerate(perm):
            perm_inv[p] = i

        perm = torch.LongTensor(perm)
        perm_inv = torch.LongTensor(perm_inv)
        return perm, perm_inv

    def forward(self, x, rev=False):
        if not rev:
            return [x[i][:, self.perm[i]] for i in range(self.n_inputs)]
        else:
            return [x[i][:, self.perm_inv[i]] for i in range(self.n_inputs)]

    def jacobian(self, x, rev=False):
        # TODO: use batch size, set as nn.Parameter so cuda() works
        return [0.] * self.n_inputs

    def output_dims(self, input_dims):
        return input_dims

class CrossConvolutions(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0):
        super(CrossConvolutions, self).__init__()
        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        pad_mode = 'zeros'

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.up_conv10 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)

        self.up_conv21 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)

        self.down_conv01 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=2, padding_mode=pad_mode, dilation=1)

        self.down_conv12 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=2, padding_mode=pad_mode, dilation=1)

        self.lr = nn.LeakyReLU(self.leaky_slope)

    def forward(self, x0, x1, x2):
        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)

        out0 = self.conv_scale0_1(y0)
        out1 = self.conv_scale1_1(y1)
        out2 = self.conv_scale2_1(y2)

        y1_up = self.up_conv10(self.upsample(y1))
        y2_up = self.up_conv21(self.upsample(y2))

        y0_down = self.down_conv01(y0)
        y1_down = self.down_conv12(y1)

        out0 = out0 + y1_up
        out1 = out1 + y0_down + y2_up
        out2 = out2 + y1_down

        out0 = out0 * self.gamma0
        out1 = out1 * self.gamma1
        out2 = out2 * self.gamma2
        return out0, out1, out2

class parallel_glow_coupling_layer(nn.Module):
    def __init__(self, dims_in, F_class=CrossConvolutions, F_args={},
                 clamp=5.):
        super(parallel_glow_coupling_layer, self).__init__()
        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp

        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.s1 = F_class(self.split_len1, self.split_len2 * 2, **F_args)
        self.s2 = F_class(self.split_len2, self.split_len1 * 2, **F_args)

    def e(self, s):
        if self.clamp > 0:
            return torch.exp(self.log_e(s))
        else:
            return torch.exp(s)

    def log_e(self, s):
        if self.clamp > 0:
            return self.clamp * 0.636 * torch.atan(s / self.clamp)
        else:
            return s

    def forward(self, x, rev=False):
        x01, x02 = (x[0].narrow(1, 0, self.split_len1),
                    x[0].narrow(1, self.split_len1, self.split_len2))
        x11, x12 = (x[1].narrow(1, 0, self.split_len1),
                    x[1].narrow(1, self.split_len1, self.split_len2))
        x21, x22 = (x[2].narrow(1, 0, self.split_len1),
                    x[2].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            r02, r12, r22 = self.s2(x02, x12, x22)

            s02, t02 = r02[:, :self.split_len1], r02[:, self.split_len1:]
            s12, t12 = r12[:, :self.split_len1], r12[:, self.split_len1:]
            s22, t22 = r22[:, :self.split_len1], r22[:, self.split_len1:]

            y01 = self.e(s02) * x01 + t02
            y11 = self.e(s12) * x11 + t12
            y21 = self.e(s22) * x21 + t22

            r01, r11, r21 = self.s1(y01, y11, y21)

            s01, t01 = r01[:, :self.split_len2], r01[:, self.split_len2:]
            s11, t11 = r11[:, :self.split_len2], r11[:, self.split_len2:]
            s21, t21 = r21[:, :self.split_len2], r21[:, self.split_len2:]
            y02 = self.e(s01) * x02 + t01
            y12 = self.e(s11) * x12 + t11
            y22 = self.e(s21) * x22 + t21

        else:  # names of x and y are swapped!
            r01, r11, r21 = self.s1(x01, x11, x21)

            s01, t01 = r01[:, :self.split_len2], r01[:, self.split_len2:]
            s11, t11 = r11[:, :self.split_len2], r11[:, self.split_len2:]
            s21, t21 = r21[:, :self.split_len2], r21[:, self.split_len2:]

            y02 = (x02 - t01) / self.e(s01)
            y12 = (x12 - t11) / self.e(s11)
            y22 = (x22 - t21) / self.e(s21)

            r02, r12, r22 = self.s2(y02, y12, y22)

            s02, t02 = r02[:, :self.split_len2], r01[:, self.split_len2:]
            s12, t12 = r12[:, :self.split_len2], r11[:, self.split_len2:]
            s22, t22 = r22[:, :self.split_len2], r21[:, self.split_len2:]

            y01 = (x01 - t02) / self.e(s02)
            y11 = (x11 - t12) / self.e(s12)
            y21 = (x21 - t22) / self.e(s22)

        y0 = torch.cat((y01, y02), 1)
        y1 = torch.cat((y11, y12), 1)
        y2 = torch.cat((y21, y22), 1)

        y0 = torch.clamp(y0, -1e6, 1e6)
        y1 = torch.clamp(y1, -1e6, 1e6)
        y2 = torch.clamp(y2, -1e6, 1e6)

        jac0 = torch.sum(self.log_e(s01), dim=(1, 2, 3)) + torch.sum(self.log_e(s02), dim=(1, 2, 3))
        jac1 = torch.sum(self.log_e(s11), dim=(1, 2, 3)) + torch.sum(self.log_e(s12), dim=(1, 2, 3))
        jac2 = torch.sum(self.log_e(s21), dim=(1, 2, 3)) + torch.sum(self.log_e(s22), dim=(1, 2, 3))
        self.last_jac = [jac0, jac1, jac2]

        return [y0, y1, y2]

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


