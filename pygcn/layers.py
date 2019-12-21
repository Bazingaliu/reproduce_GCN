import math

import torch
import numpy as np

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, data_type: str = None):
        super(GraphConvolution, self).__init__()
        self.data_type = data_type

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(2 * in_features if data_type == "elliptic" else in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)

        # Inductive settings
        support = torch.spmm(adj, input)
        output = torch.mm(torch.cat([support, input], dim=-1) if self.data_type == "elliptic" else support, self.weight)
        # output = output / torch.norm(output, dim=1).reshape([output.shape[0], 1])

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphRandomConvolution(Module):
    def __init__(self, ):
        super(GraphRandomConvolution, self).__init__()

    def forward(self, input, adj):
        pass