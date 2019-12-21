import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, data_type: str = None):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, True, data_type)
        self.gc2 = GraphConvolution(nhid, nclass, True, data_type)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class ImprovedGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(ImprovedGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        pass

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


if __name__ == '__main__':
    pass
