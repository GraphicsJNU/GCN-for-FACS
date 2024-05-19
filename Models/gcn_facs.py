import torch
from torch_geometric.nn import ChebConv
from torch.nn import Module, ModuleList, Linear
from Models.layers import get_activation_layer, get_normalization_layer

class GraphEncoder(Module):
    def __init__(self, v_cnt, in_dim, out_dim, K, ttype=torch.float32, 
                 normalization="Group Norm", activation="ReLU", bias=False):
        super(GraphEncoder, self).__init__()

        channels = [in_dim, 128, 64, 32, out_dim]
        self.channels = channels
        self.v_cnt = v_cnt
        self.gcns = ModuleList()
        self.norms = ModuleList()
        self.activation = get_activation_layer(activation)
        for i in range(len(channels) - 2):
            self.gcns.append(ChebConv(channels[i], channels[i + 1], K, bias=True).type(ttype))
            self.norms.append(get_normalization_layer(normalization, channels[i + 1], ttype=ttype))

        self.last_linear = Linear(v_cnt * channels[-2], channels[-1], bias=True).type(ttype)

    def forward(self, x, edge_index):
        for i in range(len(self.gcns)):
            x = self.gcns[i](x, edge_index)
            x = self.norms[i](x)
            x = self.activation(x)
        
        x = self.last_linear(x.reshape(-1, self.v_cnt * self.channels[-2]))

        return x
    
class GCNFACS(Module):
    def __init__(self, v_cnt, in_dim, out_dim, K, ttype=torch.float32, 
                 normalization="Group Norm", activation="ReLU", bias=False):
        super(GCNFACS, self).__init__()
        self.encoder = GraphEncoder(v_cnt, in_dim, out_dim, K, ttype, 
                                    normalization, activation, bias)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.encoder(x, edge_index)
        
        return z