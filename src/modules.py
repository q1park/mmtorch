import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class NonLinear(nn.Module):
    def __init__(self, d_in, d_ff, d_out, dropout=0., gain=1.):
        super(NonLinear, self).__init__()
        self.w_1 = nn.Linear(d_in, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, x_mask=None):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class EmbeddingNonLinear(nn.Module):
    def __init__(self, size, d_out):
        super(EmbeddingNonLinear, self).__init__()
        self.layer = nn.Sequential(
            nn.Embedding(size, d_out, padding_idx=0), 
            NonLinear(d_out, 2*d_out, d_out)
        )
        
    def forward(self, x, x_mask=None):
        return self.layer(x)
    
class MergeLayer(nn.Module):
    def __init__(self, merges):
        super(MergeLayer, self).__init__()
        self.merges = merges

    def forward(self, x, x_mask=None):
        return {k:torch.cat([x[var] for var in v], dim=-1) for k,v in self.merges.items()}
    
class FlattenLayer(nn.Module):
    def __init__(self, flat):
        super(FlattenLayer, self).__init__()
        self.flat = flat

    def forward(self, x):
        nb = list(x.values())[0].size(0)
        return {k:v.view(nb, -1) for k,v in x.items()}

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRU(nn.Module):
    def __init__(self, d, n_layers, d_hid):
        super(GRU, self).__init__()
        self.d = d
        self.d_hid = d_hid
        self.n_layers = n_layers
        self.rnn = nn.GRU(input_size=d, hidden_size=d_hid, num_layers=n_layers, batch_first=True)

    def forward(self, x, x_mask):
        batch_size = x.size(0)
        x_len = [x.item() if x.item()>=1 else 1 for x in x_mask.sum(dim=1)]
        x_pack = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        output_pack, hidden = self.rnn(x_pack, torch.randn(self.n_layers, batch_size, self.d_hid))
        output_pad, _ = pad_packed_sequence(output_pack, batch_first=True)
        output_pad = output_pad*x_mask[:,:max(x_len)].unsqueeze(-1)
        hidden = hidden.transpose(0, 1)
        return torch.cat([output_pad, hidden], dim=1).mean(dim=1, keepdims=True)
    
class EmbeddingGRU(nn.Module):
    def __init__(self, size, d, n_layers, d_hid):
        super(EmbeddingGRU, self).__init__()
        self.embedding = nn.Embedding(size, d, padding_idx=0)
        self.gru = GRU(d, n_layers, d_hid)
        
    def forward(self, x, x_mask):
        x = self.embedding(x)
        x = self.gru(x, x_mask)
        return x
    
class GroupLayer(nn.Module):
    def __init__(self, modules):
        super(GroupLayer, self).__init__()
        self.blocks = nn.ModuleDict(OrderedDict(modules))
        self.requires_mask = set((GRU, EmbeddingGRU))
        
    def forward(self, x, x_mask=None):
        out = {}
        for k,v in self.blocks.items():
            if type(v) in self.requires_mask:
                out[k] = v(x[k], x['seq_mask'])
            else:
                out[k] = v(x[k])
        return out
    
class GroupModel(nn.Module):
    def __init__(self, layers):
        super(GroupModel, self).__init__()
        self.layers = nn.ModuleDict(OrderedDict(layers))

    def forward(self, x):
        for i, (k,v) in enumerate(self.layers.items()):
            x = v(x)
        return x['pred']