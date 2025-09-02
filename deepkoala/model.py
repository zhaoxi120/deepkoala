import torch, torch.nn as nn, torch.nn.functional as F

class GRUClassifier(nn.Module):
    def __init__(self, hidden:int, layers:int, n_cls:int):
        super().__init__()
        self.embed = nn.Embedding(22, 16, padding_idx=0)
        self.rnn   = nn.GRU(16, hidden, num_layers=layers)
        self.ffn   = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.cls   = nn.Linear(hidden, n_cls)
    def forward(self, x, lens):
        x = self.embed(x).permute(1,0,2).contiguous()
        x,_ = self.rnn(x)
        idx = torch.arange(lens.size(0), device=lens.device)
        x  = x[lens, idx]
        return self.cls(self.ffn(x))