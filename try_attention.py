import torch 
import torch.nn as nn

class Attention(nn.Module): 
    def init(self, dmodel, dk): 
        super(Attention, self).init() 
        self.dk = dk 
        self.W = nn.Linear(dmodel, dk) 
        self.V = nn.Linear(dmodel, dk) 
        self.a = nn.Linear(dmodel, 1)

    def forward(self, Q, K, V):
        a = self.a(Q)
        a = torch.tanh(a + self.W(Q) + K)
        a = self.V(a)
        a = torch.softmax(a, dim=-1)
        return a * V
    
#使用 Attention 模块
attention = Attention(dmodel=64, dk=32) 
Q = torch.randn(1, 1, 64) 
K = torch.randn(1, 32, 32) 
V = torch.randn(1, 32, 32)

output = attention(Q, K, V) 
print(output.shape) 
# torch.Size([1, 1, 32]) ```
#16