import torch
import math
d_model=32
seq_len=6
position = torch.arange(0, seq_len).unsqueeze(1)
A = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model))
print((position*A).shape)

pe = torch.zeros(seq_len, d_model)
#create a vector of shape (seq len)
position = torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model)) 
pe[:, 0::2] = torch.sin(position*div_term)
pe[:, 1::2] = torch.cos(position*div_term)

pe = pe.unsqueeze(0) # (1, seq_len, d_model)
print(pe.shape)