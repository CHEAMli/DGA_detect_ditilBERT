import torch
import torch.nn as nn
a = torch.tensor([1,2])
emb = nn.Embedding(4,4)
print(emb(a))
print(torch.cuda.is_available())