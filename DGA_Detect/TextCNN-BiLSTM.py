import torch
import torch.nn as nn
import torch.nn.functional as F

class CharacterBertTextCNNBiLSTM_GlobalPool(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, dropout_rate):
        super().__init__()

        self.convs = nn.ModuleList([nn.Conv2d(1, 256, (k, embedding_dim)) for k in (2, 3, 4)])
        self.dropout = nn.Dropout(dropout_rate)

        # 双向LSTM
        self.lstm = nn.LSTM(256 * len((2, 3, 4)), hidden_dim, batch_first=True, bidirectional=True)

        # 全局池化替代Attention
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, seq_len, emb_dim]
        conv_out = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], dim=1)

        lstm_out, _ = self.lstm(conv_out.unsqueeze(1))  # [batch, seq_len, hidden*2]

        # 全局平均池化代替Attention机制
        pooled_out = torch.mean(lstm_out, dim=1)

        out = self.fc(pooled_out)

        return out