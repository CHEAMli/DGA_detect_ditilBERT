import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNNClassifer(torch.nn.Module):
    def __init__(self):
        super(TextCNNClassifer, self).__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, 768)) for k in (2, 3, 4)])
        # 定义 3 个卷积层，分别提取 2-gram、3-gram、4-gram 特征
        # 输入通道为 1，输出通道为 256，每个卷积核大小为 (k, 768)
        self.dropout = nn.Dropout(0.2)# Dropout，用于防止过拟合，随机丢弃 20% 的神经元
        self.fc = nn.Linear(256 * len((2, 3, 4)), 2) # 全连接层，将卷积后的特征映射到输出类别，2 表示二分类

    def conv_and_pool(self, x, conv):# 卷积 -> ReLU 激活 -> 最大池化
        x = F.relu(conv(x)).squeeze(3) # 移除宽度维度
        x = F.max_pool1d(x, x.size(2)).squeeze(2)# 应用 1D 最大池化并移除多余维度
        return x

    def forward(self, encoding): # 输入为 [batch_size, sequence_length, embedding_dim]
        # only use the first h in the sequence
        out = encoding
        out = out.unsqueeze(1) # 增加通道维度，变为 [batch_size, 1, sequence_length, embedding_dim]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)# 对每个卷积层提取的特征进行拼接
        out = self.dropout(out)# 应用 Dropout，防止过拟合
        out = self.fc(out)# 全连接层，将特征映射到分类输出

        return out


if __name__ == '__main__':
    model = TextCNNClassifer()
    model()

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class TextCNNClassifer(torch.nn.Module):
#     def __init__(self):
#         super(TextCNNClassifer, self).__init__()
#
#         # 定义卷积层，使用不同卷积核大小（2, 3, 4）
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(1, 256, (k, 768)) for k in (2, 3, 4)]
#         )
#
#         # 定义双向LSTM层
#         self.lstm = nn.LSTM(
#             input_size=256 * len((2, 3, 4)),  # 输入特征大小
#             hidden_size=128,                 # LSTM隐藏层大小
#             num_layers=1,                    # LSTM层数
#             bidirectional=True,              # 双向LSTM
#             batch_first=True                 # 输入的维度为 (batch_size, seq_len, feature_size)
#         )
# n
#         # Dropout层，用于正则化
#         self.dropout = nn.Dropout(0.2)
#
#         # 全连接层，输入为双向LSTM的输出（128 * 2），输出为2分类
#         self.fc = nn.Linear(128 * 2, 2)
#
#     def conv_and_pool(self, x, conv):
#         """
#         卷积 + 激活函数 + 最大池化操作
#         """
#         # 卷积操作，激活函数ReLU
#         x = F.relu(conv(x)).squeeze(3)  # 移除第3维(宽度)
#         # 最大池化操作
#         x = F.max_pool1d(x, x.size(2)).squeeze(2)  # 移除最后一维(池化后的步长)
#         return x
#
#     def forward(self, encoding):
#         """
#         前向传播
#         """
#         # 输入的 `encoding`，形状为 [batch_size, seq_len, embedding_dim=768]
#         out = encoding
#
#         # 增加一个通道维度，用于卷积输入
#         out = out.unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
#
#         # 通过每个卷积层
#         out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  # [batch_size, 256 * len(kernels)]
#
#         # 添加一个序列维度，维度调整为 (batch_size, seq_len=1, feature_size)
#         out = out.unsqueeze(1)  # [batch_size, 1, 256 * len(kernels)]
#
#         # 通过双向LSTM层
#         out, _ = self.lstm(out)  # LSTM输出，out: [batch_size, seq_len, hidden_size * 2]
#
#         # 移除序列维度，取LSTM最后一步的输出
#         out = out[:, -1, :]  # [batch_size, hidden_size * 2]
#
#         # Dropout层
#         out = self.dropout(out)
#
#         # 全连接层，映射到2分类
#         out = self.fc(out)  # [batch_size, 2]
#
#         return out
#
#
# if __name__ == '__main__':
#     # 创建模型实例
#     model = TextCNNClassifer()
#
#     # 定义一个随机输入，形状为 [batch_size, seq_len, embedding_dim=768]
#     input_tensor = torch.randn(8, 50, 768)  # 假设batch_size=8, seq_len=50
#
#     # 前向传播
#     output = model(input_tensor)
#
#     # 打印输出形状
#     print(output.shape)  # [8, 2]