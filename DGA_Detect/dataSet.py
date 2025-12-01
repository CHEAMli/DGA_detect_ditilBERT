
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from torch.utils.data import Dataset
from modeling.character_bert import CharacterBertModel
from utils.character_cnn import CharacterIndexer
from transformers import BertTokenizer
from transformers import DistilBertTokenizer
from transformers import DistilBertConfig


class DGADataset(Dataset):
    """
    自定义数据集，用于加载白名单域名和 DGA 域名，并生成适合 CharacterBERT 的输入。
    """

    def __init__(self):
        # 加载白名单域名数据
        with open("./Dataset/train_white.txt", "r", encoding="utf-8") as f:
            self.white = f.readlines()
        # 加载 DGA 域名数据
        with open("./Dataset/train_dga.txt", "r", encoding="utf-8") as f:
            self.dga = f.readlines()

        # 白名单数据长度
        self.white_len = len(self.white)

        # 合并白名单和 DGA 数据
        self.df = self.white + self.dga

        # 初始化 CharacterIndexer，用于将 token 转换为字符索引
        self.character_indexer = CharacterIndexer()

        # 初始化 DistilBERT 的 tokenizer
        self.tokenizer_distilbert = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        #config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
        # 打印配置信息
        #print(config)
    def __getitem__(self, item):
        """
        获取指定索引的数据样本。
        Args:
            item: 数据索引。
        Returns:
            一个字典，包含字符索引张量、目标标签等。
        """
        # 获取原始文本数据并去除首尾空格
        text = self.df[item].strip()

        # 添加特殊标记 [CLS] 和 [SEP]
        tokens = ['[CLS]', *list(text), '[SEP]']

        # 将 token 转换为字符索引，并填充为固定长度的张量
        char_ids = self.character_indexer.as_padded_tensor([tokens], maxlen=64)
        encoding = self.tokenizer_distilbert(
            text,
            padding='max_length',
            truncation=True,
            max_length=64,  # 与 CharacterBERT 的 maxlen 保持一致
            return_tensors='pt'
        )
        token_ids = encoding['input_ids'].squeeze(0)  # 去掉 batch 维度
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 根据索引判断标签（白名单为 0，DGA 为 1）
        if item < self.white_len:
            targets = 0
        else:
            targets = 1

        # 返回包含字符索引、DistilBERT 的 token_ids、attention_mask 和标签的字典
        return {
            'char_ids': char_ids[0],  # 获取第一个样本的字符索引
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'targets': torch.tensor(targets, dtype=torch.long)
        }


    def __len__(self):
        """
        返回数据集的总长度。
        """
        return len(self.df)

if __name__ == '__main__':
    # 创建数据集实例
    dataset = DGADataset()

    # 测试数据集的前 10 个样本
    for i in range(2):
        sample = dataset[i]
        print(f"Sample {i}:")
        print("Char IDs:", sample['char_ids'].shape)  # 输出字符索引的形状
        print("Targets:", sample['targets'])


