import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
import os
# 导入你的模型结构和数据集类
from train import DistilBERTClassifier  # 压缩后的模型类
from dataSet import DGADataset  # 你的数据集类
from torch.utils.data import Subset
from collections import defaultdict
import random


# --------------------------
# 1. 加载测试数据（与训练时保持一致）
# --------------------------
def load_test_data(batch_size=128, test_size=0.3, random_state=42):
    """加载与训练时相同的测试集"""
    dataset = DGADataset()
    # 若已保存划分好的测试集，直接加载（推荐，确保与训练时一致）
    if os.path.exists("./x_saved_splits/test_dataset.pt"):
        print("加载已保存的测试集...")
        test_dataset = torch.load("./x_saved_splits/test_dataset.pt")
    else:
        # 若未保存，重新划分（需与训练时的划分逻辑一致）
        print("重新划分测试集（与训练时逻辑一致）...")
        class_indices = defaultdict(list)
        for i in range(len(dataset)):
            class_indices[dataset[i]['targets'].item()].append(i)
        test_indices = []
        for label, indices in class_indices.items():
            random.seed(random_state)
            random.shuffle(indices)
            test_size_i = int(len(indices) * test_size)
            test_indices.extend(indices[:test_size_i])
        test_dataset = Subset(dataset, test_indices)
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试时无需打乱
        num_workers=0,
        drop_last=False  # 保留最后一个不完整批次
    )
    return test_loader


# --------------------------
# 2. 模型测试函数
# --------------------------
def test_compressed_model(model_path, device, test_loader):
    model = DistilBERTClassifier(num_labels=2)
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    # 处理FP16模型
    if "fp16" in model_path.lower():
        model = model.half()  # 模型参数转为FP16

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            # 关键修正：input_ids强制为Long类型，attention_mask按需转为FP16
            input_ids = batch['token_ids'].to(device, dtype=torch.long)  # 必须是整数类型
            attention_mask = batch['attention_mask'].to(device)

            # 仅转换attention_mask的精度（与模型匹配）
            if "fp16" in model_path.lower():
                attention_mask = attention_mask.half()

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(batch['targets'].cpu().numpy())

    # 计算指标（保持不变）
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["white", "dga"])

    return {
        "accuracy": accuracy, "f1": f1,
        "confusion_matrix": cm, "classification_report": report
    }

# --------------------------
# 3. 主函数：运行测试并输出结果
# --------------------------
if __name__ == "__main__":
    # 配置参数
    compressed_model_path = "model/compact_distilbert_fp16.pth"  # 压缩模型路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128  # 与训练时的batch_size一致

    # 步骤1：加载测试数据（与原始模型测试用的测试集相同）
    test_loader = load_test_data(batch_size=batch_size)

    # 步骤2：测试压缩后的模型
    metrics = test_compressed_model(
        model_path=compressed_model_path,
        device=device,
        test_loader=test_loader
    )

    # 步骤3：输出测试结果
    print("\n===== 压缩后模型的准确率测试结果 =====")
    print(f"准确率（Accuracy）: {metrics['accuracy']:.4f}")
    print(f"F1值: {metrics['f1']:.4f}")
    print("\n混淆矩阵:")
    print(metrics['confusion_matrix'])
    print("\n分类报告:")
    print(metrics['classification_report'])

    # （可选）与原始模型对比
    # 若需对比，可加载原始模型（如未压缩的DistilBERT），用相同流程测试后比较指标差异