import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn import metrics
import torch.utils.data.distributed
import time
from collections import defaultdict
import random
from torch.utils.data import Subset
# 注释掉教师模型相关导入（无需CharacterBERT）
# from modeling.character_bert import CharacterBertModel
from dataSet import DGADataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import DistilBertModel, DistilBertConfig

class DistilBERTClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(DistilBERTClassifier, self).__init__()
        # 保持原配置（若需对比，建议与蒸馏版本的学生模型配置一致）
        self.config = DistilBertConfig(
            vocab_size=30522,
            dim=768,
            n_layers=6,
            n_heads=12,
            hidden_dim=3072,
            dropout=0.1,
            attention_dropout=0.1,
            max_position_embeddings=512,
            activation="relu"
        )
        self.distilbert = DistilBertModel(self.config)

        self.pre_classifier = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# 注释掉教师模型定义（无需使用）
# class CharacterBERTClassifier(nn.Module):
#     def __init__(self, character_bert_model, embedding_dim, num_labels=2):
#         super(CharacterBERTClassifier, self).__init__()
#         self.character_bert = character_bert_model
#         self.classifier = nn.Linear(embedding_dim, num_labels)
#
#     def forward(self, char_ids):
#         embeddings, _ = self.character_bert(char_ids)
#         pooled_output = embeddings[:, 0, :]
#         logits = self.classifier(pooled_output)
#         return logits

def create_or_write_file(folder_path, model, file_name, content):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, model) as file:
        file.write(content)

def stratified_random_split(dataset, test_size, random_state=42):
    class_indices = defaultdict(list)
    for i in range(len(dataset)):
        class_indices[dataset[i]['targets'].item()].append(i)
    test_sizes = {label: int(len(indices) * test_size) for label, indices in class_indices.items()}
    train_indices = []
    test_indices = []
    for label, indices in class_indices.items():
        random.seed(random_state)
        random.shuffle(indices)
        test_indices.extend(indices[:test_sizes[label]])
        train_indices.extend(indices[test_sizes[label]:])
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    return train_subset, test_subset

def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize=None)
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)
    plt.tight_layout()
    plt.colorbar()
    thresh = cm.max() / 2.
    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1,1,1) if i == j and cm[i, j] > thresh else (0, 0, 0)
            value = str(cm[j, i])
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
        plt.clf()

def prepare_data():
    dataset = DGADataset()
    print("=======划分训练集、测试集======")
    if os.path.exists("./saved_splits/train_dataset.pt") and os.path.exists("./saved_splits/test_dataset.pt"):
        print("=======加载已保存的划分结果======")
        train_dataset = torch.load("./saved_splits/train_dataset.pt", weights_only=False)
        test_dataset = torch.load("./saved_splits/test_dataset.pt", weights_only=False)
    else:
        print("=======首次划分训练集、测试集======")
        train_dataset, test_dataset = stratified_random_split(dataset, test_size, random_state=42)
        os.makedirs("./saved_splits", exist_ok=True)
        torch.save(train_dataset, "./saved_splits/train_dataset.pt")
        torch.save(test_dataset, "./saved_splits/test_dataset.pt")
        print("=======划分完毕并保存======")
    print('train_dataset:', len(train_dataset))
    print('test_dataset:', len(test_dataset))
    training_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    testing_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return training_loader, testing_loader

# 关键修改1：简化train函数，去掉教师模型、软损失相关参数和逻辑
def train(
    student_model,  # 去掉teacher_model参数
    epochs,
    training_loader,
    testing_loader,
    optimizer,
    criterion_hard,  # 仅保留硬损失
    device
):
    num_batch = len(training_loader.dataset) / batch_size

    for epoch in range(epochs):
        print("==================== epoch %d ====================" % epoch)
        log_context = "==================== epoch %d ====================" % epoch
        # 日志文件夹命名改为"epoch_no_distill_"，区分蒸馏版本
        log_file_path = "./logs/epoch_no_distill_%d/" % epoch
        if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)
        log_file = os.path.join(log_file_path, "训练结果.txt")
        with open(log_file, "w") as file:
            file.write(str(log_context) + "\n")
        student_model.train()

        for bid, data in enumerate(training_loader, 0):
            targets = data['targets'].to(device)
            # 仅保留学生模型输入（去掉教师模型的char_ids）
            token_ids = data['token_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)

            # 去掉教师模型前向传播和软标签损失计算
            # 仅学生模型前向传播
            student_logits = student_model(token_ids, attention_mask)

            # 仅计算硬标签损失（无软损失）
            loss = criterion_hard(student_logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算准确率（与原逻辑一致）
            pred_choice = student_logits.max(1)[1]
            correct = pred_choice.eq(targets).cpu().sum()
            accuracy = correct.item() / float(batch_size)
            if bid%10 == 0:
                print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, bid, num_batch, loss.item(), accuracy))
                with open(log_file, "a") as file:
                    file.write('[%d: %d/%d] train loss: %f accuracy: %f\n' % (epoch, bid, num_batch, loss.item(), accuracy))

        if epoch == epochs - 1:
            # 模型保存路径添加"no_distill_"前缀，区分蒸馏版本
            torch.save(student_model.module.state_dict(), "model/no_distill_compact_distilbert_fp32.pth")
            student_fp16 = student_model.module.half()
            torch.save(student_fp16.state_dict(), "model/no_distill_compact_distilbert_fp16.pth")

        fin_outputs, fin_targets = test(student_model, testing_loader)
        confusion_matrix = metrics.confusion_matrix(fin_targets, fin_outputs)
        label_name = ['white', 'dga']
        draw_confusion_matrix(label_true=fin_targets,
                              label_pred=fin_outputs,
                              label_name=label_name,
                              title="Epoch%d Confusion Matrix (No Distill)" % epoch,
                              pdf_save_path="./logs/epoch_no_distill_%d/Confusion_Matrix.jpg" % epoch,
                              dpi=300)

        tp, fn, fp, tn = confusion_matrix.ravel()
        accuracy = round((tp + tn) / (tp + tn + fp + fn), 4)
        # 处理除零警告（补充优化）
        precision = round(tp / (tp + fp) if (tp + fp) != 0 else 0.0, 4)
        recall_score = round(tp / (tp + fn) if (tp + fn) != 0 else 0.0, 4)
        f1_score = round((2 * precision * recall_score) / (precision + recall_score) if (precision + recall_score) != 0 else 0.0, 4)
        false_positive_rate = round(fn / (tp + fn) if (tp + fn) != 0 else 0.0, 4)
        false_negative_rate = round(fp / (fp + tn) if (fp + tn) != 0 else 0.0, 4)

        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score = {f1_score}")
        print(f"召回率 = {recall_score}")
        print(f"精确率 = {precision}")
        print(f"误报率 = {false_positive_rate}")
        print(f"漏报率 = {false_negative_rate}")

        log_context = "fin_outputs:"+str(fin_outputs)+"\n"
        log_context += "fin_targets:"+str(fin_targets)+"\n"
        log_context += f"Accuracy Score = {accuracy}"+"\n"
        log_context += f"F1 Score = {f1_score}"+"\n"
        log_context += f"召回率 = {recall_score}"+"\n"
        log_context += f"精确率 = {precision}"+"\n"
        log_context += f"误报率 = {false_positive_rate}" + "\n"
        log_context += f"漏报率 = {false_negative_rate}" + "\n"
        create_or_write_file("./logs/epoch_no_distill_%d/" % epoch, "a", "训练结果.txt", str(log_context))

# test函数无需修改（保持与原逻辑一致）
def test(model, testing_loader):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bid, data in enumerate(testing_loader, 0):
            targets = data['targets']
            ids = data['token_ids'].cuda()
            attention_mask = data['attention_mask'].cuda()
            outputs = model(ids, attention_mask)
            outputs = outputs.max(1)[1]
            fin_targets.extend(targets.cpu().numpy())
            fin_outputs.extend(outputs.cpu().numpy())
    return fin_outputs, fin_targets

# 无用函数注释掉（无需使用）
# def label_smoothing(inputs, epsilon=0.1):
#     return ((1 - epsilon) * inputs) + (epsilon / 2)
#
# def loss_fun(outputs, targets):
#     return torch.nn.BCEWithLogitsLoss()(outputs, targets)

if __name__ == '__main__':
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    test_size = 0.3
    seed = 42

    # 仅保留硬损失（去掉软损失相关定义）
    criterion_hard = torch.nn.CrossEntropyLoss()
    # 注释掉软损失和蒸馏相关参数
    # criterion_soft = torch.nn.KLDivLoss(reduction='batchmean')
    # temperature =5
    # a=0.7

    # 注释掉教师模型初始化（无需加载CharacterBERT）
    # character_bert_model = CharacterBertModel.from_pretrained("./pretrained-models/general_character_bert").cuda()
    # character_bert_model.eval()
    # embedding_dim = character_bert_model.config.hidden_size
    # teacher_model = CharacterBERTClassifier(character_bert_model, embedding_dim).cuda()

    # 初始化学生模型（与蒸馏版本配置一致，保证对比公平）
    student_model = DistilBERTClassifier(num_labels=2).cuda()
    gpus = [0]
    # 注释掉教师模型的DataParallel包装
    # teacher_model = torch.nn.DataParallel(teacher_model, device_ids=gpus, output_device=gpus[0])
    student_model = torch.nn.DataParallel(student_model, device_ids=gpus, output_device=gpus[0])

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('Device:', device)

    training_loader, testing_loader = prepare_data()

    # 优化器配置与蒸馏版本一致（保证对比公平）
    optimizer = torch.optim.SGD(student_model.parameters(), lr=1e-02)

    # 调用修改后的train函数（仅传入学生模型和硬损失）
    train(
        student_model=student_model,
        epochs=1,
        training_loader=training_loader,
        testing_loader=testing_loader,
        optimizer=optimizer,
        criterion_hard=criterion_hard,
        device=device
    )
    end_time = time.time()
    print('Took %f second' % (end_time - start_time))

