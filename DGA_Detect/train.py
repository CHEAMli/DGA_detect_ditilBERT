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
from modeling.character_bert import CharacterBertModel
from dataSet import DGADataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from transformers import DistilBertModel,DistilBertConfig

class DistilBERTClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(DistilBERTClassifier, self).__init__()
        # 核心配置：减小维度+减少层数（参数仅为原模型的1/4左右）
        self.config = DistilBertConfig(
            vocab_size=30522,  # 保持原词汇表大小
            dim=768,  # 隐藏层维度从768→384（减半）
            n_layers=6,  # Transformer层数从6→3（减半）
            n_heads=12,  # 注意力头数从12→6（与384维度匹配：384/6=64）
            hidden_dim=3072,  # 前馈层维度从3072→1536（同步减半）
            dropout=0.1,  # 保持原dropout率
            attention_dropout=0.1,
            max_position_embeddings=512,  # 保持最大序列长度
            activation="relu"  # 与原激活函数一致
        )
        # 基于压缩配置初始化DistilBERT

        self.distilbert = DistilBertModel(self.config)

        # 分类头（与压缩后的维度匹配）
        self.pre_classifier = nn.Linear(768, 768)  # 输入维度改为384
        self.classifier = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        # DistilBERT 需要 input_ids 和 attention_mask
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs[0]  # (batch_size, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (batch_size, dim) 取第一个token的输出 [CLS]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
class CharacterBERTClassifier(nn.Module):
    def __init__(self, character_bert_model, embedding_dim, num_labels=2):
        super(CharacterBERTClassifier, self).__init__()
        self.character_bert = character_bert_model
        self.classifier = nn.Linear(embedding_dim, num_labels)

    def forward(self, char_ids):
        # 假设 encode 函数返回 (embeddings, _)
        embeddings, _ = self.character_bert(char_ids)  # embeddings: (batch_size, seq_len, embedding_dim)
        # 取第一个 token 的嵌入作为代表（类似于 [CLS]）
        # 这里假设第一个 token 是 [CLS]，如果不是，需要调整
        pooled_output = embeddings[:, 0, :]  # (batch_size, embedding_dim)
        logits = self.classifier(pooled_output)
        return logits

"""
def encode(char_ids):
    #batch_size, seq_len, max_chars_per_token = char_ids.size()
    #char_ids_reshaped = char_ids.view(batch_size * seq_len, max_chars_per_token)
    with torch.no_grad():
        embeddings_for_batch, _ = character_bert_model(char_ids)  # 假设 character_bert_model 可以直接接受 char_ids
    #embeddings_for_batch = embeddings_for_batch.view(batch_size, seq_len, -1)
    return embeddings_for_batch, _
"""
def create_or_write_file(folder_path, model, file_name, content):
    # 检查文件夹是否存在，不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # 构建文件路径
    file_path = os.path.join(folder_path, file_name)

    # 写入文件
    with open(file_path, model) as file:
        file.write(content)

def stratified_random_split(dataset, test_size, random_state=42):
    # 统计每个类别的样本索引
    class_indices = defaultdict(list)
    for i in range(len(dataset)):
        class_indices[dataset[i]['targets'].item()].append(i)

    # 计算每个类别需要划分的样本数量
    test_sizes = {label: int(len(indices) * test_size) for label, indices in class_indices.items()}

    # 随机划分样本索引
    train_indices = []
    test_indices = []
    for label, indices in class_indices.items():
        # 设置随机种子以确保可重复性
        random.seed(random_state)
        random.shuffle(indices)

        # 划分样本索引
        test_indices.extend(indices[:test_sizes[label]])
        train_indices.extend(indices[test_sizes[label]:])

    # 创建子集对象
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
            color = (1,1,1) if i == j and cm[i, j] > thresh else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = str(cm[j, i])
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    #plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
        # 清除当前的图形窗口
        plt.clf()

def prepare_data():
    dataset = DGADataset()
    sample = dataset[0]

    print("=======划分训练集、测试集======")
    # 检查是否已存在保存的划分结果
    if os.path.exists("./saved_splits/train_dataset.pt") and os.path.exists("./saved_splits/test_dataset.pt"):
        print("=======加载已保存的划分结果======")
        train_dataset = torch.load("./saved_splits/train_dataset.pt",weights_only=False)
        test_dataset = torch.load("./saved_splits/test_dataset.pt",weights_only=False)
    else:
        print("=======首次划分训练集、测试集======")
        train_dataset, test_dataset = stratified_random_split(dataset, test_size, random_state=42)

        # 保存划分结果
        os.makedirs("./saved_splits", exist_ok=True)  # 确保目录存在
        torch.save(train_dataset, "./saved_splits/train_dataset.pt")
        torch.save(test_dataset, "./saved_splits/test_dataset.pt")
        print("=======划分完毕并保存======")
    print('train_dataset:', len(train_dataset))
    print('test_dataset:', len(test_dataset))

    training_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    testing_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    return training_loader, testing_loader

def train(
    teacher_model,
    student_model,
    epochs,
    training_loader,
    testing_loader,
    optimizer,
    criterion_hard,
    criterion_soft,
    temperature,
    device,
    α
):
    num_batch = len(training_loader.dataset) / batch_size

    for epoch in range(epochs):
        print("==================== epoch %d ====================" % epoch)
        log_context = "==================== epoch %d ====================" % epoch
        log_file_path = "./logs/epoch_distill_%d/" % epoch
        if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)
        log_file = os.path.join(log_file_path, "训练结果.txt")
        with open(log_file, "w") as file:
            file.write(str(log_context) + "\n")
        student_model.train()
        teacher_model.eval()  # 教师模型在训练期间保持评估模式

        for bid, data in enumerate(training_loader, 0):
            targets = data['targets'].to(device)
            # 获取教师模型的输入
            char_ids = data['char_ids'].to(device)

            # 获取学生模型的输入
            token_ids = data['token_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)

            with torch.no_grad():
                # 使用 CharacterBERT 的 encode 函数获取嵌入
               # embeddings_teacher, _ = encode(char_ids)  # embeddings_teacher: (batch_size, seq_len, embedding_dim)
                # 使用教师分类器得到 logits
                teacher_logits = teacher_model(char_ids)  # (batch_size, num_labels)

            # 学生模型前向传播
            student_logits = student_model(token_ids, attention_mask)  # (batch_size, num_labels)

            # 计算硬标签损失
            loss_hard = criterion_hard(student_logits, targets)
            # 计算软标签损失
            # 将教师和学生的 logits 转换为概率分布
            teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
            student_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=1)
            loss_soft = criterion_soft(student_probs, teacher_probs) * (temperature ** 2)
            # 总损失  通过损失权重α调整软硬损失的比例
            loss = loss_hard + loss_soft
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算准确率
            pred_choice = student_logits.max(1)[1]
            correct = pred_choice.eq(targets).cpu().sum()
            accuracy = correct.item() / float(batch_size)
            if bid%10 == 0:
                print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, bid, num_batch, loss.item(), accuracy))
                with open(log_file, "a") as file:
                         file.write('[%d: %d/%d] train loss: %f accuracy: %f\n' % (epoch, bid, num_batch, loss.item(), accuracy))

        if epoch == epochs - 1:
            # 保存FP32模型（备份）
            torch.save(student_model.module.state_dict(), "model/compact_distilbert_fp32.pth")
            # 转换为FP16并保存（体积减半）
            student_fp16 = student_model.module.half()  # 转为半精度
            torch.save(student_fp16.state_dict(), "model/compact_distilbert_fp16.pth")

        fin_outputs, fin_targets = test(student_model, testing_loader)

        #print("fin_outputs:", fin_outputs)
        #print("fin_targets:", fin_targets)
        # 定义标签名字
        confusion_matrix = metrics.confusion_matrix(fin_targets, fin_outputs)

        label_name = ['white', 'dga']
        draw_confusion_matrix(label_true=fin_targets,  # y_gt=[0,5,1,6,3,...]
                              label_pred=fin_outputs,  # y_pred=[0,5,1,6,3,...]
                              label_name=label_name,
                              title="Epoch%d Confusion Matrix" % epoch,
                              pdf_save_path="./logs/epoch_distill_%d/Confusion_Matrix.jpg" % epoch,
                              dpi=300)


        tp, fn, fp, tn = confusion_matrix.ravel()
        # print(tp, fn, fp, tn)
        # 计算准确率
        accuracy = round((tp + tn) / (tp + tn + fp + fn), 4)
        # print("准确率:", accuracy)
        # 计算精确率
        precision = round(tp / (tp + fp), 4)
        # print("精确率:", precision)
        # 计算召回率
        recall_score = round(tp / (tp + fn), 4)
        # print("召回率:", recall_score)
        # 计算f1值
        f1_score = round((2 * precision * recall_score) / (precision + recall_score), 4)
        # print("f1值:", f1_score)
        # 计算误报率
        false_positive_rate = round(fn / (tp + fn), 4)
        # print("误报率:", false_positive_rate)
        # 计算漏报率
        false_negative_rate = round(fp / (fp + tn), 4)
        # print("漏报率:", false_negative_rate)
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
        create_or_write_file("./logs/epoch_distill_%d/" % epoch, "a", "训练结果.txt", str(log_context))


def test(model, testing_loader):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bid, data in enumerate(testing_loader, 0):
            targets = data['targets']
            ids = data['token_ids'].cuda()
            attention_mask = data['attention_mask'].cuda()

            # 学生模型前向传播
            outputs = model(ids, attention_mask)  # (batch_size, num_labels)

            # 获取预测类别
            outputs = outputs.max(1)[1]
            targets = targets  # 已经是类别索引

            fin_targets.extend(targets.cpu().numpy())
            fin_outputs.extend(outputs.cpu().numpy())
    return fin_outputs, fin_targets
    # return fin_outputs, fin_targets, fin_sources

def label_smoothing(inputs, epsilon=0.1):
    return ((1 - epsilon) * inputs) + (epsilon / 2)

def loss_fun(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

if __name__ == '__main__':
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    test_size = 0.3
    train_size = 0.7
    seed = 42

    criterion_hard = torch.nn.CrossEntropyLoss()
    criterion_soft = torch.nn.KLDivLoss(reduction='batchmean')
    temperature =5
    a=0.7

    character_bert_model = CharacterBertModel.from_pretrained("./pretrained-models/general_character_bert").cuda()
    character_bert_model.eval()  # 教师模型在蒸馏过程中不需要梯度更新
    # 定义教师分类器，假设 embedding_dim 已知
    # 需要根据 CharacterBERT 的嵌入维度来设置
     # 请根据 CharacterBERT 的实际嵌入维度进行修改
    embedding_dim = character_bert_model.config.hidden_size
    teacher_model = CharacterBERTClassifier(character_bert_model, embedding_dim).cuda()

    # 初始化学生模型（DistilBERT）
    student_model = DistilBERTClassifier(num_labels=2).cuda()
    gpus = [0]
    teacher_model = torch.nn.DataParallel(teacher_model, device_ids=gpus, output_device=gpus[0])
    student_model = torch.nn.DataParallel(student_model, device_ids=gpus, output_device=gpus[0])


    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('Device:', device)

    # 获取预训练数据集、测试集、各联邦学习参与方的训练数据集
    training_loader, testing_loader = prepare_data()

    # 定义一个SGD优化器
    optimizer = torch.optim.SGD(student_model.parameters(), lr=1e-02)#原来1e-02
    # 开始训练
    # 开始训练
    train(
        teacher_model=teacher_model,
        student_model=student_model,
        epochs=1,
        training_loader=training_loader,
        testing_loader=testing_loader,
        optimizer=optimizer,
        criterion_hard=criterion_hard,
        criterion_soft=criterion_soft,
        temperature=temperature,
        device=device,
        a=a
    )
    end_time = time.time()
    print('Took %f second' % (end_time - start_time))