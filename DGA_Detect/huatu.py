import matplotlib.pyplot as plt
import numpy as np

# 模拟数据（示例）
epochs = np.arange(1, 50)  # 训练轮次，1 到 49
accuracy = np.random.uniform(0.91, 1.0, len(epochs))  # 准确率
f1 = np.random.uniform(0.91, 1.0, len(epochs))        # F1 分数
precision = np.random.uniform(0.91, 1.0, len(epochs)) # 精确率
recall = np.random.uniform(0.91, 1.0, len(epochs))    # 召回率

# 模拟数据波动（让曲线更接近示例）
accuracy += np.sin(epochs * 0.5) * 0.01
f1 += np.cos(epochs * 0.5) * 0.01
precision += np.sin(epochs * 0.6) * 0.01
recall += np.cos(epochs * 0.6) * 0.01

# 绘制图形
plt.figure(figsize=(10, 6))  # 设置画布大小

# 绘制各指标曲线
plt.plot(epochs, accuracy, label='准确率', color='blue', linestyle='-', marker='o', markersize=3)
plt.plot(epochs, f1, label='F1 分数', color='orange', linestyle='-', marker='o', markersize=3)
plt.plot(epochs, precision, label='精确率', color='green', linestyle='-', marker='o', markersize=3)
plt.plot(epochs, recall, label='召回率', color='lightblue', linestyle='-', marker='o', markersize=3)

# 添加标题和标签
plt.title('训练轮次与指标变化曲线')
plt.xlabel('训练轮次 (Epoch)')
plt.ylabel('指标值')

# 设置 y 轴范围
plt.ylim(0.91, 1.0)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
plt.legend()

# 显示图形
plt.show()