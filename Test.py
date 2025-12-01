import re


def parse_federated_results(filename):
    """
    从联邦学习实验结果文件中提取服务器端测试集评估结果

    Args:
        filename: 实验结果文本文件名

    Returns:
        提取的服务器端评估结果列表
    """
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

    # 使用正则表达式匹配所有轮次的服务器端评估结果
    pattern = r"Round (\d+) 评估结果.*?服务器端测试集评估结果:.*?- 准确率: ([\d.]+)%.*?- 精确率: ([\d.]+)%.*?- 召回率: ([\d.]+)%.*?- F1分数: ([\d.]+)%"
    matches = re.findall(pattern, content, re.DOTALL)

    # 将结果保存为列表
    results = []
    for match in matches:
        round_num = int(match[0])
        accuracy = float(match[1])
        precision = float(match[2])
        recall = float(match[3])
        f1_score = float(match[4])
        results.append((round_num, accuracy, precision, recall, f1_score))

    return results


def print_results_table(results):
    """
    打印结果表格

    Args:
        results: 解析出的结果列表
    """
    print("轮数\t准确率\t精确率\t召回率\tF1分数")
    print("-" * 50)
    for result in results:
        print(f"{result[0]}\t{result[1]:.2f}\t{result[2]:.2f}\t{result[3]:.2f}\t{result[4]:.2f}")


if __name__ == "__main__":
    # 替换为你的实验结果文件名
    filename = "./federated_results_20250319_192745 3 30.txt"

    try:
        results = parse_federated_results(filename)
        print_results_table(results)

        # 打印最后一轮结果
        last_round = results[-1]
        print("\n最终模型性能:")
        print(f"准确率: {last_round[1]:.2f}%")
        print(f"精确率: {last_round[2]:.2f}%")
        print(f"召回率: {last_round[3]:.2f}%")
        print(f"F1分数: {last_round[4]:.2f}%")

    except Exception as e:
        print(f"解析文件时出错: {e}")