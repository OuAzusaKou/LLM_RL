import re
import matplotlib.pyplot as plt
from collections import Counter

# 读取文件并提取样本长度
def extract_sample_lengths(file_path):
    sample_lengths = []
    with open(file_path, 'r') as file:
        for line in file:
            # 使用正则表达式匹配样本长度值
            match = re.search(r"样本长度 (\d+)", line)
            if match:
                sample_length = int(match.group(1))
                sample_lengths.append(sample_length)
    return sample_lengths

# 统计不同区间的样本数量
def count_in_ranges(sample_lengths, ranges):
    counts = {range_str: 0 for range_str in ranges}
    
    for length in sample_lengths:
        for range_str in ranges:
            min_length, max_length = map(int, range_str.split('~'))
            if min_length <= length < max_length:
                counts[range_str] += 1
                break
    return counts

# 绘制分布图
def plot_distribution(counts):
    ranges = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(ranges, values)
    plt.xlabel('样本长度区间')
    plt.ylabel('样本数量')
    plt.title('样本长度区间分布')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main(file_path):
    # 提取样本长度
    sample_lengths = extract_sample_lengths(file_path)
    
    # 定义样本长度的区间
    ranges = ["1000~1500", "1500~2000", "2000~2500", "2500~3000", "3000~3500", "3500~4000", "4000~5000","5000~8192", "8192~20000"]
    
    # 统计不同区间的样本数量
    counts = count_in_ranges(sample_lengths, ranges)
    
    # 打印每个区间的数量
    for range_str, count in counts.items():
        print(f"{range_str} 样本长度区间样本数量 ：{count}个")
    
    # 绘制分布图
    plot_distribution(counts)

# 调用主函数，替换为你的文件路径
file_path = "overlen.txt"
main(file_path)

