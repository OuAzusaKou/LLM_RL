import re
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

def parse_log_file(file_path):
    """解析日志文件，提取所有样本长度值"""
    pattern = re.compile(r'样本长度\s+(\d+)\s+超过最大长度\s+\d+')
    lengths = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            match = pattern.search(line)
            if match:
                try:
                    length = int(match.group(1))
                    lengths.append(length)
                except ValueError:
                    print(f"警告：第 {line_num} 行数值格式异常，跳过无效数据")
    return lengths

def plot_distribution(data, output_path='length_distribution.png'):
    """绘制样本长度分布图"""
    plt.figure(figsize=(12, 6))
    
    # 绘制直方图
    plt.subplot(1, 2, 1)
    plt.hist(data, bins=30, color='#1f77b4', edgecolor='black')
    plt.title('样本长度直方图')
    plt.xlabel('长度值')
    plt.ylabel('频次')
    
    # 绘制TOP10条目
    top_counts = defaultdict(int)
    for l in data:
        top_counts[l] += 1
    
    plt.subplot(1, 2, 2)
    sorted_items = sorted(top_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    plt.barh([f"{k} ({v}次)" for k, v in sorted_items], [v for k, v in sorted_items], color='#ff7f0e')
    plt.title('高频长度TOP10')
    plt.xlabel('出现次数')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"分布图已保存至 {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析日志中的样本长度分布')
    parser.add_argument('file_path', help='日志文件路径')
    parser.add_argument('--bins', type=int, default=30, help='直方图分箱数')
    args = parser.parse_args()

    # 解析日志文件
    try:
        lengths = parse_log_file(args.file_path)
        print(f"共提取到 {len(lengths)} 个有效样本长度")
    except FileNotFoundError:
        print("错误：文件未找到，请检查路径")
        exit(1)
    
    # 绘制分布图
    plot_distribution(lengths, f'length_dist_bins{args.bins}.png')
