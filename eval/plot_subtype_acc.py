import json
import zhplot
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib as mpl



def load_evaluation_results(file_path):
    """从JSON文件加载评估结果"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None

def plot_comparison(result1, result2, model1_name="基线模型", model2_name="改进模型", sort_by="accuracy"):
    """绘制两个模型的准确率对比图"""
    # 在绘图前设置中文字体
    # setup_chinese_font()
    
    # 提取子类型统计信息
    stats1 = result1["subtype_statistics"]
    stats2 = result2["subtype_statistics"]
    
    # 合并所有子类型
    all_subtypes = set(stats1.keys()) | set(stats2.keys())
    
    # 准备数据
    data = []
    for subtype in all_subtypes:
        acc1 = stats1.get(subtype, {"accuracy": 0, "total": 0})["accuracy"] if subtype in stats1 else 0
        acc2 = stats2.get(subtype, {"accuracy": 0, "total": 0})["accuracy"] if subtype in stats2 else 0
        total1 = stats1.get(subtype, {"total": 0})["total"] if subtype in stats1 else 0
        total2 = stats2.get(subtype, {"total": 0})["total"] if subtype in stats2 else 0
        data.append((subtype, acc1, acc2, total1, total2))
    
    # 排序
    if sort_by == "accuracy":
        # 按平均准确率排序
        data.sort(key=lambda x: (x[1] + x[2]) / 2, reverse=True)
    elif sort_by == "sample_count":
        # 按样本总数排序
        data.sort(key=lambda x: x[3] + x[4], reverse=True)
    elif sort_by == "difference":
        # 按准确率差异排序
        data.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
    
    # 分离数据
    subtypes = [item[0] for item in data]
    acc1_values = [item[1] for item in data]
    acc2_values = [item[2] for item in data]
    total1_values = [item[3] for item in data]
    total2_values = [item[4] for item in data]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # 设置宽度和位置
    width = 0.35
    x = np.arange(len(subtypes))
    
    # 绘制准确率柱状图
    bar1 = ax1.bar(x - width/2, acc1_values, width, label=f"{model1_name} 准确率", color='skyblue')
    bar2 = ax1.bar(x + width/2, acc2_values, width, label=f"{model2_name} 准确率", color='lightcoral')
    
    # 添加准确率标签
    def add_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1%}', ha='center', va='bottom', fontsize=8)
    
    add_labels(bar1, acc1_values)
    add_labels(bar2, acc2_values)
    
    # 设置准确率图表属性
    ax1.set_ylabel('准确率')
    ax1.set_title('不同模型在各缺陷类型上的准确率对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(subtypes, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 1.1)  # 确保有足够空间显示标签
    
    # 修改样本数量柱状图部分
    ax2.bar(x, total1_values, width, label="样本数量", color='lightgray', alpha=0.6)
    
    # 设置样本数量图表属性
    ax2.set_ylabel('样本数量')
    ax2.set_xlabel('缺陷类型')
    ax2.set_xticks(x)
    ax2.set_xticklabels(subtypes, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加样本数量标签
    for i, v in enumerate(total1_values):
        if v > 0:
            ax2.text(i, v, str(v), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    print("对比图已保存为 model_comparison.png")
    
    # 绘制总体准确率对比
    fig, ax = plt.subplots(figsize=(8, 6))
    overall_acc = [result1["overall_accuracy"], result2["overall_accuracy"]]
    models = [model1_name, model2_name]
    
    bars = ax.bar(models, overall_acc, color=['skyblue', 'lightcoral'])
    
    # 添加准确率标签
    for bar, value in zip(bars, overall_acc):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2%}', ha='center', va='bottom')
    
    ax.set_ylabel('准确率')
    ax.set_title('模型总体准确率对比')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('overall_accuracy_comparison.png', dpi=300)
    print("总体准确率对比图已保存为 overall_accuracy_comparison.png")

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 统计两种类型样本的准确率
    def get_type_stats(result, expected_type):
        samples = [item for item in result["detailed_results"] 
                  if item["expected"] == expected_type]
        total = len(samples)
        correct = sum(1 for item in samples 
                     if expected_type in item["predicted"])
        accuracy = correct / total if total > 0 else 0
        return total, correct, accuracy
    
    # 获取两个模型的统计数据
    fp1_total, fp1_correct, fp1_acc = get_type_stats(result1, "疑似误报")
    fp2_total, fp2_correct, fp2_acc = get_type_stats(result2, "疑似误报")
    
    real1_total, real1_correct, real1_acc = get_type_stats(result1, "缺陷真实存在")
    real2_total, real2_correct, real2_acc = get_type_stats(result2, "缺陷真实存在")
    
    # 绘制准确率条形图
    x = np.arange(2)
    width = 0.35
    
    # 绘制疑似误报准确率
    bars1 = ax.bar(x - width/2, [fp1_acc, fp2_acc], width, 
                  label='疑似误报准确率', color='skyblue')
    
    # 绘制真实缺陷准确率
    bars2 = ax.bar(x + width/2, [real1_acc, real2_acc], width,
                  label='真实缺陷准确率', color='lightcoral')
    
    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}', ha='center', va='bottom')
    
    add_labels(bars1)
    add_labels(bars2)
    
    # 设置图表属性
    ax.set_ylabel('准确率')
    ax.set_title('模型在不同类型样本上的准确率对比')
    ax.set_xticks(x)
    ax.set_xticklabels([model1_name, model2_name])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.1)
    
    # 添加样本数量信息
    ax.text(0.02, 0.95, 
            f'疑似误报样本数: {fp1_total}\n真实缺陷样本数: {real1_total}', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('type_accuracy_comparison.png', dpi=300)
    print(f"类型准确率对比图已保存为 type_accuracy_comparison.png")
    print(f"\n疑似误报样本数: {fp1_total}")
    print(f"真实缺陷样本数: {real1_total}")
    print(f"\n{model1_name}:")
    print(f"疑似误报准确率: {fp1_acc:.2%}")
    print(f"真实缺陷准确率: {real1_acc:.2%}")
    print(f"\n{model2_name}:")
    print(f"疑似误报准确率: {fp2_acc:.2%}")
    print(f"真实缺陷准确率: {real2_acc:.2%}")

def main():
    # 加载评估结果
    result1_path = "/9950backfile/liguoqi/wangzihang/LLM_RL/evaluation_results.json"
    result2_path = "/9950backfile/liguoqi/wangzihang/LLM_RL/evaluation_results_0528_sft.json"  # 假设第二个模型的结果文件
    
    result1 = load_evaluation_results(result1_path)
    
    # 检查第二个文件是否存在，不存在则使用第一个文件（仅展示单模型结果）
    try:
        result2 = load_evaluation_results(result2_path)
    except:
        print(f"警告: 找不到文件 {result2_path}，将只展示单模型结果")
        result2 = result1
        
    if result1 is None:
        print("错误: 无法加载评估结果")
        return
    
    # 绘制对比图
    model1_name = "基线模型"
    model2_name = "改进模型" if result1 != result2 else "基线模型"
    
    # 按准确率排序绘制对比图
    plot_comparison(result1, result2, model1_name, model2_name, sort_by="accuracy")
    
    print("数据可视化完成")

if __name__ == "__main__":
    main()

# 使用仿宋字体

