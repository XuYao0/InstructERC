#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 swift_data/meld 数据集中随机抽样脚本

使用示例:
    python code/sample_swift_data.py --num_samples 256
    python code/sample_swift_data.py --num_samples 256 --split train --output swift_data/meld/train_sampled_256.jsonl
    python code/sample_swift_data.py --num_samples 100 --split valid --seed 42
"""

import json
import random
import argparse
from pathlib import Path


def load_jsonl(file_path):
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """保存为 JSONL 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ 已保存 {len(data)} 条数据到: {file_path}")


def sample_data(input_file, num_samples, output_file, seed=None):
    """随机抽样数据"""
    # 设置随机种子以确保可复现
    if seed is not None:
        random.seed(seed)
        print(f"🎲 使用随机种子: {seed}")
    
    # 加载数据
    print(f"📂 正在加载数据: {input_file}")
    data = load_jsonl(input_file)
    total_samples = len(data)
    print(f"📊 数据集总数: {total_samples} 条")
    
    # 检查抽样数量
    if num_samples > total_samples:
        print(f"⚠️  警告: 请求抽样数量 ({num_samples}) 大于数据集总数 ({total_samples})")
        print(f"📝 将使用全部数据")
        sampled_data = data
    else:
        # 随机抽样
        sampled_data = random.sample(data, num_samples)
        print(f"✨ 已随机抽样 {num_samples} 条数据")
    
    # 保存抽样数据
    save_jsonl(sampled_data, output_file)
    
    # 打印统计信息
    print("\n" + "="*50)
    print("📈 抽样统计:")
    print(f"  原始数据: {total_samples} 条")
    print(f"  抽样数据: {len(sampled_data)} 条")
    print(f"  抽样比例: {len(sampled_data)/total_samples*100:.2f}%")
    
    # 分析情感标签分布
    analyze_label_distribution(sampled_data)
    print("="*50)


def analyze_label_distribution(data):
    """分析情感标签分布"""
    label_counts = {}
    for item in data:
        # 从 assistant 的回复中提取标签
        for msg in item['messages']:
            if msg['role'] == 'assistant':
                label = msg['content']
                label_counts[label] = label_counts.get(label, 0) + 1
                break
    
    if label_counts:
        print("\n📊 情感标签分布:")
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        for label, count in sorted_labels:
            percentage = count / len(data) * 100
            print(f"  {label:10s}: {count:4d} ({percentage:5.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='从 swift_data/meld 数据集中随机抽样',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 从训练集抽样 256 条数据
  python code/sample_swift_data.py --num_samples 256
  
  # 从验证集抽样 100 条数据，并指定随机种子
  python code/sample_swift_data.py --num_samples 100 --split valid --seed 42
  
  # 自定义输出文件路径
  python code/sample_swift_data.py --num_samples 256 --output custom_output.jsonl
        """
    )
    
    parser.add_argument(
        '--num_samples', '-n',
        type=int,
        required=True,
        help='抽样数量，例如: 256'
    )
    
    parser.add_argument(
        '--split', '-s',
        type=str,
        default='train',
        choices=['train', 'valid', 'test'],
        help='选择数据集分割 (默认: train)'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help='输入文件路径 (默认: swift_data/meld/{split}.jsonl)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出文件路径 (默认: swift_data/meld/{split}_sampled_{num}.jsonl)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子，用于确保结果可复现 (默认: None)'
    )
    
    args = parser.parse_args()
    
    # 确定输入文件路径
    if args.input:
        input_file = Path(args.input)
    else:
        input_file = Path(f'swift_data/meld/{args.split}.jsonl')
    
    if not input_file.exists():
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        return
    
    # 确定输出文件路径
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f'swift_data/meld/{args.split}_sampled_{args.num_samples}.jsonl')
    
    # 创建输出目录
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*50)
    print("🚀 开始随机抽样")
    print("="*50)
    
    # 执行抽样
    sample_data(input_file, args.num_samples, output_file, args.seed)


if __name__ == '__main__':
    main()

