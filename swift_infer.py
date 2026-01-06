"""
ms-swift 推理脚本 - MELD 情感识别任务评估

使用训练好的模型进行推理和评估
"""

import json
import os
import argparse
from typing import List, Dict
from collections import Counter

# 需要安装: pip install ms-swift
from swift.llm import PtEngine, RequestConfig, get_template
from sklearn.metrics import accuracy_score, f1_score, classification_report


LABEL_SET = ['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry']


def load_test_data(test_file: str) -> List[Dict]:
    """加载测试数据"""
    data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_label(response: str) -> str:
    """从模型响应中提取标签"""
    response = response.strip().lower()
    
    # 直接匹配
    for label in LABEL_SET:
        if label.lower() in response:
            return label
    
    # 如果没找到，返回最可能的（基于编辑距离）
    return response.split()[0] if response else "neutral"


def evaluate(model_path: str, test_file: str, batch_size: int = 8):
    """
    使用训练好的模型进行评估
    
    Args:
        model_path: 模型路径（可以是原始模型或微调后的checkpoint）
        test_file: 测试数据文件路径
        batch_size: 批处理大小
    """
    
    print(f"Loading model from: {model_path}")
    print(f"Loading test data from: {test_file}")
    
    # 加载测试数据
    test_data = load_test_data(test_file)
    print(f"Loaded {len(test_data)} test samples")
    
    # 初始化推理引擎
    engine = PtEngine(model_path)
    
    # 准备推理请求
    predictions = []
    ground_truths = []
    raw_outputs = []  # 保存原始输出
    
    request_config = RequestConfig(
        max_tokens=20,
        temperature=0.0,  # greedy decoding
    )
    
    print("\nRunning inference...")
    
    # 批量推理
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        
        # 构建推理输入
        infer_requests = []
        for item in batch:
            messages = item["messages"][:2]  # system + user
            infer_requests.append({"messages": messages})
            ground_truths.append(item["messages"][2]["content"])
        
        # 推理
        responses = engine.infer(infer_requests, request_config=request_config)
        
        for resp in responses:
            raw_output = resp.choices[0].message.content
            raw_outputs.append(raw_output)
            pred_label = extract_label(raw_output)
            predictions.append(pred_label)
        
        if (i + batch_size) % 100 == 0:
            print(f"Processed {min(i + batch_size, len(test_data))}/{len(test_data)}")
    
    # 计算指标
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    # 转换为数字标签
    label_to_idx = {label: idx for idx, label in enumerate(LABEL_SET)}
    
    pred_indices = [label_to_idx.get(p, 0) for p in predictions]
    gold_indices = [label_to_idx.get(g, 0) for g in ground_truths]
    
    acc = accuracy_score(gold_indices, pred_indices)
    f1_weighted = f1_score(gold_indices, pred_indices, average='weighted')
    f1_macro = f1_score(gold_indices, pred_indices, average='macro')
    
    print(f"\nAccuracy: {acc*100:.2f}%")
    print(f"Weighted F1: {f1_weighted*100:.2f}%")
    print(f"Macro F1: {f1_macro*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(gold_indices, pred_indices, 
                               target_names=LABEL_SET, digits=4, 
                               zero_division=0))
    
    # 真实标签分布
    print("\nGround Truth Distribution:")
    truth_counter = Counter(ground_truths)
    for label in LABEL_SET:
        count = truth_counter.get(label, 0)
        print(f"  {label}: {count} ({count/len(ground_truths)*100:.1f}%)")
    
    # 预测分布
    print("\nPrediction Distribution:")
    pred_counter = Counter(predictions)
    for label in LABEL_SET:
        count = pred_counter.get(label, 0)
        print(f"  {label}: {count} ({count/len(predictions)*100:.1f}%)")
    
    # 显示一些预测示例
    print("\n" + "="*60)
    print("Sample Predictions (first 5):")
    print("="*60)
    for i in range(min(5, len(predictions))):
        print(f"\n#{i+1}")
        print(f"  True Label:    {ground_truths[i]}")
        print(f"  Predicted:     {predictions[i]}")
        print(f"  Raw Output:    {raw_outputs[i][:100]}...")  # 截取前100字符
        print(f"  Match:         {'✓' if predictions[i] == ground_truths[i] else '✗'}")
    
    return {
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model on MELD test set')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--test_file', type=str, 
                        default='swift_data/meld/test.jsonl',
                        help='Path to test data file')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    
    args = parser.parse_args()
    
    results = evaluate(args.model_path, args.test_file, args.batch_size)
    
    # 保存结果
    output_file = os.path.join(os.path.dirname(args.model_path), "eval_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

