"""
ms-swift 推理脚本 - MELD 情感识别任务评估

使用训练好的模型进行推理和评估
"""

import json
import os
import argparse
import logging
import sys
from datetime import datetime
from typing import List, Dict
from collections import Counter

# 需要安装: pip install ms-swift
from swift.llm import PtEngine, RequestConfig, get_template
from sklearn.metrics import accuracy_score, f1_score, classification_report


LABEL_SET = ['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry']


def setup_logger(log_file: str = None):
    """
    设置日志系统，同时输出到控制台和文件
    
    Args:
        log_file: 日志文件路径，如果为 None 则自动生成
    """
    if log_file is None:
        # 自动生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/inference_{timestamp}.log"
    
    # 创建 logs 目录
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else "logs", exist_ok=True)
    
    # 配置 logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的 handlers
    logger.handlers = []
    
    # 文件 handler - 使用 UTF-8 编码
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # 控制台 handler - 兼容 Windows
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Windows 控制台编码兼容性处理
    if sys.platform == 'win32':
        try:
            # 尝试设置 UTF-8 编码
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
        except:
            pass
    
    # 添加 handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


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
    
    logging.info(f"Loading model from: {model_path}")
    logging.info(f"Loading test data from: {test_file}")
    
    # 加载测试数据
    test_data = load_test_data(test_file)
    logging.info(f"Loaded {len(test_data)} test samples")
    
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
    
    logging.info("\nRunning inference...")
    
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
            logging.info(f"Processed {min(i + batch_size, len(test_data))}/{len(test_data)}")
    
    # 计算指标
    logging.info("\n" + "="*60)
    logging.info("Evaluation Results")
    logging.info("="*60)
    
    # 转换为数字标签
    label_to_idx = {label: idx for idx, label in enumerate(LABEL_SET)}
    
    pred_indices = [label_to_idx.get(p, 0) for p in predictions]
    gold_indices = [label_to_idx.get(g, 0) for g in ground_truths]
    
    acc = accuracy_score(gold_indices, pred_indices)
    f1_weighted = f1_score(gold_indices, pred_indices, average='weighted')
    f1_macro = f1_score(gold_indices, pred_indices, average='macro')
    
    logging.info(f"\nAccuracy: {acc*100:.2f}%")
    logging.info(f"Weighted F1: {f1_weighted*100:.2f}%")
    logging.info(f"Macro F1: {f1_macro*100:.2f}%")
    
    logging.info("\nClassification Report:")
    report = classification_report(gold_indices, pred_indices, 
                               target_names=LABEL_SET, digits=4, 
                               zero_division=0)
    logging.info("\n" + report)
    
    # 真实标签分布
    logging.info("\nGround Truth Distribution:")
    truth_counter = Counter(ground_truths)
    for label in LABEL_SET:
        count = truth_counter.get(label, 0)
        logging.info(f"  {label}: {count} ({count/len(ground_truths)*100:.1f}%)")
    
    # 预测分布
    logging.info("\nPrediction Distribution:")
    pred_counter = Counter(predictions)
    for label in LABEL_SET:
        count = pred_counter.get(label, 0)
        logging.info(f"  {label}: {count} ({count/len(predictions)*100:.1f}%)")
    
    # 显示一些预测示例
    logging.info("\n" + "="*60)
    logging.info("Sample Predictions (first 5):")
    logging.info("="*60)
    for i in range(min(5, len(predictions))):
        logging.info(f"\n#{i+1}")
        logging.info(f"  True Label:    {ground_truths[i]}")
        logging.info(f"  Predicted:     {predictions[i]}")
        logging.info(f"  Raw Output:    {raw_outputs[i][:100]}...")  # 截取前100字符
        logging.info(f"  Match:         {'✓' if predictions[i] == ground_truths[i] else '✗'}")
    
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
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to log file (default: auto-generated in logs/ directory)')
    
    args = parser.parse_args()
    
    # 设置日志系统
    logger, log_file = setup_logger(args.log_file)
    logging.info("="*60)
    logging.info("MELD Emotion Recognition - Inference & Evaluation")
    logging.info("="*60)
    logging.info(f"Log file: {log_file}")
    logging.info(f"Model: {args.model_path}")
    logging.info(f"Test data: {args.test_file}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info("="*60 + "\n")
    
    # 执行评估
    results = evaluate(args.model_path, args.test_file, args.batch_size)
    
    # 保存结果
    output_file = os.path.join(os.path.dirname(args.model_path), "eval_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"\nResults saved to: {output_file}")
    logging.info(f"Log saved to: {log_file}")
    logging.info("\n" + "="*60)
    logging.info("Inference completed successfully!")


if __name__ == "__main__":
    main()

