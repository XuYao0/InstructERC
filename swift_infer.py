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
    """
    从模型响应中提取标签
    
    策略：遍历所有标签，找出在模型输出中位置最靠前的标签
    如果没有找到任何标签，返回空字符串
    """
    response = response.strip().lower()
    
    # 如果响应为空，返回空字符串
    if not response:
        return ""
    
    # 方法1: 找出所有标签在response中的位置，返回位置最靠前的
    label_positions = {}
    for label in LABEL_SET:
        pos = response.find(label.lower())
        if pos != -1:  # 找到了该标签
            label_positions[label] = pos
    
    # 如果找到了标签，返回位置最靠前的
    if label_positions:
        # 按位置排序，返回位置最小（最靠前）的标签
        earliest_label = min(label_positions, key=label_positions.get)
        return earliest_label
    
    # 方法2: 如果没有找到任何标签，尝试按单词匹配（避免子串匹配问题）
    words = response.split()
    for word in words:
        word_clean = word.strip('.,!?;:"\'-')
        if word_clean in LABEL_SET:
            return word_clean
    
    # 如果都没找到，返回空字符串（表示未能提取到标签）
    return ""


def evaluate(model_path: str, test_file: str, batch_size: int = 8, output_file: str = None):
    """
    使用训练好的模型进行评估
    
    Args:
        model_path: 模型路径（可以是原始模型或微调后的checkpoint）
        test_file: 测试数据文件路径
        batch_size: 批处理大小
        output_file: 详细结果输出文件路径（流式写入，避免中途中断导致数据丢失）
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
        max_tokens=200,
        temperature=0.0,  # greedy decoding
    )
    
    logging.info("\nRunning inference...")
    
    # 打开输出文件（流式写入模式）
    output_fp = None
    if output_file:
        output_fp = open(output_file, 'w', encoding='utf-8')
        logging.info(f"Streaming results to: {output_file}")
    
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
        
        for idx, resp in enumerate(responses):
            raw_output = resp.choices[0].message.content
            raw_outputs.append(raw_output)
            pred_label = extract_label(raw_output)
            predictions.append(pred_label)
            
            # 构建详细结果
            sample_idx = i + idx
            ground_truth = batch[idx]["messages"][2]["content"]
            # 空标签表示未能提取到有效标签，算作预测错误
            is_correct = (pred_label == ground_truth) if pred_label else False
            
            result_item = {
                "index": sample_idx,
                "input_messages": batch[idx]["messages"][:2],  # system + user
                "ground_truth": ground_truth,
                "model_output": raw_output,
                "predicted_label": pred_label if pred_label else "",  # 空标签记录为空字符串
                "is_correct": is_correct,
                "extraction_failed": not bool(pred_label)  # 标记是否提取失败
            }
            
            # 立即写入文件（流式保存，避免中途中断导致数据丢失）
            if output_fp:
                output_fp.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                output_fp.flush()  # 立即刷新到磁盘
        
        if (i + batch_size) % 100 == 0:
            logging.info(f"Processed {min(i + batch_size, len(test_data))}/{len(test_data)}")
    
    # 关闭输出文件
    if output_fp:
        output_fp.close()
        logging.info(f"All results saved to: {output_file}")
    
    # 计算指标
    logging.info("\n" + "="*60)
    logging.info("Evaluation Results")
    logging.info("="*60)
    
    # 转换为数字标签
    label_to_idx = {label: idx for idx, label in enumerate(LABEL_SET)}
    
    # 对于空标签（提取失败），映射到 -1（不存在的类别）
    pred_indices = [label_to_idx.get(p, -1) if p else -1 for p in predictions]
    gold_indices = [label_to_idx.get(g, 0) for g in ground_truths]
    
    acc = accuracy_score(gold_indices, pred_indices)
    f1_weighted = f1_score(gold_indices, pred_indices, average='weighted')
    f1_macro = f1_score(gold_indices, pred_indices, average='macro')
    
    # 统计提取失败的样本数
    failed_extractions = sum(1 for p in predictions if not p)
    total_samples = len(predictions)
    
    logging.info(f"\nAccuracy: {acc*100:.2f}%")
    logging.info(f"Weighted F1: {f1_weighted*100:.2f}%")
    logging.info(f"Macro F1: {f1_macro*100:.2f}%")
    logging.info(f"\nLabel Extraction Statistics:")
    logging.info(f"  Total samples: {total_samples}")
    logging.info(f"  Successfully extracted: {total_samples - failed_extractions} ({(total_samples - failed_extractions)/total_samples*100:.2f}%)")
    logging.info(f"  Failed to extract: {failed_extractions} ({failed_extractions/total_samples*100:.2f}%)")
    
    logging.info("\nClassification Report:")
    # 只统计真实的标签类别（0-6），忽略提取失败的 -1
    valid_label_indices = list(range(len(LABEL_SET)))
    report = classification_report(
        gold_indices, 
        pred_indices, 
        labels=valid_label_indices,  # 明确指定只统计这 7 个类别
        target_names=LABEL_SET, 
        digits=4, 
        zero_division=0
    )
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
    # 统计空标签（提取失败）
    empty_count = pred_counter.get("", 0)
    if empty_count > 0:
        logging.info(f"  [EMPTY - Extraction Failed]: {empty_count} ({empty_count/len(predictions)*100:.1f}%)")
    
    # 显示一些预测示例
    logging.info("\n" + "="*60)
    logging.info("Sample Predictions (first 5):")
    logging.info("="*60)
    for i in range(min(5, len(predictions))):
        logging.info(f"\n#{i+1}")
        logging.info(f"  True Label:    {ground_truths[i]}")
        logging.info(f"  Predicted:     {predictions[i] if predictions[i] else '[EMPTY - Failed to extract]'}")
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
    
    # 生成时间戳和模型名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(args.model_path.rstrip('/\\'))
    
    # 确保 logs 目录存在
    os.makedirs("logs", exist_ok=True)
    
    # 准备输出文件路径
    detailed_output_file = f"logs/detailed_outputs_{model_name}_{timestamp}.jsonl"
    
    # 执行评估（结果会流式写入 detailed_output_file）
    results = evaluate(args.model_path, args.test_file, args.batch_size, detailed_output_file)
    
    # 保存评估指标
    metrics_file = f"logs/eval_results_{model_name}_{timestamp}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logging.info(f"\nMetrics saved to: {metrics_file}")
    logging.info(f"Detailed outputs saved to: {detailed_output_file}")
    
    logging.info(f"Log saved to: {log_file}")
    logging.info("\n" + "="*60)
    logging.info("Inference completed successfully!")


if __name__ == "__main__":
    main()

