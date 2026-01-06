"""
将 InstructERC 项目的 MELD 数据集转换为 ms-swift 框架支持的格式

ms-swift 支持的数据格式:
1. 简单格式: {"query": "问题", "response": "回答"}
2. 带system: {"system": "系统提示", "query": "问题", "response": "回答"}  
3. messages格式: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

本脚本保持原 InstructERC 的 prompt 格式，转换为 ms-swift 兼容的 JSONL 文件
"""

import pickle
import json
import os
import argparse
from typing import List, Dict, Tuple


# ==================== 配置 ====================
LABEL_SET = {
    'iemocap': ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
    'meld': ['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'],
    'EmoryNLP': ['Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared']
}

LABEL_TEXT_SET = {
    'iemocap': 'happy, sad, neutral, angry, excited, frustrated',
    'meld': 'neutral, surprise, fear, sad, joyful, disgust, angry',
    'EmoryNLP': 'Joyful, Mad, Peaceful, Neutral, Sad, Powerful, Scared'
}

SYSTEM_PROMPT = "Now you are expert of sentiment and emotional analysis."


def load_raw_data(data_path: str, dataset: str) -> Tuple[Dict, Dict, Dict, List, List, List]:
    """
    加载原始 pkl 数据
    
    Returns:
        speaker_info, emotion_labels, sentence_dict, train_ids, test_ids, valid_ids
    """
    data = pickle.load(open(data_path, 'rb'))
    
    if dataset == 'meld':
        speaker_info = data[0]
        emotion_labels = data[1]
        sentence_dict = data[3]
        train_ids = data[4]
        test_ids = data[5]
        valid_ids = data[6]
    elif dataset in ['iemocap', 'EmoryNLP']:
        speaker_info = data[0]
        emotion_labels = data[1]
        sentence_dict = data[2]
        train_ids = data[3]
        test_ids = data[4]
        valid_ids = data[5]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return speaker_info, emotion_labels, sentence_dict, train_ids, test_ids, valid_ids


def process_speaker_labels(speaker_info: Dict, conv_ids: List, dataset: str) -> Dict:
    """
    处理 speaker labels，将 one-hot 或字符串转换为数字 ID
    """
    speaker_label_dict = {}
    
    for conv_id in conv_ids:
        temp_speaker_list = []
        for speaker_label in speaker_info[conv_id]:
            if dataset == 'iemocap':
                # IEMOCAP: 'M' -> 0, 'F' -> 1
                if speaker_label == 'M':
                    temp_speaker_list.append(0)
                else:
                    temp_speaker_list.append(1)
            else:
                # MELD/EmoryNLP: one-hot vector -> index
                temp_speaker_list.append(speaker_label.index(1))
        speaker_label_dict[conv_id] = temp_speaker_list
    
    return speaker_label_dict


def build_prompt(
    sentence_dict: Dict,
    speaker_label_dict: Dict,
    emotion_labels: Dict,
    conv_id: str,
    conv_turn: int,
    window: int,
    dataset: str
) -> Tuple[str, str]:
    """
    构建单个样本的 prompt 和 target
    
    Args:
        sentence_dict: 对话句子字典
        speaker_label_dict: 说话人标签字典
        emotion_labels: 情感标签字典
        conv_id: 对话ID
        conv_turn: 当前轮次
        window: 历史窗口大小
        dataset: 数据集名称
    
    Returns:
        (query, response) 元组
    """
    # 计算窗口起始位置
    index_w = max(conv_turn - window, 0)
    
    # 构建对话历史部分
    conversation_parts = []
    for i in range(index_w, conv_turn + 1):
        speaker_id = speaker_label_dict[conv_id][i]
        utterance = sentence_dict[conv_id][i]
        conversation_parts.append(f'Speaker_{speaker_id}:"{utterance}"')
    
    conversation_text = '\t '.join(conversation_parts)
    
    # 获取目标话语
    target_utterance = conversation_parts[-1]
    
    # 构建完整的 query
    query = (
        f"The following conversation noted between '### ###' involves several speakers. "
        f"### {conversation_text} ### "
        f"Please select the emotional label of <{target_utterance}> from <{LABEL_TEXT_SET[dataset]}>:"
    )
    
    # 获取 response (情感标签)
    emotion_idx = emotion_labels[conv_id][conv_turn]
    response = LABEL_SET[dataset][emotion_idx]
    
    return query, response


def convert_to_swift_format(
    sentence_dict: Dict,
    speaker_label_dict: Dict,
    emotion_labels: Dict,
    conv_ids: List,
    window: int,
    dataset: str,
    format_type: str = "messages"
) -> List[Dict]:
    """
    将数据转换为 ms-swift 格式
    
    Args:
        format_type: 
            - "messages": OpenAI 风格的 messages 格式 (推荐)
            - "query_response": 简单的 query/response 格式
            - "alpaca": Alpaca 风格的 instruction/input/output 格式
    """
    swift_data = []
    
    for conv_id in conv_ids:
        num_turns = len(sentence_dict[conv_id])
        
        for conv_turn in range(num_turns):
            query, response = build_prompt(
                sentence_dict, speaker_label_dict, emotion_labels,
                conv_id, conv_turn, window, dataset
            )
            
            if format_type == "messages":
                # OpenAI 风格 messages 格式 (ms-swift 推荐)
                sample = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": response}
                    ]
                }
            elif format_type == "query_response":
                # 简单 query/response 格式
                sample = {
                    "system": SYSTEM_PROMPT,
                    "query": query,
                    "response": response
                }
            elif format_type == "alpaca":
                # Alpaca 风格格式
                sample = {
                    "instruction": SYSTEM_PROMPT + " " + query,
                    "input": "",
                    "output": response
                }
            else:
                raise ValueError(f"Unknown format_type: {format_type}")
            
            swift_data.append(sample)
    
    return swift_data


def save_jsonl(data: List[Dict], output_path: str):
    """保存为 JSONL 格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert ERC dataset to ms-swift format')
    parser.add_argument('--dataset', type=str, default='meld',
                        choices=['iemocap', 'meld', 'EmoryNLP'],
                        help='Dataset name')
    parser.add_argument('--window', type=int, default=12,
                        help='Historical context window size')
    parser.add_argument('--format', type=str, default='messages',
                        choices=['messages', 'query_response', 'alpaca'],
                        help='Output format type')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for swift dataset')
    
    args = parser.parse_args()
    
    # 设置默认路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.input_dir is None:
        args.input_dir = os.path.join(project_root, "original_data", args.dataset)
    
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, "swift_data", args.dataset)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载原始数据
    pkl_path = os.path.join(args.input_dir, f"{args.dataset}.pkl")
    print(f"Loading data from: {pkl_path}")
    
    speaker_info, emotion_labels, sentence_dict, train_ids, test_ids, valid_ids = \
        load_raw_data(pkl_path, args.dataset)
    
    # 处理所有对话的 speaker labels
    all_conv_ids = train_ids + test_ids + valid_ids
    speaker_label_dict = process_speaker_labels(speaker_info, all_conv_ids, args.dataset)
    
    print(f"\nDataset: {args.dataset}")
    print(f"Window size: {args.window}")
    print(f"Format: {args.format}")
    print(f"Train conversations: {len(train_ids)}")
    print(f"Test conversations: {len(test_ids)}")
    print(f"Valid conversations: {len(valid_ids)}")
    
    # 转换数据
    print("\nConverting data...")
    
    train_data = convert_to_swift_format(
        sentence_dict, speaker_label_dict, emotion_labels,
        train_ids, args.window, args.dataset, args.format
    )
    
    test_data = convert_to_swift_format(
        sentence_dict, speaker_label_dict, emotion_labels,
        test_ids, args.window, args.dataset, args.format
    )
    
    valid_data = convert_to_swift_format(
        sentence_dict, speaker_label_dict, emotion_labels,
        valid_ids, args.window, args.dataset, args.format
    )
    
    # 保存数据
    print("\nSaving data...")
    save_jsonl(train_data, os.path.join(args.output_dir, "train.jsonl"))
    save_jsonl(test_data, os.path.join(args.output_dir, "test.jsonl"))
    save_jsonl(valid_data, os.path.join(args.output_dir, "valid.jsonl"))
    
    # 打印示例
    print("\n" + "="*60)
    print("Sample data (first example from train set):")
    print("="*60)
    print(json.dumps(train_data[0], indent=2, ensure_ascii=False))
    
    # 统计信息
    print("\n" + "="*60)
    print("Statistics:")
    print("="*60)
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Valid samples: {len(valid_data)}")
    print(f"Total samples: {len(train_data) + len(test_data) + len(valid_data)}")
    
    # 标签分布
    label_counts = {}
    for item in train_data:
        if args.format == "messages":
            label = item["messages"][-1]["content"]
        elif args.format == "query_response":
            label = item["response"]
        else:
            label = item["output"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\nTrain label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count} ({count/len(train_data)*100:.1f}%)")
    
    print(f"\nOutput directory: {args.output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()

