"""
InstructERC 数据处理流程演示脚本
================================
本脚本展示了数据从原始读取到送入tokenizer的完整流程，
并将每一步的数据变化打印到日志文件中。

数据处理流程：
1. 原始数据读取 (pkl文件)
2. Speaker Label 提取
3. 对话窗口构建
4. Prompt 构建（指令模板）
5. JSON 中间文件生成
6. JSON 文件读取
7. Dataset 构建
8. Tokenizer 处理
"""

import pickle
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# 添加transformers支持（可选，如果环境中有的话）
try:
    from transformers import AutoTokenizer, LlamaTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("注意: transformers 未安装，tokenizer 步骤将使用模拟演示")


class DataPipelineLogger:
    """数据处理流程日志记录器"""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.log_content = []
        
    def log(self, title: str, content: str, level: int = 1):
        """记录日志"""
        header = "#" * level + " " + title
        self.log_content.append(header)
        self.log_content.append("")
        self.log_content.append(content)
        self.log_content.append("")
        self.log_content.append("---")
        self.log_content.append("")
        
    def log_dict(self, title: str, data: Dict, level: int = 1):
        """记录字典数据"""
        header = "#" * level + " " + title
        self.log_content.append(header)
        self.log_content.append("")
        self.log_content.append("```json")
        self.log_content.append(json.dumps(data, indent=2, ensure_ascii=False))
        self.log_content.append("```")
        self.log_content.append("")
        self.log_content.append("---")
        self.log_content.append("")
        
    def log_code(self, title: str, code: str, lang: str = "python", level: int = 1):
        """记录代码块"""
        header = "#" * level + " " + title
        self.log_content.append(header)
        self.log_content.append("")
        self.log_content.append(f"```{lang}")
        self.log_content.append(code)
        self.log_content.append("```")
        self.log_content.append("")
        self.log_content.append("---")
        self.log_content.append("")
    
    def save(self):
        """保存日志到文件"""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.log_content))
        print(f"日志已保存到: {self.output_path}")


def demo_data_pipeline(
    dataset_name: str = "meld",
    original_data_path: str = None,
    processed_data_path: str = None,
    output_log_path: str = None,
    window_size: int = 3,
    num_samples: int = 2,
    model_path: str = None
):
    """
    演示完整的数据处理流程
    
    Args:
        dataset_name: 数据集名称 (iemocap, meld, EmoryNLP)
        original_data_path: 原始数据目录路径
        processed_data_path: 处理后数据目录路径
        output_log_path: 输出日志路径
        window_size: 历史窗口大小（演示用较小值）
        num_samples: 演示的样本数量
        model_path: 预训练模型路径（用于tokenizer演示）
    """
    
    # 设置默认路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if original_data_path is None:
        original_data_path = os.path.join(project_root, "original_data", dataset_name)
    
    if processed_data_path is None:
        processed_data_path = os.path.join(project_root, "processed_data", dataset_name, "window")
    
    if output_log_path is None:
        output_log_path = os.path.join(project_root, "data_pipeline_demo.md")
    
    # 初始化日志记录器
    logger = DataPipelineLogger(output_log_path)
    
    # ==================== 标题 ====================
    logger.log_content.append("# InstructERC 数据处理流程演示")
    logger.log_content.append("")
    logger.log_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log_content.append("")
    logger.log_content.append(f"数据集: **{dataset_name}**")
    logger.log_content.append("")
    logger.log_content.append("---")
    logger.log_content.append("")
    
    # ==================== 配置信息 ====================
    label_set = {
        'iemocap': ['happy', 'sad', 'neutral', 'angry', 'excited', 'frustrated'],
        'meld': ['neutral', 'surprise', 'fear', 'sad', 'joyful', 'disgust', 'angry'],
        'EmoryNLP': ['Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared']
    }
    
    label_text_set = {
        'iemocap': 'happy, sad, neutral, angry, excited, frustrated',
        'meld': 'neutral, surprise, fear, sad, joyful, disgust, angry',
        'EmoryNLP': 'Joyful, Mad, Peaceful, Neutral, Sad, Powerful, Scared'
    }
    
    logger.log_dict("Step 0: 配置信息", {
        "数据集": dataset_name,
        "情感标签集": label_set[dataset_name],
        "标签文本": label_text_set[dataset_name],
        "历史窗口大小": window_size,
        "演示样本数": num_samples
    })
    
    # ==================== Step 1: 原始数据读取 ====================
    pkl_file = os.path.join(original_data_path, f"{dataset_name}.pkl")
    
    logger.log("Step 1: 原始数据读取 (pkl文件)", f"""
**文件路径**: `{pkl_file}`

**读取方式**:
```python
data = pickle.load(open('{pkl_file}', 'rb'))
```

**数据结构说明**:
- `data[0]`: speaker_info - 每个对话中每句话的说话人信息
- `data[1]`: emotion_labels - 每个对话中每句话的情感标签（数字索引）
- `data[2/3]`: sentences - 每个对话中的句子内容
- `data[3/4/5]` 或 `data[4/5/6]`: train_ids, test_ids, valid_ids - 划分的对话ID
""", level=2)
    
    # 尝试读取真实数据
    raw_data = None
    if os.path.exists(pkl_file):
        try:
            raw_data = pickle.load(open(pkl_file, 'rb'))
            
            # 根据数据集类型确定数据结构
            if dataset_name == 'meld':
                all_conv_ids = raw_data[4] + raw_data[5] + raw_data[6]
                sentence_dict = raw_data[3]
                train_ids = raw_data[4]
            else:
                all_conv_ids = raw_data[3] + raw_data[4] + raw_data[5]
                sentence_dict = raw_data[2]
                train_ids = raw_data[3]
            
            # 选取示例对话
            sample_conv_id = train_ids[0] if train_ids else all_conv_ids[0]
            sample_sentences = sentence_dict[sample_conv_id]
            sample_speakers = raw_data[0][sample_conv_id]
            sample_emotions = raw_data[1][sample_conv_id]
            
            logger.log_dict("Step 1.1: 原始数据示例 - 单个对话", {
                "对话ID": sample_conv_id,
                "句子数量": len(sample_sentences),
                "句子内容(前3句)": sample_sentences[:3],
                "说话人信息(前3)": sample_speakers[:3] if isinstance(sample_speakers[0], (str, int)) else [str(s) for s in sample_speakers[:3]],
                "情感标签索引(前3)": sample_emotions[:3]
            }, level=3)
            
        except Exception as e:
            logger.log("Step 1.1: 原始数据读取失败", f"错误: {str(e)}\n\n将使用模拟数据继续演示。", level=3)
            raw_data = None
    else:
        logger.log("Step 1.1: 原始数据文件不存在", f"文件 `{pkl_file}` 不存在，将使用模拟数据继续演示。", level=3)
    
    # 如果无法读取真实数据，使用模拟数据
    if raw_data is None:
        sample_conv_id = "dialogue_001"
        sample_sentences = [
            "also I was the point person on my companys transition from the KL-5 to GR-6 system.",
            "You mustve had your hands full.",
            "That I did. That I did.",
            "So lets talk a little bit about your duties.",
            "My duties? All right."
        ]
        sample_speakers = [0, 1, 0, 1, 0]
        sample_emotions = [0, 0, 0, 0, 1]  # 对应 label_set 的索引
        
        logger.log_dict("Step 1.1: 模拟数据示例", {
            "对话ID": sample_conv_id,
            "句子数量": len(sample_sentences),
            "句子内容": sample_sentences,
            "说话人ID": sample_speakers,
            "情感标签索引": sample_emotions,
            "情感标签文本": [label_set[dataset_name][e] for e in sample_emotions]
        }, level=3)
    
    # ==================== Step 2: Speaker Label 提取 ====================
    logger.log("Step 2: Speaker Label 处理", f"""
**处理逻辑**:
不同数据集有不同的 speaker label 处理方式：

- **IEMOCAP**: 说话人用 'M'/'F' 表示，转换为 0/1
- **MELD/EmoryNLP**: 说话人用 one-hot 向量表示，取索引值

```python
# IEMOCAP 示例
if speaker_label == 'M':
    speaker_id = 0
else:
    speaker_id = 1

# MELD/EmoryNLP 示例
speaker_id = speaker_one_hot.index(1)
```
""", level=2)
    
    # 处理后的 speaker labels
    processed_speakers = sample_speakers if isinstance(sample_speakers[0], int) else [
        s.index(1) if isinstance(s, list) else (0 if s == 'M' else 1) 
        for s in sample_speakers
    ]
    
    logger.log_dict("Step 2.1: Speaker Label 处理结果", {
        "原始 speaker 信息": [str(s) for s in sample_speakers[:3]],
        "处理后的 speaker ID": processed_speakers[:3],
        "格式化后": [f"Speaker_{s}" for s in processed_speakers[:3]]
    }, level=3)
    
    # ==================== Step 3: 对话窗口构建 ====================
    logger.log("Step 3: 历史对话窗口构建", f"""
**窗口机制说明**:
为了让模型理解对话上下文，需要构建历史对话窗口。
当前话语之前的 `window_size` 轮对话会被包含在输入中。

```python
window_size = {window_size}

# 对于对话中的第 conv_turn 轮
index_w = max(conv_turn - window_size, 0)  # 窗口起始位置
history = sentences[index_w : conv_turn + 1]  # 包含历史和当前话语
```
""", level=2)
    
    # 演示不同轮次的窗口
    window_examples = []
    for conv_turn in range(min(len(sample_sentences), num_samples + 2)):
        index_w = max(conv_turn - window_size, 0)
        window = {
            "当前轮次": conv_turn,
            "窗口起始索引": index_w,
            "包含的话语轮次": list(range(index_w, conv_turn + 1)),
            "话语内容": sample_sentences[index_w:conv_turn + 1],
            "说话人": [f"Speaker_{processed_speakers[i]}" for i in range(index_w, conv_turn + 1)]
        }
        window_examples.append(window)
    
    logger.log_dict("Step 3.1: 窗口构建示例", {"窗口示例": window_examples}, level=3)
    
    # ==================== Step 4: Prompt 模板构建 ====================
    logger.log("Step 4: Prompt 指令模板构建", f"""
**Prompt 模板结构**:

```
Now you are expert of sentiment and emotional analysis. 
The following conversation noted between '### ###' involves several speakers. 
### 
    Speaker_X: "utterance_1"
    Speaker_Y: "utterance_2"
    ...
    Speaker_Z: "target_utterance"
### 
Please select the emotional label of <target_utterance> from <label_list>:
```

**模板组成部分**:
1. **系统提示**: "Now you are expert of sentiment and emotional analysis."
2. **任务说明**: "The following conversation noted between '### ###' involves several speakers."
3. **对话内容**: 带有说话人标识的历史对话，用 ### 包围
4. **分类指令**: 要求从标签集中选择目标话语的情感标签
""", level=2)
    
    # 构建完整的 prompt 示例
    prompt_examples = []
    for conv_turn in range(min(len(sample_sentences), num_samples)):
        index_w = max(conv_turn - window_size, 0)
        
        # 构建 prompt
        prompt = 'Now you are expert of sentiment and emotional analysis. '
        prompt += "The following conversation noted between '### ###' involves several speakers. ### "
        
        for i in range(index_w, conv_turn + 1):
            speaker_label = processed_speakers[i]
            utterance = sample_sentences[i]
            prompt += f'\t Speaker_{speaker_label}:"{utterance}"'
        
        target_utterance = prompt.split('\t')[-1]
        prompt += ' ### '
        prompt += f'Please select the emotional label of <{target_utterance}> from <{label_text_set[dataset_name]}>:'
        
        # 获取目标标签
        target_label = label_set[dataset_name][sample_emotions[conv_turn]]
        
        prompt_examples.append({
            "轮次": conv_turn,
            "完整Prompt": prompt,
            "目标标签 (target)": target_label
        })
    
    logger.log_dict("Step 4.1: Prompt 构建示例", {"示例": prompt_examples}, level=3)
    
    # ==================== Step 5: JSON 中间文件 ====================
    logger.log("Step 5: JSON 中间文件生成", f"""
**JSON 文件格式**:
每行一个 JSON 对象，包含 `input` 和 `target` 两个字段。

```json
{{"input": "<完整的Prompt>", "target": "<情感标签>"}}
```

**文件路径**: 
- 训练集: `{processed_data_path}/train.json`
- 测试集: `{processed_data_path}/test.json`  
- 验证集: `{processed_data_path}/valid.json`
""", level=2)
    
    # JSON 格式示例
    json_examples = []
    for ex in prompt_examples:
        json_examples.append({
            "input": ex["完整Prompt"],
            "target": ex["目标标签 (target)"]
        })
    
    logger.log_code("Step 5.1: JSON 文件内容示例", 
                    "\n".join([json.dumps(ex, ensure_ascii=False) for ex in json_examples]),
                    lang="json", level=3)
    
    # ==================== Step 6: 数据加载 ====================
    logger.log("Step 6: JSON 数据加载 (read_data 函数)", f"""
**加载逻辑**:

```python
def read_data(file_name, percent, random_seed):
    f = open(file_name, 'r', encoding='utf-8').readlines()
    data = [json.loads(d) for d in f]
    
    inputs = []
    targets = []
    for d in data:
        if pd.isnull(d['target']) or pd.isna(d['target']):
            continue
        inputs.append(d['input'])
        targets.append(d['target'])
    
    df_data = pd.DataFrame({{'input': inputs, 'output': targets}})
    
    # 随机采样（用于数据比例控制）
    num_samples = int(len(df_data) * percent)
    df_data = df_data.sample(n=num_samples, random_state=random_seed)
    
    return df_data
```
""", level=2)
    
    # DataFrame 示例
    df_example = {
        "columns": ["input", "output"],
        "data": [
            [json_examples[0]["input"][:100] + "...", json_examples[0]["target"]],
        ]
    }
    if len(json_examples) > 1:
        df_example["data"].append([json_examples[1]["input"][:100] + "...", json_examples[1]["target"]])
    
    logger.log_dict("Step 6.1: DataFrame 结构示例", df_example, level=3)
    
    # ==================== Step 7: Dataset 构建 ====================
    logger.log("Step 7: Seq2SeqDataset 构建", f"""
**Dataset 类说明**:

```python
class Seq2SeqDataset(Dataset):
    def __init__(self, args, data, mode):
        inputs = list(data["input"])
        outputs = list(data['output'])
        self.examples = [[i, o] for i, o in zip(inputs, outputs)]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return self.examples[index]  # 返回 [input, output] 对
```

**数据格式**:
- 每个样本是一个列表: `[input_text, target_text]`
- `input_text`: 完整的 Prompt 文本
- `target_text`: 情感标签文本
""", level=2)
    
    dataset_example = {
        "样本数量": len(json_examples),
        "单个样本格式": ["input_text", "target_text"],
        "样本示例": [
            [json_examples[0]["input"][:80] + "...", json_examples[0]["target"]]
        ]
    }
    logger.log_dict("Step 7.1: Dataset 样本结构", dataset_example, level=3)
    
    # ==================== Step 8: Tokenizer 处理 ====================
    logger.log("Step 8: Tokenizer 处理 (preprocess_data_batch)", f"""
**Tokenizer 处理流程** (decoder-only 模型，如 LLaMA):

```python
def preprocess_data_batch(data, tokenizer, args):
    inputs = [d[0] for d in data]   # Prompt 文本
    targets = [d[-1] for d in data]  # 目标标签
    
    # Step 8.1: 对 input 进行 tokenize
    inputs = tokenizer(
        inputs,
        max_length=args.max_length - 1,
        truncation=True
    )
    
    # Step 8.2: 对 target 进行 tokenize (不添加特殊token)
    targets = tokenizer(
        targets,
        add_special_tokens=False,
    )
    
    # Step 8.3: 拼接 input_ids 和 target_ids
    input_ids = inputs['input_ids']
    target_ids = targets['input_ids']
    concat_input = [input_ids[i] + target_ids[i] for i in range(len(input_ids))]
    
    # Step 8.4: 添加 EOS token
    concat_input = [c_ids + [tokenizer.eos_token_id] for c_ids in concat_input]
    
    # Step 8.5: 构建 type_token_ids (区分 input 和 target)
    # 0 表示 input 部分，1 表示 target 部分
    type_token_ids = [[0] * len(input_ids[i]) + [1] * (len(concat_input[i]) - len(input_ids[i])) 
                      for i in range(len(input_ids))]
    
    # Step 8.6: Padding 到相同长度 (左填充)
    max_batch_length = max(len(c) for c in concat_input)
    concat_input = [[tokenizer.pad_token_id] * (max_batch_length - len(ids)) + ids 
                    for ids in concat_input]
    
    # Step 8.7: 构建 labels (input 部分设为 -100，不参与 loss 计算)
    labels = concat_input.clone()
    labels[type_token_ids == 0] = -100
    
    return {{
        "input_ids": concat_input,      # 完整的 token ids
        "attention_mask": attention_mask,
        "type_token_ids": type_token_ids,
        "labels": labels                 # 用于计算 loss
    }}
```
""", level=2)
    
    # 尝试使用真实 tokenizer 演示
    tokenizer = None
    if HAS_TRANSFORMERS and model_path and os.path.exists(model_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.unk_token
        except:
            tokenizer = None
    
    # Tokenizer 示例
    sample_input = json_examples[0]["input"]
    sample_target = json_examples[0]["target"]
    
    if tokenizer:
        # 使用真实 tokenizer
        input_tokens = tokenizer(sample_input, truncation=True, max_length=256)
        target_tokens = tokenizer(sample_target, add_special_tokens=False)
        
        tokenizer_example = {
            "原始 input 文本": sample_input[:100] + "...",
            "原始 target 文本": sample_target,
            "input_ids (前20个)": input_tokens['input_ids'][:20],
            "input_ids 长度": len(input_tokens['input_ids']),
            "target_ids": target_tokens['input_ids'],
            "target_ids 长度": len(target_tokens['input_ids']),
            "拼接后总长度": len(input_tokens['input_ids']) + len(target_tokens['input_ids']) + 1,
            "EOS token id": tokenizer.eos_token_id,
            "PAD token id": tokenizer.pad_token_id
        }
    else:
        # 模拟 tokenizer 结果
        tokenizer_example = {
            "原始 input 文本": sample_input[:100] + "...",
            "原始 target 文本": sample_target,
            "input_ids (模拟, 前20个)": [1, 2045, 366, 526, 4832, 310, 19688, 322, 23023, 1848, 7418, 29889, 450, 1494, 14983, 11682, 1546, 525, 2277, 29937],
            "input_ids 长度 (模拟)": 180,
            "target_ids (模拟)": [21104, 1705],  # "neutral" 的模拟 token ids
            "target_ids 长度 (模拟)": 2,
            "拼接后总长度 (模拟)": 183,
            "说明": "实际 token ids 取决于具体的 tokenizer"
        }
    
    logger.log_dict("Step 8.1: Tokenization 示例", tokenizer_example, level=3)
    
    # 最终输出格式
    final_output = {
        "说明": "这是最终送入模型的数据格式",
        "input_ids": {
            "shape": "[batch_size, max_seq_length]",
            "内容": "PAD...PAD + input_tokens + target_tokens + EOS",
            "示例": "[0, 0, ..., 1, 2045, 366, ..., 21104, 1705, 2]"
        },
        "attention_mask": {
            "shape": "[batch_size, max_seq_length]",
            "内容": "0 表示 PAD 位置，1 表示有效 token",
            "示例": "[0, 0, ..., 1, 1, 1, ..., 1, 1, 1]"
        },
        "type_token_ids": {
            "shape": "[batch_size, max_seq_length]",
            "内容": "0 表示 input (PAD + prompt)，1 表示 target",
            "示例": "[0, 0, ..., 0, 0, 0, ..., 1, 1, 1]"
        },
        "labels": {
            "shape": "[batch_size, max_seq_length]",
            "内容": "input 部分为 -100 (不计算loss)，target 部分为真实 token id",
            "示例": "[-100, -100, ..., -100, -100, ..., 21104, 1705, 2]"
        }
    }
    
    logger.log_dict("Step 8.2: 最终模型输入格式", final_output, level=3)
    
    # ==================== 数据流程图 ====================
    logger.log("数据处理流程总结", f"""
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        InstructERC 数据处理流程                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Step 1: 原始数据  │  {dataset_name}.pkl
│  (Pickle 文件)    │  包含: speakers, emotions, sentences, split_ids
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Step 2-3: 预处理 │  提取 speaker labels
│  窗口构建        │  构建历史对话窗口 (window_size={window_size})
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Step 4: Prompt  │  添加指令模板:
│  模板构建        │  "Now you are expert of sentiment..."
│                  │  对话内容 + 分类指令
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Step 5: JSON    │  {{"input": "<prompt>", "target": "<label>"}}
│  文件生成        │  保存为 train.json, test.json, valid.json
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Step 6: 数据加载 │  read_data() -> DataFrame
│  (DataFrame)     │  columns: ['input', 'output']
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Step 7: Dataset │  Seq2SeqDataset
│  构建            │  examples = [[input, output], ...]
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Step 8: Tokenize│  preprocess_data_batch()
│  处理            │  - tokenize input & target
│                  │  - 拼接: input + target + EOS
│                  │  - 左填充 PAD
│                  │  - 构建 labels (-100 for input)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  模型输入        │  {{
│  (Final Output)  │    "input_ids": tensor,
│                  │    "attention_mask": tensor,
│                  │    "labels": tensor
│                  │  }}
└──────────────────┘
```
""", level=2)
    
    # 保存日志
    logger.save()
    
    return output_log_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='InstructERC 数据处理流程演示')
    parser.add_argument('--dataset', type=str, default='meld', 
                        choices=['iemocap', 'meld', 'EmoryNLP'],
                        help='数据集名称')
    parser.add_argument('--window_size', type=int, default=3,
                        help='历史窗口大小（演示用）')
    parser.add_argument('--num_samples', type=int, default=2,
                        help='演示的样本数量')
    parser.add_argument('--output', type=str, default=None,
                        help='输出日志文件路径')
    parser.add_argument('--model_path', type=str, default=None,
                        help='预训练模型路径（用于tokenizer演示）')
    
    args = parser.parse_args()
    
    output_path = demo_data_pipeline(
        dataset_name=args.dataset,
        window_size=args.window_size,
        num_samples=args.num_samples,
        output_log_path=args.output,
        model_path=args.model_path
    )
    
    print(f"\n演示完成！请查看输出文件: {output_path}")

