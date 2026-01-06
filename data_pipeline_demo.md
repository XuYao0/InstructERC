# InstructERC 数据处理流程演示

生成时间: 2026-01-06 15:25:04

数据集: **meld**

---

# Step 0: 配置信息

```json
{
  "数据集": "meld",
  "情感标签集": [
    "neutral",
    "surprise",
    "fear",
    "sad",
    "joyful",
    "disgust",
    "angry"
  ],
  "标签文本": "neutral, surprise, fear, sad, joyful, disgust, angry",
  "历史窗口大小": 3,
  "演示样本数": 2
}
```

---

## Step 1: 原始数据读取 (pkl文件)


**文件路径**: `D:\Project\InstructERC\original_data\meld\meld.pkl`

**读取方式**:
```python
data = pickle.load(open('D:\Project\InstructERC\original_data\meld\meld.pkl', 'rb'))
```

**数据结构说明**:
- `data[0]`: speaker_info - 每个对话中每句话的说话人信息
- `data[1]`: emotion_labels - 每个对话中每句话的情感标签（数字索引）
- `data[2/3]`: sentences - 每个对话中的句子内容
- `data[3/4/5]` 或 `data[4/5/6]`: train_ids, test_ids, valid_ids - 划分的对话ID


---

### Step 1.1: 原始数据示例 - 单个对话

```json
{
  "对话ID": 0,
  "句子数量": 14,
  "句子内容(前3句)": [
    "also I was the point person on my companys transition from the KL-5 to GR-6 system.",
    "You mustve had your hands full.",
    "That I did. That I did."
  ],
  "说话人信息(前3)": [
    "[1, 0, 0, 0, 0, 0, 0, 0, 0]",
    "[0, 1, 0, 0, 0, 0, 0, 0, 0]",
    "[1, 0, 0, 0, 0, 0, 0, 0, 0]"
  ],
  "情感标签索引(前3)": [
    0,
    0,
    0
  ]
}
```

---

## Step 2: Speaker Label 处理


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


---

### Step 2.1: Speaker Label 处理结果

```json
{
  "原始 speaker 信息": [
    "[1, 0, 0, 0, 0, 0, 0, 0, 0]",
    "[0, 1, 0, 0, 0, 0, 0, 0, 0]",
    "[1, 0, 0, 0, 0, 0, 0, 0, 0]"
  ],
  "处理后的 speaker ID": [
    0,
    1,
    0
  ],
  "格式化后": [
    "Speaker_0",
    "Speaker_1",
    "Speaker_0"
  ]
}
```

---

## Step 3: 历史对话窗口构建


**窗口机制说明**:
为了让模型理解对话上下文，需要构建历史对话窗口。
当前话语之前的 `window_size` 轮对话会被包含在输入中。

```python
window_size = 3

# 对于对话中的第 conv_turn 轮
index_w = max(conv_turn - window_size, 0)  # 窗口起始位置
history = sentences[index_w : conv_turn + 1]  # 包含历史和当前话语
```


---

### Step 3.1: 窗口构建示例

```json
{
  "窗口示例": [
    {
      "当前轮次": 0,
      "窗口起始索引": 0,
      "包含的话语轮次": [
        0
      ],
      "话语内容": [
        "also I was the point person on my companys transition from the KL-5 to GR-6 system."
      ],
      "说话人": [
        "Speaker_0"
      ]
    },
    {
      "当前轮次": 1,
      "窗口起始索引": 0,
      "包含的话语轮次": [
        0,
        1
      ],
      "话语内容": [
        "also I was the point person on my companys transition from the KL-5 to GR-6 system.",
        "You mustve had your hands full."
      ],
      "说话人": [
        "Speaker_0",
        "Speaker_1"
      ]
    },
    {
      "当前轮次": 2,
      "窗口起始索引": 0,
      "包含的话语轮次": [
        0,
        1,
        2
      ],
      "话语内容": [
        "also I was the point person on my companys transition from the KL-5 to GR-6 system.",
        "You mustve had your hands full.",
        "That I did. That I did."
      ],
      "说话人": [
        "Speaker_0",
        "Speaker_1",
        "Speaker_0"
      ]
    },
    {
      "当前轮次": 3,
      "窗口起始索引": 0,
      "包含的话语轮次": [
        0,
        1,
        2,
        3
      ],
      "话语内容": [
        "also I was the point person on my companys transition from the KL-5 to GR-6 system.",
        "You mustve had your hands full.",
        "That I did. That I did.",
        "So lets talk a little bit about your duties."
      ],
      "说话人": [
        "Speaker_0",
        "Speaker_1",
        "Speaker_0",
        "Speaker_1"
      ]
    }
  ]
}
```

---

## Step 4: Prompt 指令模板构建


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


---

### Step 4.1: Prompt 构建示例

```json
{
  "示例": [
    {
      "轮次": 0,
      "完整Prompt": "Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speakers. ### \t Speaker_0:\"also I was the point person on my companys transition from the KL-5 to GR-6 system.\" ### Please select the emotional label of < Speaker_0:\"also I was the point person on my companys transition from the KL-5 to GR-6 system.\"> from <neutral, surprise, fear, sad, joyful, disgust, angry>:",
      "目标标签 (target)": "neutral"
    },
    {
      "轮次": 1,
      "完整Prompt": "Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speakers. ### \t Speaker_0:\"also I was the point person on my companys transition from the KL-5 to GR-6 system.\"\t Speaker_1:\"You mustve had your hands full.\" ### Please select the emotional label of < Speaker_1:\"You mustve had your hands full.\"> from <neutral, surprise, fear, sad, joyful, disgust, angry>:",
      "目标标签 (target)": "neutral"
    }
  ]
}
```

---

## Step 5: JSON 中间文件生成


**JSON 文件格式**:
每行一个 JSON 对象，包含 `input` 和 `target` 两个字段。

```json
{"input": "<完整的Prompt>", "target": "<情感标签>"}
```

**文件路径**: 
- 训练集: `D:\Project\InstructERC\processed_data\meld\window/train.json`
- 测试集: `D:\Project\InstructERC\processed_data\meld\window/test.json`  
- 验证集: `D:\Project\InstructERC\processed_data\meld\window/valid.json`


---

### Step 5.1: JSON 文件内容示例

```json
{"input": "Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speakers. ### \t Speaker_0:\"also I was the point person on my companys transition from the KL-5 to GR-6 system.\" ### Please select the emotional label of < Speaker_0:\"also I was the point person on my companys transition from the KL-5 to GR-6 system.\"> from <neutral, surprise, fear, sad, joyful, disgust, angry>:", "target": "neutral"}
{"input": "Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speakers. ### \t Speaker_0:\"also I was the point person on my companys transition from the KL-5 to GR-6 system.\"\t Speaker_1:\"You mustve had your hands full.\" ### Please select the emotional label of < Speaker_1:\"You mustve had your hands full.\"> from <neutral, surprise, fear, sad, joyful, disgust, angry>:", "target": "neutral"}
```

---

## Step 6: JSON 数据加载 (read_data 函数)


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
    
    df_data = pd.DataFrame({'input': inputs, 'output': targets})
    
    # 随机采样（用于数据比例控制）
    num_samples = int(len(df_data) * percent)
    df_data = df_data.sample(n=num_samples, random_state=random_seed)
    
    return df_data
```


---

### Step 6.1: DataFrame 结构示例

```json
{
  "columns": [
    "input",
    "output"
  ],
  "data": [
    [
      "Now you are expert of sentiment and emotional analysis. The following conversation noted between '##...",
      "neutral"
    ],
    [
      "Now you are expert of sentiment and emotional analysis. The following conversation noted between '##...",
      "neutral"
    ]
  ]
}
```

---

## Step 7: Seq2SeqDataset 构建


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


---

### Step 7.1: Dataset 样本结构

```json
{
  "样本数量": 2,
  "单个样本格式": [
    "input_text",
    "target_text"
  ],
  "样本示例": [
    [
      "Now you are expert of sentiment and emotional analysis. The following conversati...",
      "neutral"
    ]
  ]
}
```

---

## Step 8: Tokenizer 处理 (preprocess_data_batch)


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
    
    return {
        "input_ids": concat_input,      # 完整的 token ids
        "attention_mask": attention_mask,
        "type_token_ids": type_token_ids,
        "labels": labels                 # 用于计算 loss
    }
```


---

### Step 8.1: Tokenization 示例

```json
{
  "原始 input 文本": "Now you are expert of sentiment and emotional analysis. The following conversation noted between '##...",
  "原始 target 文本": "neutral",
  "input_ids (模拟, 前20个)": [
    1,
    2045,
    366,
    526,
    4832,
    310,
    19688,
    322,
    23023,
    1848,
    7418,
    29889,
    450,
    1494,
    14983,
    11682,
    1546,
    525,
    2277,
    29937
  ],
  "input_ids 长度 (模拟)": 180,
  "target_ids (模拟)": [
    21104,
    1705
  ],
  "target_ids 长度 (模拟)": 2,
  "拼接后总长度 (模拟)": 183,
  "说明": "实际 token ids 取决于具体的 tokenizer"
}
```

---

### Step 8.2: 最终模型输入格式

```json
{
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
```

---

## 数据处理流程总结


```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        InstructERC 数据处理流程                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Step 1: 原始数据  │  meld.pkl
│  (Pickle 文件)    │  包含: speakers, emotions, sentences, split_ids
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Step 2-3: 预处理 │  提取 speaker labels
│  窗口构建        │  构建历史对话窗口 (window_size=3)
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
│  Step 5: JSON    │  {"input": "<prompt>", "target": "<label>"}
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
│  模型输入        │  {
│  (Final Output)  │    "input_ids": tensor,
│                  │    "attention_mask": tensor,
│                  │    "labels": tensor
│                  │  }
└──────────────────┘
```


---
