# ms-swift 数据集 - MELD 情感识别

本目录包含已转换为 ms-swift 框架格式的 MELD 数据集。

## 📊 数据统计

| 数据集 | 样本数 | 文件 |
|--------|--------|------|
| 训练集 | 9,989 | `meld/train.jsonl` |
| 测试集 | 2,610 | `meld/test.jsonl` |
| 验证集 | 1,109 | `meld/valid.jsonl` |
| **总计** | **13,708** | - |

## 🏷️ 标签分布 (训练集)

| 标签 | 数量 | 占比 |
|------|------|------|
| neutral | 4,710 | 47.2% |
| joyful | 1,743 | 17.4% |
| surprise | 1,205 | 12.1% |
| angry | 1,109 | 11.1% |
| sad | 683 | 6.8% |
| disgust | 271 | 2.7% |
| fear | 268 | 2.7% |

## 📝 数据格式

使用 ms-swift 标准的 **messages 格式** (OpenAI 风格):

```json
{
  "messages": [
    {"role": "system", "content": "Now you are expert of sentiment and emotional analysis."},
    {"role": "user", "content": "The following conversation noted between '### ###' involves several speakers. ### Speaker_0:\"...\"\t Speaker_1:\"...\" ### Please select the emotional label of <...> from <neutral, surprise, fear, sad, joyful, disgust, angry>:"},
    {"role": "assistant", "content": "neutral"}
  ]
}
```

## 🚀 快速开始

### 1. 安装 ms-swift

```bash
pip install ms-swift
```

### 2. 开始训练

**Linux/macOS:**
```bash
bash swift_train.sh
```

**Windows PowerShell:**
```powershell
.\swift_train.ps1
```

**或使用命令行:**
```bash
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset swift_data/meld/train.jsonl \
    --val_dataset swift_data/meld/valid.jsonl \
    --output_dir experiments/swift_meld \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4 \
    --lora_rank 16
```

### 3. 推理评估

```bash
python swift_infer.py --model_path experiments/swift_meld/checkpoint-xxx
```

## 📁 目录结构

```
InstructERC/
├── swift_data/
│   └── meld/
│       ├── train.jsonl     # 训练数据
│       ├── test.jsonl      # 测试数据
│       └── valid.jsonl     # 验证数据
├── swift_train.sh          # Linux/Mac 训练脚本
├── swift_train.ps1         # Windows 训练脚本
├── swift_infer.py          # 推理评估脚本
└── code/
    └── convert_to_swift.py # 数据转换脚本
```

## 🔧 重新生成数据

如需修改窗口大小或其他参数，可重新运行转换脚本:

```bash
python code/convert_to_swift.py \
    --dataset meld \
    --window 12 \
    --format messages \
    --output_dir swift_data/meld
```

**参数说明:**
- `--dataset`: 数据集名称 (meld, iemocap, EmoryNLP)
- `--window`: 历史对话窗口大小 (默认: 12)
- `--format`: 输出格式 (messages, query_response, alpaca)
- `--output_dir`: 输出目录

## 📌 支持的模型

ms-swift 支持多种模型，推荐使用:

- **Qwen 系列**: Qwen2.5-7B-Instruct, Qwen3-8B 等
- **LLaMA 系列**: Llama-3.1-8B-Instruct 等
- **ChatGLM 系列**: ChatGLM4-9B 等

详见: [ms-swift 支持的模型列表](https://swift.readthedocs.io/zh-cn/latest/)

