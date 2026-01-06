# ============================================================
# ms-swift 训练脚本 - MELD 情感识别任务 (Windows PowerShell)
# ============================================================
#
# 使用方法:
#   1. 确保已安装 ms-swift: pip install ms-swift
#   2. 修改下面的参数
#   3. 运行: .\swift_train.ps1
#
# 数据集路径: swift_data/meld/
#   - train.jsonl: 9989 samples
#   - test.jsonl: 2610 samples  
#   - valid.jsonl: 1109 samples
# ============================================================

# ==================== 配置参数 ====================

# 模型路径 (可以是 ModelScope/HuggingFace 模型ID 或本地路径)
$MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
# $MODEL_PATH = "Qwen/Qwen3-8B"
# $MODEL_PATH = "D:\models\your_local_model"

# 数据集路径
$TRAIN_DATASET = "swift_data/meld/train.jsonl"
$VAL_DATASET = "swift_data/meld/valid.jsonl"

# 输出目录
$OUTPUT_DIR = "experiments/swift_meld_qwen"

# ==================== LoRA 训练 ====================

swift sft `
    --model $MODEL_PATH `
    --train_type lora `
    --dataset $TRAIN_DATASET `
    --val_dataset $VAL_DATASET `
    --output_dir $OUTPUT_DIR `
    --num_train_epochs 3 `
    --per_device_train_batch_size 4 `
    --per_device_eval_batch_size 4 `
    --gradient_accumulation_steps 8 `
    --learning_rate 2e-4 `
    --warmup_ratio 0.1 `
    --logging_steps 10 `
    --eval_steps 100 `
    --save_steps 100 `
    --save_total_limit 3 `
    --lora_rank 16 `
    --lora_alpha 32 `
    --lora_dropout 0.05 `
    --target_modules all-linear `
    --max_length 2048 `
    --gradient_checkpointing true

