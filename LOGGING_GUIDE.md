# 📋 日志功能使用指南

本项目已为训练和推理脚本添加完整的日志功能，所有输出将同时显示在控制台并保存到日志文件中。

## 🎯 功能特性

✅ **双重输出**：同时输出到控制台和日志文件  
✅ **自动时间戳**：日志文件名包含时间戳，便于管理  
✅ **完整记录**：记录所有训练/推理过程的输出  
✅ **自动创建目录**：自动创建 `logs/` 目录  
✅ **中文支持**：完美支持中文字符编码

## 📂 日志文件存储

所有日志文件存储在 `logs/` 目录下：

```
InstructERC/
├── logs/
│   ├── train_20260106_183000.log       # 训练日志
│   ├── train_20260106_200000.log
│   ├── inference_20260106_183500.log   # 推理日志
│   ├── inference_20260106_201000.log
│   └── README.md                        # 日志目录说明
```

## 🚀 使用方法

### 1️⃣ 训练脚本

#### Linux/macOS

```bash
# 运行训练（日志自动保存）
bash swift_train.sh

# 日志文件会自动生成在 logs/train_YYYYMMDD_HHMMSS.log
```

#### Windows PowerShell

```powershell
# 运行训练（日志自动保存）
.\swift_train.ps1

# 日志文件会自动生成在 logs\train_YYYYMMDD_HHMMSS.log
```

**输出示例：**
```
==================================================
MELD Emotion Recognition - Training
==================================================
训练日志将保存到: logs/train_20260106_183000.log
==================================================

[训练过程输出...]

==================================================
训练完成！日志已保存到: logs/train_20260106_183000.log
==================================================
```

### 2️⃣ 推理脚本

#### 基础用法（自动生成日志文件）

```bash
# 日志会自动保存到 logs/inference_YYYYMMDD_HHMMSS.log
python swift_infer.py \
    --model_path /path/to/model \
    --test_file swift_data/meld/test.jsonl \
    --batch_size 8
```

#### 指定日志文件路径

```bash
# 自定义日志文件位置
python swift_infer.py \
    --model_path /path/to/model \
    --test_file swift_data/meld/test.jsonl \
    --batch_size 8 \
    --log_file my_custom_log.log
```

**输出示例：**
```
============================================================
MELD Emotion Recognition - Inference & Evaluation
============================================================
Log file: logs/inference_20260106_183500.log
Model: /home/user/data/qwen2.5-3b-instruct
Test data: swift_data/meld/test.jsonl
Batch size: 8
============================================================

Loading model from: /home/user/data/qwen2.5-3b-instruct
Loading test data from: swift_data/meld/test.jsonl
Loaded 2610 test samples

Running inference...
[推理过程...]

============================================================
Evaluation Results
============================================================

Accuracy: 65.23%
Weighted F1: 63.45%
Macro F1: 58.92%

[详细结果...]

Results saved to: /home/user/data/qwen2.5-3b-instruct/eval_results.json
Log saved to: logs/inference_20260106_183500.log
============================================================
Inference completed successfully!
```

## 📊 日志内容说明

### 训练日志内容

- ✅ 训练开始时间和配置信息
- ✅ 模型加载和初始化信息
- ✅ 数据集加载统计
- ✅ 训练过程中的损失值
- ✅ 每个 epoch 的进度条
- ✅ 验证集评估结果
- ✅ 模型检查点保存信息
- ✅ 训练完成时间

### 推理日志内容

- ✅ 推理配置信息
- ✅ 模型加载信息
- ✅ 测试数据统计
- ✅ 批量推理进度
- ✅ 评估指标（准确率、F1 等）
- ✅ 详细分类报告
- ✅ 真实标签和预测标签分布
- ✅ 预测样例展示（包含原始输出）

## 🔍 查看和监控日志

### 实时查看日志（训练时推荐）

#### Linux/macOS
```bash
# 实时追踪最新日志
tail -f logs/train_20260106_183000.log

# 显示最后 100 行
tail -n 100 logs/train_20260106_183000.log
```

#### Windows PowerShell
```powershell
# 实时追踪最新日志
Get-Content logs/train_20260106_183000.log -Wait -Tail 50

# 显示最后 100 行
Get-Content logs/train_20260106_183000.log -Tail 100
```

### 搜索日志内容

#### Linux/macOS
```bash
# 搜索包含 "Accuracy" 的行
grep "Accuracy" logs/inference_20260106_183500.log

# 搜索包含 "error" 的行（不区分大小写）
grep -i "error" logs/train_20260106_183000.log
```

#### Windows PowerShell
```powershell
# 搜索包含 "Accuracy" 的行
Select-String -Path logs/inference_20260106_183500.log -Pattern "Accuracy"

# 搜索包含 "error" 的行（不区分大小写）
Select-String -Path logs/train_20260106_183000.log -Pattern "error" -CaseSensitive:$false
```

## 📈 日志分析技巧

### 1. 提取训练损失变化
```bash
# Linux/macOS
grep "loss" logs/train_20260106_183000.log | grep "step"

# Windows
Select-String -Path logs/train_20260106_183000.log -Pattern "loss.*step"
```

### 2. 查看所有评估结果
```bash
# Linux/macOS
grep -A 10 "Evaluation Results" logs/*.log

# Windows
Select-String -Path logs/*.log -Pattern "Evaluation Results" -Context 0,10
```

### 3. 比较多次实验的准确率
```bash
# Linux/macOS
grep "Accuracy:" logs/inference_*.log

# Windows
Select-String -Path logs/inference_*.log -Pattern "Accuracy:"
```

## 🗑️ 日志管理

### 列出所有日志文件

```bash
# Linux/macOS
ls -lh logs/*.log

# Windows
Get-ChildItem logs/*.log | Format-Table Name, Length, LastWriteTime
```

### 清理旧日志

```bash
# Linux/macOS - 删除 30 天前的日志
find logs/ -name "*.log" -mtime +30 -delete

# Windows PowerShell - 删除 30 天前的日志
Get-ChildItem logs/*.log | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-30)} | Remove-Item

# 仅保留最新的 5 个日志文件
# Linux/macOS
ls -t logs/train_*.log | tail -n +6 | xargs rm -f

# Windows PowerShell
Get-ChildItem logs/train_*.log | Sort-Object LastWriteTime -Descending | Select-Object -Skip 5 | Remove-Item
```

### 压缩旧日志节省空间

```bash
# Linux/macOS
gzip logs/*.log

# Windows PowerShell
Compress-Archive -Path logs/*.log -DestinationPath logs/archive_$(Get-Date -Format 'yyyyMMdd').zip
```

## ⚙️ 高级配置

### 修改日志格式

如需修改日志格式，编辑 `swift_infer.py` 中的 `setup_logger` 函数：

```python
file_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',  # 修改这里
    datefmt='%Y-%m-%d %H:%M:%S'  # 修改时间格式
)
```

### 修改日志级别

```python
# 只记录警告和错误
logger.setLevel(logging.WARNING)

# 记录详细调试信息
logger.setLevel(logging.DEBUG)
```

## 📌 注意事项

1. **磁盘空间**：长时间训练会产生大量日志，定期清理
2. **编码问题**：所有日志文件使用 UTF-8 编码
3. **权限问题**：确保程序有写入 `logs/` 目录的权限
4. **并行训练**：多卡训练时每个进程可能产生独立的日志

## 🐛 故障排查

### 问题：日志文件没有生成

**解决方案：**
```bash
# 检查 logs 目录是否存在
ls -la logs/

# 手动创建目录
mkdir -p logs

# 检查权限
chmod 755 logs/
```

### 问题：日志中文乱码

**解决方案：**
```bash
# 使用正确的编码查看
# Linux/macOS
cat logs/train_20260106_183000.log

# Windows - 设置控制台编码为 UTF-8
chcp 65001
type logs\train_20260106_183000.log
```

### 问题：日志文件过大

**解决方案：**
- 使用日志轮转（log rotation）
- 定期压缩或删除旧日志
- 调整日志级别，减少输出

## 💡 最佳实践

1. ✅ **定期备份重要的训练日志**
2. ✅ **为重要实验添加描述性的日志文件名**
3. ✅ **在日志中添加实验配置信息**
4. ✅ **使用版本控制管理日志分析脚本**
5. ✅ **定期清理不需要的日志文件**

## 📚 相关文档

- [swift_infer.py](swift_infer.py) - 推理脚本（包含日志功能）
- [swift_train.sh](swift_train.sh) - Linux/macOS 训练脚本
- [swift_train.ps1](swift_train.ps1) - Windows 训练脚本
- [logs/README.md](logs/README.md) - 日志目录说明

---

如有问题或建议，请查看项目文档或提交 Issue。

