# 日志文件目录

本目录用于存储训练和推理过程中生成的日志文件。

## 📁 日志文件命名规则

- **训练日志**: `train_YYYYMMDD_HHMMSS.log`
- **推理日志**: `inference_YYYYMMDD_HHMMSS.log`

## 📝 日志内容

日志文件包含完整的训练/推理过程输出，包括：

### 训练日志
- 模型加载信息
- 训练参数配置
- 每个 epoch 的训练进度
- 损失值和学习率变化
- 验证集评估结果
- 模型检查点保存信息

### 推理日志
- 模型加载信息
- 测试数据加载
- 推理进度
- 评估指标（准确率、F1 分数等）
- 详细分类报告
- 标签分布统计
- 预测样例展示

## 🔍 查看日志

```bash
# Linux/macOS
tail -f logs/train_20260106_183000.log

# Windows PowerShell
Get-Content logs/train_20260106_183000.log -Wait -Tail 50
```

## 🗑️ 清理旧日志

定期清理旧日志文件以节省磁盘空间：

```bash
# 删除 30 天前的日志
find logs/ -name "*.log" -mtime +30 -delete

# Windows PowerShell
Get-ChildItem logs/*.log | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-30)} | Remove-Item
```

