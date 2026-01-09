# 对话情感识别（ERC）任务的演进与Decoder模型性能分析

对话情感识别（Emotion Recognition in Conversation, ERC）作为情感计算的重要分支，其核心任务是在给定对话内容的基础上，准确识别出话语所表达的情感类别。

## ERC任务的技术演进
在前大语言模型时代，BERT系列模型在ERC任务中占据了主导地位。这类模型通过Transformer架构提取话语的深层语义特征，并将特征向量送入分类器进行情感分类，在当时达到了最优性能水平。

在Transformer架构普及之前，研究者们曾尝试多种机器学习方法来解决ERC任务。然而，这些方法的效果均未能超越BERT系列模型。值得注意的是，除了纯粹的BERT微调外，大量研究通过在BERT基础上引入图神经网络（GNN）、循环神经网络（RNN）等辅助结构，进一步提升了模型在ERC任务上的整体性能。

从GNN、RNN到Transformer的技术演进过程，其性能提升主要归因于两个关键因素：

### 1. 上下文建模能力的提升
Transformer模型通过自注意力机制（Self-Attention）显著增强了上下文信息的利用效率。这种机制使得模型能够在对话上下文中动态地建立话语到情感的映射关系，实际上构建了一个更加精细和动态的情感词典系统。这种思路与早期直接将特定词汇（如"混蛋"映射为"愤怒"）的静态词典方法一脉相承，但具备更强的上下文适应能力。

### 2. 世界知识与语境理解能力
经过大规模预训练的Transformer模型具备了丰富的世界知识和深层的语境理解能力。这种能力的提升为情感识别提供了重要的语义基础，使模型能够更好地理解复杂的情感表达和隐含的情绪线索。

基于上述分析，本研究假设：具备更大参数量的Decoder-only模型，由于其更丰富的知识储备和更强的生成能力，在ERC任务上的表现可能优于传统的Encoder-only模型。虽然从纯粹的分类任务角度看，Encoder-only模型在理论上有其优势，但由于Decoder模型在生成任务中的巨大商业价值，导致同参数规模的Encoder-only模型在实际中几乎不存在，这也限制了直接的性能对比研究。

此外，考虑到情感计算不仅包含情感理解，还包括情感生成任务，Decoder架构具备统一整个情感计算pipeline的潜力，而Encoder架构则难以胜任生成任务。因此，本研究聚焦于当前主流Decoder模型在情感理解任务中的性能表现。

## 相关研究现状

近年来，针对大语言模型在情感识别任务上的研究逐渐展开。已有工作对ChatGPT 3.5、LLaMA2等模型的情感识别能力进行了系统评估。研究发现，ChatGPT 3.5在零样本（zero-shot）和少样本（few-shot）学习场景下展现出比传统BERT系列模型更强的迁移能力。然而，这些闭源模型缺乏微调能力，限制了其性能的进一步提升。

研究表明，经过全量微调的BERT系列模型凭借其千万级参数量，在特定任务上的性能能够超越ChatGPT 3.5。同时，研究者们通过LoRA（Low-Rank Adaptation）方法对LLaMA2-7B等开源模型进行高效微调，结合辅助任务（如说话者身份识别），在使用相同训练数据量的情况下，已经能够达到BERT系列模型的最优效果。

在MELD这一标准的ERC基准数据集上，基于BERT系列模型的方法通常能够达到65%-70%的加权F1分数。相比之下，未经微调的LLaMA2-7B模型的性能约为30%-40%，经过LoRA微调后可提升至70%左右。

尽管AI技术在过去两年间取得了显著进展，但主流的技术报告多聚焦于模型的数理逻辑和推理能力，对情感计算能力的系统性评估相对缺乏。因此，本研究对最新的Qwen系列模型在ERC任务中的表现进行全面评估。

## 实验设置

本研究选取Qwen2.5-3B-Instruct和Qwen3-4B-Instruct两个代表性的开源模型，在MELD数据集的测试集上进行零样本情感识别能力评估。实验设置如下：

- **评测数据集**：MELD测试集，包含2,610个标注样本
- **评测指标**：情感标签预测的加权F1分数（weighted F1-score）
- **推理框架**：Microsoft Swift框架，采用默认配置
- **计算精度**：bfloat16混合精度
- **情感标签**：neutral, surprise, fear, sad, joyful, disgust, angry## 实验结果与分析

### 数据样例展示

以下是MELD数据集中的一个典型测试样例：

```json
{
  "index": 6, 
  "input_messages": [
    {"role": "system", "content": "Now you are expert of sentiment and emotional analysis."},
    {"role": "user", "content": "The following conversation noted between '### ###' involves several speakers. ### Speaker_0:"Come on, Lydia, you can do it."\t Speaker_0:"Push!"\t Speaker_0:"Push 'em out, push 'em out, harder, harder."\t Speaker_0:"Push 'em out, push 'em out, way out!" ### Please select the emotional label of <Speaker_0:"Push 'em out, push 'em out, way out!"> from <neutral, surprise, fear, sad, joyful, disgust, angry>:"}
  ],
  "ground_truth": "joyful", 
  "model_output": "The emotional label for <Speaker_0:\"Push 'em out, push 'em out, way out!\"> is: **joyful**.

Explanation: The phrase \"Push 'em out, push 'em out, way out!\" is delivered in a strong, encouraging, and energetic tone, typical of a supportive and positive emotional state during a moment of achievement or climax—such as childbirth. The repeated, emphatic commands suggest excitement, confidence, and joy rather than fear, surprise, or negative emotions. While there may be a sense of urgency, the overall tone is one of encouragement and celebration, aligning most closely with **joyful**.",
  "predicted_label": "joyful", 
  "is_correct": true, 
  "extraction_failed": false
}
```

在该样例中，模型成功识别出分娩场景中的鼓励性话语应归类为"joyful"（喜悦），正确理解了语境中的积极情感色彩。
### Qwen2.5-3B-Instruct模型性能分析

Qwen2.5-3B-Instruct模型的零样本情感识别性能汇总如下表所示：

#### 总体性能指标
- **准确率（Accuracy）**：50.11%
- **加权F1分数**：51.90%
- **宏平均F1分数**：32.67%

#### 标签提取统计
- **总样本数**：2,610
- **成功提取**：2,603（99.73%）
- **提取失败**：7（0.27%）

#### 分类性能详述

| 情感类别 | 精确率 | 召回率 | F1分数 | 样本数 |
|----------|--------|--------|--------|--------|
| neutral  | 0.7764 | 0.5143 | 0.6188 | 1,256  |
| surprise | 0.3595 | 0.6192 | 0.4549 | 281    |
| fear     | 0.0909 | 0.1400 | 0.1102 | 50     |
| sad      | 0.3155 | 0.2548 | 0.2819 | 208    |
| joyful   | 0.6086 | 0.4602 | 0.5241 | 402    |
| disgust  | 0.1129 | 0.2059 | 0.1458 | 68     |
| angry    | 0.3730 | 0.6638 | 0.4776 | 345    |

#### 数据集分布分析

**真实标签分布：**
- neutral: 48.1% (1,256)
- surprise: 10.8% (281)
- angry: 13.2% (345)
- joyful: 15.4% (402)
- sad: 8.0% (208)
- disgust: 2.6% (68)
- fear: 1.9% (50)

**模型预测分布：**
- neutral: 31.9% (832)
- angry: 23.5% (614)
- surprise: 18.5% (484)
- joyful: 11.6% (304)
- disgust: 4.8% (124)
- fear: 3.0% (77)
- sad: 6.4% (168)

#### 性能分析

1. **标签提取可靠性**：模型在99.73%的案例中成功提取了情感标签，表明其输出格式稳定性较高。

2. **类别不平衡影响**：模型倾向于将样本预测为训练集中占比较高的类别（neutral），而对小样本类别（fear、disgust）的识别能力较弱。

3. **特定类别表现**：在angry类别上表现相对较好（F1: 0.4776），在neutral类别上精确率较高（0.7764）但召回率偏低，表明模型相对保守地判断neutral类别。

### Qwen3-4B-Instruct模型性能分析

Qwen3-4B-Instruct模型在MELD测试集上的零样本情感识别性能如下：

#### 总体性能指标
- **准确率（Accuracy）**：54.06%
- **加权F1分数**：55.27%
- **宏平均F1分数**：43.96%

#### 标签提取统计
- **总样本数**：2,610
- **成功提取**：2,610（100.00%）
- **提取失败**：0（0.00%）

#### 分类性能详述

| 情感类别 | 精确率 | 召回率 | F1分数 | 样本数 |
|----------|--------|--------|--------|--------|
| neutral  | 0.8238 | 0.5510 | 0.6603 | 1,256  |
| surprise | 0.3411 | 0.7295 | 0.4649 | 281    |
| fear     | 0.2143 | 0.3600 | 0.2687 | 50     |
| sad      | 0.4314 | 0.4231 | 0.4272 | 208    |
| joyful   | 0.4668 | 0.6990 | 0.5598 | 402    |
| disgust  | 0.2397 | 0.4265 | 0.3069 | 68     |
| angry    | 0.6203 | 0.2841 | 0.3897 | 345    |

#### 预测分布分析

**模型预测分布：**
- neutral: 32.2% (840)
- surprise: 23.0% (601)
- joyful: 23.1% (602)
- sad: 7.8% (204)
- angry: 6.1% (158)
- disgust: 4.6% (121)
- fear: 3.2% (84)

#### 性能特点

1. **完美的标签提取**：Qwen3-4B模型在所有样本上都成功提取了情感标签，展现出更强的输出稳定性。

2. **整体性能提升**：相比Qwen2.5-3B，准确率提升了3.95%，加权F1分数提升了3.37%，宏平均F1分数提升了11.29%。

3. **类别性能改善**：在joyful类别上表现显著提升（F1从0.5241提升至0.5598），在neutral类别上的精确率也有所提高（从0.7764提升至0.8238）。

## 综合分析与比较

### 模型性能对比

| 模型 | 参数规模 | 准确率 | 加权F1 | 宏平均F1 | 标签提取成功率 |
|------|----------|--------|--------|----------|----------------|
| Qwen2.5-3B-Instruct | 3B | 50.11% | 51.90% | 32.67% | 99.73% |
| Qwen3-4B-Instruct | 4B | 54.06% | 55.27% | 43.96% | 100.00% |
| LLaMA2-7B (未微调) | 7B | 30-40% | 30-40% | - | - |
| LLaMA2-7B (LoRA微调) | 7B | ~70% | ~70% | - | - |
| BERT系列 (SOTA) | - | - | 65-70% | - | - |

### 关键发现

1. **参数量与性能关系**：Qwen3-4B相比Qwen2.5-3B在各项指标上都有明显提升，验证了更大参数量的Decoder模型在情感理解任务上的优势。

2. **零样本学习潜力**：Qwen系列模型在零样本设置下已经达到了50%以上的准确率，显示出优秀的迁移学习能力。

3. **与微调模型对比**：未经微调的Qwen模型性能仍低于经过微调的LLaMA2和BERT模型，但差距在可接受范围内，具备通过微调进一步提升的潜力。

## 典型案例分析

### 成功案例：分娩场景情感识别

在第6号样本中，两个模型都成功识别出了分娩场景中鼓励性话语的喜悦情感：

**输入**："Push 'em out, push 'em out, way out!"
**真实标签**：joyful
**两模型预测**：joyful ✅

这个案例展示了模型对上下文语境的理解能力，能够识别出在分娩这种特殊场景中，强烈的鼓励话语实际上表达的是喜悦和支持的情感。

### 识别困难案例：微妙情感表达

在第0号样本中：

**输入**："Why do all you're coffee mugs have numbers on the bottom?"
**真实标签**：surprise
**Qwen2.5-3B预测**：angry ❌
**Qwen3-4B预测**：neutral ❌

两个模型都未能正确识别出这个询问中隐含的惊讶情感。Qwen2.5-3B过度解读为愤怒，而Qwen3-4B则过于保守地判断为中性。这表明模型在识别微妙情感表达方面仍有改进空间。

### 模型改进示例：连续鼓励语境的识别

在分娩场景的连续对话中，Qwen3-4B相比Qwen2.5-3B显示出更好的语境一致性：

- **Qwen2.5-3B**：在多轮鼓励话语中，有时将joyful情感误判为angry
- **Qwen3-4B**：更稳定地识别出整个对话序列中的joyful情感

这体现了更大参数模型在维持长距离语境一致性方面的优势。

## 结论与展望

本研究对Qwen2.5-3B-Instruct和Qwen3-4B-Instruct模型在MELD数据集上的零样本情感识别能力进行了系统评估。主要结论如下：

1. **性能表现**：Qwen3-4B模型达到54.06%的准确率和55.27%的加权F1分数，相比Qwen2.5-3B有显著提升。

2. **规模效应**：模型参数量的增加带来了情感识别性能的全面提升，特别是在宏平均F1分数上的大幅提升（+11.29%）表明在少数类别上的识别能力得到了增强。

3. **零样本潜力**：两个模型在零样本设置下都展现出了相当的情感理解能力，表明预训练过程中学习到的世界知识对情感识别任务具有积极的迁移效果。

4. **改进方向**：模型在识别微妙情感和在长距离语境中维持情感一致性方面仍有提升空间。

**未来工作建议**：
- 探索LoRA等参数高效微调方法，进一步提升模型性能
- 研究多模态情感识别，结合语音语调等信息
- 构建更大规模的对话情感数据集，覆盖更丰富的情感类别
- 开发面向特定领域的情感识别模型，如客服对话、心理咨询等场景
