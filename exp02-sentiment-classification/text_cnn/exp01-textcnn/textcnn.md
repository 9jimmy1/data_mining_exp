# 实验1：基于TextCNN的Amazon数据集情感分析 总结报告

## 一、实验完成情况

实验已成功完成！TextCNN模型在Amazon评论数据集上实现了82.52%的测试准确率。以下是完整的总结和各个程序的详细注释。

## 二、各程序文件详解

### 1. `data_loader.py` - 数据加载与预处理

```python
"""
数据加载和预处理模块
功能：加载CSV格式的Amazon评论数据，进行文本预处理，构建词汇表，创建PyTorch数据集
关键步骤：
1. 使用自定义CSV解析器处理混合格式（train.csv和dev.csv/test.csv格式不同）
2. 文本清洗：去除非字母字符、HTML标签、转换为小写
3. 使用nltk分词器进行分词
4. 构建词汇表：统计词频，为每个词分配索引
5. 将文本转换为索引序列，并进行填充/截断
6. 创建PyTorch Dataset和DataLoader
"""

# 主要类和方法：
# - AmazonReviewDataset: PyTorch数据集类
# - load_csv_file(): 自定义CSV解析器
# - create_data_loaders(): 创建训练/验证/测试数据加载器
```

### 2. `model.py` - TextCNN模型定义

```python
"""
TextCNN模型定义（Kim, 2015）
功能：实现TextCNN模型，用于文本分类
模型结构：
1. 词嵌入层（Embedding Layer）：将单词索引转换为词向量
2. 卷积层（Convolutional Layer）：使用多种尺寸的卷积核（3,4,5）提取特征
3. 最大池化层（Max Pooling）：提取每个特征图的最显著特征
4. 全连接层（Fully Connected）：将特征映射到分类结果

GPU加速：模型继承自nn.Module，自动支持GPU
"""

class TextCNN(nn.Module):
    """
    参数说明：
    - vocab_size: 词汇表大小
    - embedding_dim: 词向量维度（默认300）
    - filter_sizes: 卷积核大小列表（默认[3,4,5]）
    - num_filters: 每种卷积核的数量（默认100）
    - dropout_rate: Dropout比率（防止过拟合）
    - pretrained_embeddings: 预训练词向量（可选）
    """
```

### 3. `train.py` - 模型训练与评估

```python
"""
模型训练、验证和测试脚本
功能：训练TextCNN模型，监控过拟合，评估模型性能
主要流程：
1. 设置GPU/CPU设备
2. 加载数据
3. 创建模型
4. 训练循环（10个epoch）
5. 验证和测试
6. 保存结果

GPU加速实现：
- setup_device()函数自动检测并配置GPU
- 使用torch.cuda.is_available()检测GPU
- 将模型和数据移到GPU：model.to(device), data.to(device)
- 启用cuDNN加速：torch.backends.cudnn.benchmark = True

过拟合监测方法：
1. 分离训练集、验证集、测试集
2. 每轮训练后验证，保存验证集最佳模型
3. 监控训练集和验证集的准确率差异
4. 使用学习率调度器（ReduceLROnPlateau）
5. 添加Dropout层（dropout_rate=0.5）

测试流程：
1. 加载验证集上表现最佳的模型
2. 在独立的测试集上评估
3. 计算准确率、精确率、召回率、F1分数
4. 绘制混淆矩阵
"""
```

### 4. `utils.py` - 工具函数

```python
"""
工具函数模块
功能：提供GPU信息检查、模型参数统计、结果保存等辅助功能
"""

# 主要功能：
# - print_gpu_info(): 打印GPU信息和内存使用情况
# - count_parameters(): 统计模型参数数量
# - save_classification_report(): 保存分类报告
# - set_seed(): 设置随机种子确保可重复性
```

### 5. `fix_csv_format.py` - 数据格式修复

```python
"""
CSV格式修复脚本（已使用，可保留为参考）
功能：修复原始CSV文件的格式不一致问题
问题：train.csv每列都有双引号，dev.csv/test.csv只有部分有引号
解决：统一为所有字段都用双引号包围的标准CSV格式
"""
```

## 三、实验流程详解

### 1. GPU加速实现

```python
# GPU检测和设置
def setup_device():
    if torch.cuda.is_available():  # 检测CUDA是否可用
        device = torch.device("cuda:0")  # 使用第一个GPU
        torch.cuda.set_device(device)  # 设置当前设备
        # 启用cuDNN加速
        torch.backends.cudnn.benchmark = True
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    return device

# 在训练循环中使用GPU
sequences = sequences.to(device)  # 将数据移动到GPU
labels = labels.to(device)  # 将标签移动到GPU
model = model.to(device)  # 将模型移动到GPU
```

### 2. 过拟合监测策略

```python
# 1. 数据划分：训练集（2000条）、验证集（1001条）、测试集（1001条）
# 2. 早停法：保存验证集最佳模型
if val_acc > best_val_accuracy:
    best_val_accuracy = val_acc
    save_model(model, vocab, epoch, optimizer, val_loss, val_acc, save_dir)

# 3. 学习率调度：验证损失不改善时降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',  # 监控损失
    factor=0.5,  # 学习率衰减因子
    patience=2   # 容忍2个epoch不改善
)

# 4. 监控指标差异
print(f"训练准确率: {train_acc:.4f} | 验证准确率: {val_acc:.4f}")
# 训练准确率远高于验证准确率时可能过拟合
```

### 3. 模型测试流程

```python
# 1. 加载最佳模型
checkpoint = torch.load(best_model_path)
best_model.load_state_dict(checkpoint['model_state_dict'])

# 2. 在测试集上评估
test_loss, test_acc, test_precision, test_recall, test_f1, test_preds, test_labels = validate(
    best_model, device, test_loader, criterion, mode="测试"
)

# 3. 计算各项指标
print(f"测试准确率: {test_acc:.4f}")
print(f"测试精确率: {test_precision:.4f}")
print(f"测试召回率: {test_recall:.4f}")
print(f"测试F1分数: {test_f1:.4f}")

# 4. 绘制混淆矩阵
plot_confusion_matrix(test_preds, test_labels, config['log_dir'])
```

## 四、实验结果

### 关键指标
- **最佳验证准确率**: 85.21%（第3个epoch）
- **测试准确率**: 82.52%
- **F1分数**: 82.50%
- **训练时间**: 约760秒（GPU加速）

### 过拟合分析
从训练过程可以看出明显的过拟合现象：
- Epoch 1-3: 训练和验证准确率同步上升
- Epoch 4-10: 训练准确率接近100%，但验证准确率停滞在83-85%
- 最佳模型出现在第3个epoch（早停法有效）

### GPU加速效果
- 第1个epoch: 732.4秒（包含数据加载和初始化）
- 后续每个epoch: 2-4秒
- GPU利用率：RTX 5060 Laptop GPU，8.55GB显存

## 五、PPT制作指南

### 第1页：封面
**标题**: 实验1：基于TextCNN的Amazon评论情感分析  
**副标题**: GPU加速的深度学习文本分类  
**内容**:
- 姓名/学号
- 课程名称
- 日期

### 第2页：实验概述
**要点**:
1. **目标**: 使用TextCNN进行情感二分类（正面/负面）
2. **数据集**: Amazon产品评论（360万条，采样2000条训练）
3. **技术**: PyTorch、GPU加速、TextCNN
4. **成果**: 82.52%测试准确率

### 第3页：数据预处理
**要点**:
1. **数据格式**: CSV文件，包含polarity、title、text三列
2. **预处理步骤**:
   ```python
   # 文本清洗
   text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)  # 去除非字母字符
   text = text.lower()  # 转换为小写
   
   # 分词和序列化
   tokens = word_tokenize(text)  # 使用nltk分词
   sequence = [vocab.get(token, vocab['<UNK>']) for token in tokens]  # 转换为索引
   ```

### 第4页：TextCNN模型架构
**图示**: TextCNN模型结构图
**要点**:
1. **嵌入层**: 300维词向量
2. **卷积层**: 3种卷积核（3,4,5），各100个
3. **池化层**: 最大池化
4. **全连接层**: 二分类输出

```python
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, 
                 filter_sizes=[3,4,5], num_filters=100):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), 2)
```

### 第5页：GPU加速实现
**要点**:
1. **自动检测GPU**: `torch.cuda.is_available()`
2. **数据迁移**: `.to(device)`方法
3. **cuDNN加速**: `torch.backends.cudnn.benchmark = True`
4. **显存管理**: 自动管理

**代码片段**:
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # 模型移到GPU
sequences = sequences.to(device)  # 数据移到GPU
```

### 第6页：过拟合监测策略
**要点**:
1. **验证集监控**: 使用dev.csv作为验证集
2. **早停法**: 保存验证集最佳模型
3. **学习率调度**: 损失不改善时降低学习率
4. **Dropout**: 0.5的dropout率

**图示**: 训练/验证准确率对比图

### 第7页：训练过程与结果
**表格**: 训练过程记录
| Epoch | 训练准确率 | 验证准确率 | 学习率 |
|-------|------------|------------|--------|
| 1     | 63.75%     | 77.02%     | 0.001  |
| 2     | 80.85%     | 82.82%     | 0.001  |
| 3     | 91.85%     | 85.21%     | 0.001  |
| ...   | ...        | ...        | ...    |

**最终结果**: 测试准确率82.52%，F1分数82.50%

### 第8页：混淆矩阵分析
**图示**: 混淆矩阵图
**分析**:
- 真实负面494条，正确预测408条（82.6%）
- 真实正面507条，正确预测418条（82.4%）
- 模型在正负类上表现均衡

### 第9页：性能优化建议
**要点**:
1. **数据层面**: 增加训练数据，数据增强
2. **模型层面**: 尝试BERT等预训练模型，调整超参数
3. **训练层面**: 调整学习率策略，添加L2正则化
4. **工程层面**: 混合精度训练，分布式训练

### 第10页：总结与展望
**总结**:
1. 成功实现了基于TextCNN的情感分析模型
2. GPU加速显著提升训练速度
3. 实现了过拟合监测和早停机制

**展望**:
1. 扩展到多分类问题
2. 尝试更先进的模型架构
3. 部署到实际应用场景

## 六、文件清单说明

### 必需文件（实验核心）:
1. `exp01-textcnn/src/data_loader.py` - 数据加载
2. `exp01-textcnn/src/model.py` - 模型定义
3. `exp01-textcnn/src/train.py` - 训练脚本
4. `exp01-textcnn/src/utils.py` - 工具函数

### 辅助文件（可保留参考）:
1. `check_csv_format.py` - CSV格式检查（已使用）
2. `fix_csv_format.py` - CSV格式修复（已使用）

### 输出文件:
1. `exp01-textcnn/saved_models/` - 保存的模型
2. `exp01-textcnn/logs/` - 日志和图表
3. `datasets/fixed/` - 修复后的CSV文件
