"""
训练脚本
包含模型训练、验证、测试和GPU加速功能
核心训练流程：数据加载 -> 模型定义 -> 训练 -> 验证 -> 测试 -> 保存
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import sys

# ========== 解决matplotlib中文显示问题 ==========
# 设置中文字体，避免中文显示为方框
import matplotlib
# 尝试不同的中文字体设置方式
try:
    # 方法1: 使用系统字体（Windows）
    if sys.platform.startswith('win'):
        # Windows系统，使用微软雅黑
        matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    else:
        # Linux/Mac系统，使用常见中文字体
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei', 'STHeiti']

    # 确保负号正常显示
    matplotlib.rcParams['axes.unicode_minus'] = False
    print("✓ 已设置中文字体支持")
except Exception as e:
    print(f"⚠ 中文字体设置失败，将使用默认字体: {e}")
# ==============================================

# 导入自定义模块
from data_loader import create_data_loaders
from model import TextCNN

def setup_device():
    """
    设置训练设备（GPU/CPU）
    优先使用GPU，如果GPU可用则自动使用GPU加速
    返回: device - 训练设备

    原理：PyTorch通过CUDA调用NVIDIA GPU进行并行计算
    """
    # 检查CUDA是否可用：需要安装NVIDIA驱动和CUDA工具包
    if torch.cuda.is_available():
        # 获取GPU数量：支持多GPU训练
        gpu_count = torch.cuda.device_count()
        print(f"找到 {gpu_count} 个GPU:")

        # 显示所有可用GPU的信息：帮助选择合适的GPU
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # 转换为GB
            print(f"  GPU {i}: {gpu_name}, 显存: {gpu_memory:.2f} GB")

        # 选择设备（默认使用第一个GPU）
        device = torch.device(f"cuda:0")

        # 设置当前设备：指定使用哪个GPU
        torch.cuda.set_device(device)

        # 获取当前设备信息
        current_device = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device)
        print(f"\n使用设备: GPU {current_device} ({current_device_name})")

        # 设置cuDNN加速（如果可用）：NVIDIA的深度神经网络加速库
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
            print("cuDNN加速已启用")

    else:
        device = torch.device("cpu")
        print("警告: 未找到GPU，使用CPU训练（速度会很慢）")
        print("建议: 安装NVIDIA GPU驱动和CUDA工具包以获得GPU加速")

    return device

def train_epoch(model, device, train_loader, optimizer, criterion, epoch, num_epochs):
    """
    训练一个epoch

    参数:
        model: 模型
        device: 设备（GPU/CPU）
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        epoch: 当前epoch
        num_epochs: 总epoch数

    返回:
        avg_loss: 平均损失
        accuracy: 准确率

    训练过程：前向传播 -> 计算损失 -> 反向传播 -> 参数更新
    """
    model.train()  # 设置模型为训练模式：启用dropout和batch normalization
    total_loss = 0
    all_preds = []
    all_labels = []

    # 进度条信息
    total_batches = len(train_loader)

    print(f"\nEpoch {epoch+1}/{num_epochs} - 训练中...")
    start_time = time.time()

    for batch_idx, (sequences, labels) in enumerate(train_loader):
        # 1. 将数据移动到设备（GPU/CPU）：数据需要和模型在同一设备
        sequences = sequences.to(device)
        labels = labels.to(device)

        # 2. 梯度清零：每次迭代前清空梯度，防止梯度累积
        optimizer.zero_grad()

        # 3. 前向传播：计算模型输出
        outputs = model(sequences)

        # 4. 计算损失：衡量预测与真实标签的差异
        loss = criterion(outputs, labels)

        # 5. 反向传播：计算梯度（链式法则）
        loss.backward()

        # 6. 更新参数：根据梯度调整模型参数
        optimizer.step()

        # 7. 记录损失和预测：用于监控训练过程
        total_loss += loss.item()  # loss.item()获取标量值

        # 获取预测结果：取最大概率的类别
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())  # 转移到CPU并转换为numpy
        all_labels.extend(labels.cpu().numpy())

        # 每10个批次打印一次进度：实时监控训练状态
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
            elapsed_time = time.time() - start_time
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = accuracy_score(all_labels, all_preds)

            # 计算进度百分比
            progress = (batch_idx + 1) / total_batches * 100

            print(f"  进度: {progress:.1f}% | "
                  f"批次: {batch_idx+1}/{total_batches} | "
                  f"损失: {avg_loss:.4f} | "
                  f"准确率: {accuracy:.4f} | "
                  f"时间: {elapsed_time:.1f}s", end='\r')

    # 计算整个epoch的指标
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    elapsed_time = time.time() - start_time
    print(f"\n  训练完成 | 平均损失: {avg_loss:.4f} | 准确率: {accuracy:.4f} | 时间: {elapsed_time:.1f}s")

    return avg_loss, accuracy

def validate(model, device, val_loader, criterion, mode="验证"):
    """
    验证或测试模型

    参数:
        model: 模型
        device: 设备（GPU/CPU）
        val_loader: 验证/测试数据加载器
        criterion: 损失函数
        mode: 模式（"验证" 或 "测试"）

    返回:
        avg_loss: 平均损失
        accuracy: 准确率
        precision: 精确率
        recall: 召回率
        f1: F1分数
        all_preds: 所有预测结果
        all_labels: 所有真实标签

    验证过程：不计算梯度，只进行前向传播
    """
    model.eval()  # 设置模型为评估模式：禁用dropout和batch normalization
    total_loss = 0
    all_preds = []
    all_labels = []

    print(f"{mode}中...")
    start_time = time.time()

    # 不计算梯度，加速推理：节省内存和计算资源
    with torch.no_grad():
        for sequences, labels in val_loader:
            # 将数据移动到设备
            sequences = sequences.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(sequences)

            # 计算损失：用于监控模型性能
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 获取预测结果
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标：全面评估模型性能
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    elapsed_time = time.time() - start_time
    print(f"  {mode}完成 | 平均损失: {avg_loss:.4f} | 准确率: {accuracy:.4f} | "
          f"精确率: {precision:.4f} | 召回率: {recall:.4f} | F1分数: {f1:.4f} | "
          f"时间: {elapsed_time:.1f}s")

    return avg_loss, accuracy, precision, recall, f1, all_preds, all_labels

def save_model(model, vocab, epoch, optimizer, loss, accuracy, save_dir):
    """
    保存模型和训练状态

    参数:
        model: 模型
        vocab: 词汇表
        epoch: 当前epoch
        optimizer: 优化器
        loss: 损失值
        accuracy: 准确率
        save_dir: 保存目录

    保存内容：模型参数、优化器状态、词汇表、训练配置
    """
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 生成文件名：包含时间和epoch信息
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"textcnn_epoch{epoch+1}_{timestamp}.pth")

    # 保存模型状态：可以恢复训练或用于推理
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),  # 模型参数
        'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
        'loss': loss,
        'accuracy': accuracy,
        'vocab': vocab,  # 词汇表（用于后续文本处理）
        'vocab_size': len(vocab),
        'model_config': {  # 模型配置（用于重建模型）
            'embedding_dim': model.embedding_dim,
            'num_classes': model.num_classes,
            'filter_sizes': model.filter_sizes,
            'num_filters': model.num_filters,
            'dropout_rate': model.dropout.p if hasattr(model.dropout, 'p') else 0.5
        }
    }, model_path)

    print(f"模型已保存到: {model_path}")
    return model_path

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir):
    """
    绘制训练历史图表
    用于监控训练过程和检测过拟合

    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
        save_dir: 保存目录

    过拟合检测：训练损失持续下降但验证损失上升
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    # 创建图表：损失曲线和准确率曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制损失曲线：监控收敛情况
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 绘制准确率曲线：监控性能提升
    axes[1].plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表：用于实验报告
    chart_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"训练历史图表已保存到: {chart_path}")

    # 显示图表（如果可能）
    try:
        plt.show()
    except:
        pass

    plt.close()

def plot_confusion_matrix(all_preds, all_labels, save_dir):
    """
    绘制混淆矩阵
    详细分析模型的分类性能

    参数:
        all_preds: 预测标签列表
        all_labels: 真实标签列表
        save_dir: 保存目录

    混淆矩阵说明：对角线上的值越大越好，非对角线表示分类错误
    """
    os.makedirs(save_dir, exist_ok=True)

    # 计算混淆矩阵：显示每个类别的分类情况
    cm = confusion_matrix(all_labels, all_preds)

    # 绘制混淆矩阵：使用热力图可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 保存图表
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存到: {cm_path}")

    # 显示图表（如果可能）
    try:
        plt.show()
    except:
        pass

    plt.close()

def main():
    """主训练函数：完整的训练流程"""
    print("=" * 60)
    print("实验1: 基于TextCNN的Amazon数据集情感分析")
    print("=" * 60)

    # ========== 1. 设置设备（GPU/CPU） ==========
    print("\n1. 设置训练设备...")
    device = setup_device()

    # ========== 2. 设置超参数 ==========
    print("\n2. 设置超参数...")
    config = {
        # 数据参数
        'batch_size': 32,  # 每批数据量，影响内存使用和训练速度
        'max_length': 256,  # 最大序列长度（截断或填充）
        'sample_size': 1000,  # 每个类别采样数量（平衡数据集）

        # 模型参数
        'embedding_dim': 300,  # 词向量维度
        'filter_sizes': [3, 4, 5],  # 卷积核大小（提取3-5gram特征）
        'num_filters': 100,  # 每种卷积核数量
        'dropout_rate': 0.5,  # dropout比率（防止过拟合）

        # 训练参数
        'num_epochs': 10,  # 训练轮数
        'learning_rate': 0.001,  # 学习率（控制参数更新步长）
        'weight_decay': 1e-4,  # 权重衰减（L2正则化，防止过拟合）

        # 保存路径
        'save_dir': r"D:\PycharmProjects\exp02-sentiment-classificationn\exp01-textcnn\saved_models",
        'log_dir': r"D:\PycharmProjects\exp02-sentiment-classificationn\exp01-textcnn\logs",
    }

    print("超参数配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # ========== 3. 加载数据 ==========
    print("\n3. 加载数据集...")

    # 首先尝试使用修复后的文件（如果存在）
    fixed_train_path = r"D:\PycharmProjects\exp02-sentiment-classificationn\datasets\fixed\train.csv"

    if os.path.exists(fixed_train_path):
        print("检测到修复后的CSV文件，使用修复版本...")
        # 临时修改create_data_loaders函数中的路径
        import data_loader
        original_create = data_loader.create_data_loaders

        def patched_create_data_loaders(batch_size=32, max_length=256, sample_size=1000):
            """修补版本的数据加载器，使用修复后的CSV文件"""
            base_path = r"D:\PycharmProjects\exp02-sentiment-classificationn\datasets\fixed"
            train_path = os.path.join(base_path, "train.csv")
            dev_path = os.path.join(base_path, "dev.csv")
            test_path = os.path.join(base_path, "test.csv")

            # 直接调用AmazonReviewDataset创建数据集
            from data_loader import AmazonReviewDataset
            import torch
            from torch.utils.data import DataLoader

            print("创建训练集...")
            train_dataset = AmazonReviewDataset(
                train_path,
                vocab=None,
                max_length=max_length,
                sample_size=sample_size
            )

            vocab = train_dataset.get_vocab()

            print("创建验证集...")
            dev_dataset = AmazonReviewDataset(
                dev_path,
                vocab=vocab,
                max_length=max_length,
                sample_size=sample_size
            )

            print("创建测试集...")
            test_dataset = AmazonReviewDataset(
                test_path,
                vocab=vocab,
                max_length=max_length,
                sample_size=sample_size
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            return train_loader, dev_loader, test_loader, vocab

        # 使用修复的版本
        train_loader, dev_loader, test_loader, vocab = patched_create_data_loaders(
            batch_size=config['batch_size'],
            max_length=config['max_length'],
            sample_size=config['sample_size']
        )
    else:
        # 使用原始版本
        train_loader, dev_loader, test_loader, vocab = create_data_loaders(
            batch_size=config['batch_size'],
            max_length=config['max_length'],
            sample_size=config['sample_size']
        )

    # ========== 检查数据是否成功加载 ==========
    if train_loader is None or dev_loader is None or test_loader is None or vocab is None:
        print("\n数据加载失败！请检查以下问题:")
        print("1. 确保文件夹路径正确")
        print("2. 确保CSV文件存在: train.csv, dev.csv, test.csv")
        print("3. CSV文件格式应为: 每行三列: polarity,title,text (无标题行)")
        print("4. 如果dev.csv和test.csv的格式不同，请确保它们也能正确解析")
        print("\n程序退出")
        return

    # ========== 4. 创建模型 ==========
    print("\n4. 创建TextCNN模型...")
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")

    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        num_classes=2,  # 二分类：正面/负面
        filter_sizes=config['filter_sizes'],
        num_filters=config['num_filters'],
        dropout_rate=config['dropout_rate'],
        pretrained_embeddings=None,  # 不使用预训练词向量
        freeze_embeddings=False  # 词向量参与训练
    )

    # 将模型移动到设备（GPU/CPU）：模型和数据需要在同一设备
    model = model.to(device)
    print(f"模型已移动到: {device}")

    # 打印模型参数数量：了解模型复杂度
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # ========== 5. 设置损失函数和优化器 ==========
    print("\n5. 设置损失函数和优化器...")
    # 交叉熵损失函数：适用于分类问题
    criterion = nn.CrossEntropyLoss()

    # Adam优化器：自适应学习率，结合了AdaGrad和RMSProp的优点
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']  # L2正则化，防止过拟合
    )

    # 学习率调度器 - 简化版本（兼容所有PyTorch版本）
    # 当验证损失不再下降时，自动降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控损失的最小化
        factor=0.5,  # 学习率衰减因子
        patience=2   # 耐心值：连续2个epoch损失未改善则降低学习率
    )
    print("学习率调度器: ReduceLROnPlateau")

    print(f"损失函数: {criterion.__class__.__name__}")
    print(f"优化器: {optimizer.__class__.__name__}")
    print(f"初始学习率: {config['learning_rate']}")

    # ========== 6. 训练模型 ==========
    print("\n" + "=" * 60)
    print("开始训练模型...")
    print("=" * 60)

    # 记录训练历史：用于监控和可视化
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0.0  # 跟踪最佳验证准确率
    best_model_path = None

    for epoch in range(config['num_epochs']):
        print(f"\n{'='*40} Epoch {epoch+1}/{config['num_epochs']} {'='*40}")

        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer,
            criterion, epoch, config['num_epochs']
        )

        # 验证模型：在每个epoch后评估性能
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = validate(
            model, device, dev_loader, criterion, mode="验证"
        )

        # 记录训练历史
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 更新学习率：根据验证损失调整
        scheduler.step(val_loss)

        # 保存最佳模型：只保存性能最好的模型
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_path = save_model(
                model, vocab, epoch, optimizer,
                val_loss, val_acc, config['save_dir']
            )
            print(f"  新的最佳模型已保存 (验证准确率: {val_acc:.4f})")

        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  当前学习率: {current_lr:.6f}")

    # ========== 7. 测试最佳模型 ==========
    print("\n" + "=" * 60)
    print("测试最佳模型...")
    print("=" * 60)

    if best_model_path is not None:
        # 加载最佳模型：使用保存的checkpoint
        print(f"加载最佳模型: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)

        # 创建新模型并加载状态
        best_model = TextCNN(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['model_config']['embedding_dim'],
            num_classes=checkpoint['model_config']['num_classes'],
            filter_sizes=checkpoint['model_config']['filter_sizes'],
            num_filters=checkpoint['model_config']['num_filters'],
            dropout_rate=checkpoint['model_config']['dropout_rate']
        )
        best_model.load_state_dict(checkpoint['model_state_dict'])
        best_model = best_model.to(device)

        # 测试模型：在测试集上评估最终性能
        test_loss, test_acc, test_precision, test_recall, test_f1, test_preds, test_labels = validate(
            best_model, device, test_loader, criterion, mode="测试"
        )

        # 打印测试结果
        print("\n测试结果:")
        print(f"  准确率: {test_acc:.4f}")
        print(f"  精确率: {test_precision:.4f}")
        print(f"  召回率: {test_recall:.4f}")
        print(f"  F1分数: {test_f1:.4f}")

        # 绘制混淆矩阵：详细分析分类性能
        plot_confusion_matrix(test_preds, test_labels, config['log_dir'])
    else:
        print("警告: 未找到最佳模型，使用当前模型进行测试")
        test_loss, test_acc, test_precision, test_recall, test_f1, test_preds, test_labels = validate(
            model, device, test_loader, criterion, mode="测试"
        )

    # ========== 8. 绘制训练历史 ==========
    print("\n" + "=" * 60)
    print("绘制训练历史图表...")
    print("=" * 60)

    plot_training_history(
        train_losses, val_losses,
        train_accuracies, val_accuracies,
        config['log_dir']
    )

    # ========== 9. 保存训练配置和结果 ==========
    print("\n" + "=" * 60)
    print("保存训练配置和结果...")
    print("=" * 60)

    # 保存训练配置：便于复现实验
    config_path = os.path.join(config['log_dir'], 'training_config.txt')
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write("训练配置:\n")
        f.write("="*40 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

        f.write("\n训练结果:\n")
        f.write("="*40 + "\n")
        f.write(f"最佳验证准确率: {best_val_accuracy:.4f}\n")
        f.write(f"测试准确率: {test_acc:.4f}\n")
        f.write(f"测试精确率: {test_precision:.4f}\n")
        f.write(f"测试召回率: {test_recall:.4f}\n")
        f.write(f"测试F1分数: {test_f1:.4f}\n")

        f.write(f"\n训练设备: {device}\n")
        f.write(f"模型参数总数: {total_params:,}\n")
        f.write(f"可训练参数: {trainable_params:,}\n")
        f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"训练配置和结果已保存到: {config_path}")

    print("\n" + "=" * 60)
    print("实验1完成!")
    print("=" * 60)
    print(f"最佳模型: {best_model_path}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"日志文件保存在: {config['log_dir']}")

if __name__ == "__main__":
    # 设置随机种子以确保可重复性：相同的随机种子产生相同的结果
    torch.manual_seed(42)
    np.random.seed(42)

    # 开始训练
    main()