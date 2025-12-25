"""
工具函数模块
包含各种辅助函数：GPU信息、参数统计、结果保存、可视化等
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import json
import os

def print_gpu_info():
    """
    打印GPU信息
    帮助了解CUDA环境、GPU型号、显存使用情况
    """
    if torch.cuda.is_available():
        print("CUDA可用")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")

        device_count = torch.cuda.device_count()
        print(f"找到 {device_count} 个GPU设备:")

        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    内存总量: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"    当前内存使用: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"    最大内存使用: {torch.cuda.max_memory_allocated(i) / 1e9:.2f} GB")
    else:
        print("CUDA不可用，将使用CPU")

def count_parameters(model):
    """
    计算模型参数数量
    了解模型复杂度，帮助调整模型大小

    返回:
        total_params: 总参数数量
        trainable_params: 可训练参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"不可训练参数: {total_params - trainable_params:,}")

    return total_params, trainable_params

def save_classification_report(y_true, y_pred, target_names, save_path):
    """
    保存分类报告
    详细评估模型的分类性能

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        target_names: 类别名称
        save_path: 保存路径

    分类报告包含：精确率、召回率、F1分数、支持度等
    """
    # 生成分类报告（字典格式）
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

    # 保存为文本文件：便于人类阅读
    with open(save_path.replace('.json', '.txt'), 'w', encoding='utf-8') as f:
        f.write(classification_report(y_true, y_pred, target_names=target_names))

    # 保存为JSON文件：便于程序解析
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"分类报告已保存到: {save_path}")

def visualize_embeddings(model, vocab, word_list, save_path=None):
    """
    可视化词向量
    使用PCA将高维词向量降维到2D进行可视化

    参数:
        model: 模型
        vocab: 词汇表
        word_list: 要可视化的词列表
        save_path: 保存路径（可选）

    原理：PCA（主成分分析）降维，保留最大方差的方向
    """
    # 获取词向量
    embeddings = []
    labels = []

    for word in word_list:
        if word in vocab:
            word_idx = vocab[word]
            # 获取词向量：通过嵌入层获取
            word_tensor = torch.tensor([word_idx])
            embedding = model.embedding(word_tensor).detach().cpu().numpy()
            embeddings.append(embedding[0])
            labels.append(word)

    if len(embeddings) == 0:
        print("没有找到可用的词")
        return

    embeddings = np.array(embeddings)

    # 使用PCA降维到2D：将300维向量降到2维
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # 可视化：散点图展示词向量分布
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)

    # 添加标签：显示每个点对应的词
    for i, label in enumerate(labels):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                     fontsize=9, alpha=0.7)

    plt.title('词向量可视化 (PCA降维)')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"词向量可视化图已保存到: {save_path}")

    plt.show()

def check_memory_usage():
    """
    检查内存使用情况
    监控GPU显存使用，避免内存溢出
    """
    if torch.cuda.is_available():
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU内存缓存: {torch.cuda.memory_cached() / 1e9:.2f} GB")
        print(f"最大GPU内存使用: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    else:
        print("无法检查GPU内存使用情况（CUDA不可用）")

def set_seed(seed=42):
    """
    设置随机种子以确保可重复性
    相同的种子会产生相同的随机序列，保证实验结果可复现

    参数:
        seed: 随机种子

    影响：PyTorch、NumPy、CUDA的随机数生成器
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

    # 设置cuDNN：确保卷积运算的确定性
    torch.backends.cudnn.deterministic = True  # 保证每次卷积算法相同
    torch.backends.cudnn.benchmark = False  # 关闭自动寻找最优卷积算法

    print(f"随机种子已设置为: {seed}")

if __name__ == "__main__":
    # 测试工具函数
    print("GPU信息:")
    print_gpu_info()

    print("\n设置随机种子:")
    set_seed(42)