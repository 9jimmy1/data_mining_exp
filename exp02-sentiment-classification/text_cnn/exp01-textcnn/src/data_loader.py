"""
数据加载和预处理模块
负责从CSV文件加载数据，并进行文本预处理
关键功能：读取Amazon评论数据、文本清洗、构建词汇表、创建数据加载器
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import os
import sys
import csv

# ========== 手动设置nltk数据路径 ==========
def setup_nltk():
    """
    设置nltk数据路径，避免从网络下载
    解决国内网络无法访问nltk官方服务器的问题
    """
    # 自定义nltk数据路径（需要预先下载好nltk数据包）
    custom_nltk_path = r"C:\Users\chris\nltk_data"

    if not os.path.exists(custom_nltk_path):
        print(f"错误: nltk数据路径不存在: {custom_nltk_path}")
        print("\n请按以下步骤手动下载nltk数据:")
        print("1. 创建文件夹: C:\\Users\\chris\\nltk_data")
        print("2. 下载punkt分词器:")
        print("   下载地址: https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip")
        print("3. 解压punkt.zip到: C:\\Users\\chris\\nltk_data\\tokenizers\\")
        print("   最终路径应该是: C:\\Users\\chris\\nltk_data\\tokenizers\\punkt\\")
        return False

    # 添加到nltk数据路径：让nltk知道去哪里找数据文件
    nltk.data.path.append(custom_nltk_path)
    print(f"✓ 已设置nltk数据路径: {custom_nltk_path}")

    # 检查punkt分词器：确保分词器可用
    try:
        nltk.data.find('tokenizers/punkt')
        print("✓ 找到punkt分词器")
        return True
    except LookupError:
        print("✗ 未找到punkt分词器")
        print(f"请确保punkt分词器在: {custom_nltk_path}\\tokenizers\\punkt\\")
        return False

# 初始化nltk：必须在所有nltk调用之前执行
if not setup_nltk():
    print("nltk初始化失败，退出程序")
    sys.exit(1)
# =========================================

def load_csv_file(file_path):
    """
    加载CSV文件，处理混合格式：
    1. train.csv: 每列都有双引号
    2. dev.csv/test.csv: 只有text列有时有双引号

    参数:
        file_path: CSV文件路径

    返回:
        df: 处理后的DataFrame
    """
    print(f"正在加载: {file_path}")

    # 存储解析后的数据
    data = []

    # 尝试不同的编码：CSV文件可能有多种编码格式
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                reader = csv.reader(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)

                row_count = 0
                for row in reader:
                    row_count += 1
                    # 跳过空行
                    if not row:
                        continue

                    # 处理train.csv格式（每列都有引号）
                    if len(row) == 1 and ',' in row[0]:
                        # 如果整个行被当作一个字段，尝试手动分割
                        parts = row[0].split(',', 2)  # 只分割前两个逗号
                        if len(parts) >= 3:
                            polarity = parts[0].strip('" ')
                            title = parts[1].strip('" ')
                            text = ','.join(parts[2:]).strip('" ')
                            data.append([polarity, title, text])
                        else:
                            print(f"警告: 行 {row_count} 格式异常: {row}")
                    elif len(row) >= 3:
                        # 正常情况，有3个或更多字段
                        polarity = str(row[0]).strip('" ')
                        title = str(row[1]).strip('" ')
                        # 合并剩余部分作为text
                        text = ','.join(row[2:]).strip('" ')
                        data.append([polarity, title, text])
                    else:
                        print(f"警告: 行 {row_count} 字段数不足: {len(row)}")

                print(f"使用 {encoding} 编码成功读取 {len(data)} 行")
                break

        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"使用 {encoding} 编码读取失败: {e}")
            continue

    if not data:
        raise ValueError(f"无法读取文件 {file_path}，尝试了多种编码")

    # 转换为DataFrame：便于数据处理
    df = pd.DataFrame(data, columns=['polarity', 'title', 'text'])

    # 确保polarity是整数类型
    df['polarity'] = pd.to_numeric(df['polarity'], errors='coerce')

    # 删除polarity为NaN的行：无效数据
    df = df.dropna(subset=['polarity'])
    df['polarity'] = df['polarity'].astype(int)

    print(f"处理后有效数据: {len(df)} 行")
    return df

class AmazonReviewDataset(Dataset):
    """Amazon评论数据集类，继承自PyTorch的Dataset"""

    def __init__(self, csv_file, vocab=None, max_length=256, sample_size=1000):
        """
        初始化数据集

        参数:
            csv_file: CSV文件路径
            vocab: 词汇表 (如果为None，则从数据构建)
            max_length: 最大序列长度（截断或填充）
            sample_size: 每个类别采样数量（平衡正负样本）
        """
        self.max_length = max_length

        # 1. 加载数据 - 使用自定义的CSV解析器
        try:
            df = load_csv_file(csv_file)
        except Exception as e:
            print(f"加载CSV文件失败: {e}")
            raise

        # 2. 数据采样：正负各取sample_size条（平衡数据集）
        # 注意：polarity=1表示负面，2表示正面（Amazon数据集标准）
        print(f"正样本数量(polarity=2): {len(df[df['polarity'] == 2])}")
        print(f"负样本数量(polarity=1): {len(df[df['polarity'] == 1])}")

        negative_reviews = df[df['polarity'] == 1]
        positive_reviews = df[df['polarity'] == 2]

        # 如果数据量不足，使用全部数据
        if len(negative_reviews) >= sample_size and len(positive_reviews) >= sample_size:
            negative_reviews = negative_reviews.sample(sample_size, random_state=42)
            positive_reviews = positive_reviews.sample(sample_size, random_state=42)
            print(f"采样: 正样本{sample_size}条, 负样本{sample_size}条")
        else:
            print(f"警告：数据量不足，使用全部数据（负样本{len(negative_reviews)}条，正样本{len(positive_reviews)}条）")

        # 合并正负样本
        self.data = pd.concat([negative_reviews, positive_reviews])
        print(f"合并后总数据量: {len(self.data)}条")

        # 3. 文本预处理：清洗和标准化文本
        print("正在进行文本预处理...")

        # 合并标题和内容作为输入文本：标题包含重要情感信息
        self.data['full_text'] = self.data['title'].fillna('').astype(str) + ' ' + self.data['text'].fillna('').astype(str)

        # 清洗文本：去除噪音，统一格式
        self.data['cleaned_text'] = self.data['full_text'].apply(self.clean_text)

        # 4. 标签处理：将1,2转换为0,1 (0:负面, 1:正面)
        # 适应PyTorch的标签格式（从0开始）
        self.data['label'] = self.data['polarity'].apply(lambda x: 0 if x == 1 else 1)

        # 5. 构建词汇表或使用现有词汇表
        # 训练集构建词汇表，验证集和测试集复用训练集的词汇表
        if vocab is None:
            print("正在构建词汇表...")
            self.vocab = self.build_vocab(self.data['cleaned_text'].tolist())
        else:
            self.vocab = vocab

        # 6. 文本转换为索引序列：将单词映射为数字
        print("正在将文本转换为索引序列...")
        self.sequences = []
        for i, text in enumerate(self.data['cleaned_text']):
            if i % 500 == 0 and i > 0:  # 每500条显示一次进度
                print(f"  处理进度: {i}/{len(self.data)}")
            self.sequences.append(self.text_to_sequence(text))

        print(f"数据集加载完成，共{len(self)}条数据")
        print(f"词汇表大小: {len(self.vocab)}")
        print(f"标签分布: 负面={sum(self.data['label']==0)}, 正面={sum(self.data['label']==1)}")

    def clean_text(self, text):
        """清洗文本：去除特殊字符，转换为小写，标准化格式"""
        if not isinstance(text, str):
            return ""

        # 转换为小写：统一大小写
        text = text.lower()
        # 去除HTML标签：可能存在的HTML格式
        text = re.sub(r'<[^>]+>', '', text)
        # 保留字母、数字和基本标点：去除特殊符号
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)
        # 去除多余空格：统一空格数量
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def build_vocab(self, texts, min_freq=3):
        """构建词汇表，统计词频，过滤低频词"""
        print("构建词汇表中...")
        # 统计词频：Counter可以高效统计
        word_counter = Counter()
        total_texts = len(texts)

        for i, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                try:
                    # 使用nltk分词：更准确的分词
                    tokens = word_tokenize(text)
                    word_counter.update(tokens)
                except Exception as e:
                    # 如果分词失败，使用简单空格分词：降级处理
                    tokens = text.split()
                    word_counter.update(tokens)

            if i % 500 == 0 and i > 0:  # 每500条显示一次进度
                print(f"  词汇表构建进度: {i}/{total_texts}")

        print(f"  发现唯一词汇: {len(word_counter)}个")

        # 创建词汇表：添加特殊token
        vocab = {
            '<PAD>': 0,  # 填充token：用于序列对齐
            '<UNK>': 1,  # 未知token：不在词汇表中的词
        }

        # 添加满足最小频率的词：过滤低频词（通常是噪声）
        idx = 2
        freq_words = 0
        for word, count in word_counter.items():
            if count >= min_freq:
                vocab[word] = idx
                idx += 1
                freq_words += 1

        print(f"  词汇表大小（包括特殊token）: {len(vocab)}")
        print(f"  满足最小频率({min_freq})的词汇: {freq_words}个")

        return vocab

    def text_to_sequence(self, text):
        """将文本转换为索引序列，进行截断或填充"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            # 返回填充序列：空文本处理
            return [0] * self.max_length

        try:
            # 使用nltk分词：支持标点符号处理
            tokens = word_tokenize(text)
        except:
            # 分词失败，使用简单空格分词：兼容性处理
            tokens = text.split()

        # 截断或填充到最大长度：统一序列长度
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]  # 截断：保留前max_length个词
        else:
            tokens = tokens + ['<PAD>'] * (self.max_length - len(tokens))  # 填充

        # 转换为索引：将单词映射为数字
        sequence = []
        for token in tokens:
            # vocab.get(token, vocab['<UNK>'])：如果词不在词汇表中，使用UNK索引
            sequence.append(self.vocab.get(token, self.vocab['<UNK>']))

        return sequence

    def __len__(self):
        """返回数据集大小（必须实现）"""
        return len(self.data)

    def __getitem__(self, idx):
        """获取单个数据样本（必须实现）"""
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.long)
        return sequence, label

    def get_vocab_size(self):
        """获取词汇表大小"""
        return len(self.vocab)

    def get_vocab(self):
        """获取词汇表"""
        return self.vocab

def create_data_loaders(batch_size=32, max_length=256, sample_size=1000):
    """
    创建训练、验证和测试数据加载器
    负责数据集的划分和加载器创建

    参数:
        batch_size: 批大小（影响内存使用和训练速度）
        max_length: 最大序列长度
        sample_size: 每个类别采样数量

    返回:
        train_loader, dev_loader, test_loader, vocab
    """
    # 路径设置：数据集存放位置
    base_path = r"D:\PycharmProjects\exp02-sentiment-classificationn\datasets"

    # 检查文件夹是否存在
    if not os.path.exists(base_path):
        print(f"警告: 路径不存在: {base_path}")
        # 尝试另一个可能的文件夹名
        alt_path = r"D:\PycharmProjects\exp02-sentiment-classificationn\dataset"
        if os.path.exists(alt_path):
            print(f"使用替代路径: {alt_path}")
            base_path = alt_path
        else:
            print(f"错误: 两个路径都不存在")
            print(f"请确保数据集文件夹存在: {base_path} 或 {alt_path}")
            return None, None, None, None

    train_path = os.path.join(base_path, "train.csv")
    dev_path = os.path.join(base_path, "dev.csv")
    test_path = os.path.join(base_path, "test.csv")

    # 检查文件是否存在
    for path in [train_path, dev_path, test_path]:
        if not os.path.exists(path):
            print(f"错误: 文件不存在: {path}")
            print(f"请确保文件已正确放置在 {base_path} 目录下")
            # 显示目录内容：帮助用户检查
            print(f"\n{base_path} 目录内容:")
            for f in os.listdir(base_path):
                print(f"  - {f}")
            return None, None, None, None

    print(f"使用数据集路径: {base_path}")

    # 创建训练集（构建词汇表）
    print("\n" + "="*50)
    print("创建训练集...")
    try:
        train_dataset = AmazonReviewDataset(
            train_path,
            vocab=None,  # 训练集需要构建词汇表
            max_length=max_length,
            sample_size=sample_size
        )
    except Exception as e:
        print(f"创建训练集失败: {e}")
        return None, None, None, None

    # 获取词汇表：用于验证集和测试集
    vocab = train_dataset.get_vocab()

    # 创建验证集（使用训练集的词汇表）：保持词汇一致性
    print("\n" + "="*50)
    print("创建验证集...")
    try:
        dev_dataset = AmazonReviewDataset(
            dev_path,
            vocab=vocab,  # 使用训练集的词汇表
            max_length=max_length,
            sample_size=sample_size
        )
    except Exception as e:
        print(f"创建验证集失败: {e}")
        return None, None, None, None

    # 创建测试集（使用训练集的词汇表）
    print("\n" + "="*50)
    print("创建测试集...")
    try:
        test_dataset = AmazonReviewDataset(
            test_path,
            vocab=vocab,  # 使用训练集的词汇表
            max_length=max_length,
            sample_size=sample_size
        )
    except Exception as e:
        print(f"创建测试集失败: {e}")
        return None, None, None, None

    # 创建数据加载器：负责批量加载数据
    print("\n" + "="*50)
    print("创建数据加载器...")

    # 训练集：需要打乱顺序，避免批次间的相关性
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时打乱数据
        num_workers=0  # 在Windows上设置为0避免多进程问题
    )

    # 验证集：不需要打乱，保持顺序以观察模型性能
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证时不打乱
        num_workers=0
    )

    # 测试集：不需要打乱，保持原始顺序
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试时不打乱
        num_workers=0
    )

    print(f"训练集批次: {len(train_loader)}")
    print(f"验证集批次: {len(dev_loader)}")
    print(f"测试集批次: {len(test_loader)}")

    return train_loader, dev_loader, test_loader, vocab

if __name__ == "__main__":
    # 测试数据加载器：验证数据加载是否正常
    train_loader, dev_loader, test_loader, vocab = create_data_loaders(batch_size=4)

    if train_loader is not None:
        # 检查一个批次的数据：验证数据形状和类型
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            print(f"批次 {batch_idx + 1}:")
            print(f"  序列形状: {sequences.shape}")  # (batch_size, max_length)
            print(f"  标签形状: {labels.shape}")    # (batch_size,)
            print(f"  标签: {labels.tolist()}")

            if batch_idx >= 0:  # 只检查第一个批次
                break
    else:
        print("数据加载失败")