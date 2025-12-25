"""
TextCNN模型定义
基于Kim (2015)的TextCNN架构
使用多个不同大小的卷积核捕捉文本的局部特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    """
    TextCNN模型
    使用多个不同大小的卷积核捕捉不同范围的上下文信息
    核心思想：用不同宽度的卷积核提取文本的n-gram特征，然后拼接进行分类
    """

    def __init__(self, vocab_size, embedding_dim=300, num_classes=2,
                 filter_sizes=[3, 4, 5], num_filters=100, dropout_rate=0.5,
                 pretrained_embeddings=None, freeze_embeddings=False):
        """
        初始化TextCNN模型

        参数:
            vocab_size: 词汇表大小
            embedding_dim: 词向量维度（通常为50/100/200/300）
            num_classes: 分类类别数（2: 正面/负面）
            filter_sizes: 卷积核大小列表，如[3,4,5]表示提取3-gram,4-gram,5-gram特征
            num_filters: 每种卷积核的数量（即每个n-gram特征图的通道数）
            dropout_rate: dropout比率，防止过拟合
            pretrained_embeddings: 预训练的词向量（如Word2Vec/GloVe）
            freeze_embeddings: 是否冻结词向量层的参数（微调时设为False）
        """
        super(TextCNN, self).__init__()

        # 保存模型参数供后续使用
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.using_pretrained = pretrained_embeddings is not None
        self.freeze_embeddings = freeze_embeddings

        # 1. 词嵌入层：将单词索引转换为密集向量表示
        # 预训练词向量可以提升模型性能，特别是当训练数据较少时
        if pretrained_embeddings is not None:
            # 使用预训练的词向量（从外部加载，如GloVe/Word2Vec）
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_embeddings,  # freeze=True时词向量不参与训练
                padding_idx=0  # 假设0是padding索引，用于序列对齐
            )
        else:
            # 随机初始化词向量（训练过程中会学习）
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=0  # 填充token的索引，其向量会被设为0且不更新
            )

        # 2. 卷积层：使用不同大小的卷积核提取局部特征
        # 每个卷积核在文本序列上滑动，提取特定n-gram的特征
        # Conv2d输入形状: (batch_size, channels, height, width)
        # 对于文本: channels=1, height=seq_len, width=embedding_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,  # 文本的通道数为1（类似灰度图像）
                out_channels=num_filters,  # 每个卷积核产生num_filters个特征图
                kernel_size=(fs, embedding_dim)  # 卷积核大小：fs×embedding_dim
            )
            for fs in filter_sizes  # 为每个filter_size创建一个卷积层
        ])

        # 3. Dropout层：随机丢弃部分神经元，防止过拟合
        # 训练时按dropout_rate概率丢弃，测试时不起作用
        self.dropout = nn.Dropout(dropout_rate)

        # 4. 全连接层：将卷积特征映射到分类结果
        # 所有卷积核的特征图拼接后的总维度
        total_filters = num_filters * len(filter_sizes)
        self.fc = nn.Linear(total_filters, num_classes)

        # 5. 初始化权重：合适的初始化可以加速收敛
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重，使用Xavier初始化策略"""
        # 如果使用随机初始化的词向量，则初始化
        if not self.using_pretrained:
            # Xavier均匀分布初始化：根据输入输出维度自动调整初始值范围
            nn.init.xavier_uniform_(self.embedding.weight)
            # 将填充token的权重设为0（不参与训练和影响）
            if self.embedding.padding_idx is not None:
                with torch.no_grad():
                    self.embedding.weight[self.embedding.padding_idx].fill_(0)

        # 初始化卷积层权重
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)  # 权重初始化
            nn.init.constant_(conv.bias, 0)      # 偏置初始化为0

        # 初始化全连接层权重
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        """
        前向传播过程
        输入: 文本序列的索引表示 -> 输出: 分类logits

        参数:
            x: 输入序列，形状为 (batch_size, seq_len)

        返回:
            logits: 分类logits，形状为 (batch_size, num_classes)
            可以加softmax得到概率分布
        """
        batch_size = x.size(0)

        # 1. 词嵌入：将单词索引转换为向量
        # 输入形状: (batch_size, seq_len)
        # 输出形状: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # 2. 添加通道维度以适应Conv2d
        # 文本可以看作1个通道的二维矩阵：高度=序列长度，宽度=词向量维度
        # 形状变换: (batch_size, seq_len, embedding_dim) -> (batch_size, 1, seq_len, embedding_dim)
        embedded = embedded.unsqueeze(1)

        # 3. 应用不同大小的卷积核提取特征
        conv_outputs = []
        for conv in self.convs:
            # 卷积操作：每个卷积核在文本序列上滑动
            # 输入: (batch_size, 1, seq_len, embedding_dim)
            # 输出: (batch_size, num_filters, new_seq_len, 1)
            # new_seq_len = seq_len - filter_size + 1
            conv_out = conv(embedded)

            # 应用ReLU激活函数：引入非线性，避免梯度消失
            conv_out = F.relu(conv_out)

            # 最大池化：提取每个特征图的最重要特征（最大响应值）
            # 沿着序列维度进行全局最大池化
            # 形状: (batch_size, num_filters, new_seq_len, 1) -> (batch_size, num_filters, 1, 1)
            pooled = F.max_pool2d(conv_out, (conv_out.size(2), 1))

            # 去除多余的维度（序列维度和最后一个维度）
            # 形状: (batch_size, num_filters, 1, 1) -> (batch_size, num_filters)
            pooled = pooled.squeeze(3).squeeze(2)

            conv_outputs.append(pooled)

        # 4. 拼接所有卷积层的输出
        # 不同尺寸卷积核提取的特征具有互补性
        # 形状: (batch_size, num_filters * len(filter_sizes))
        cat = torch.cat(conv_outputs, dim=1)

        # 5. Dropout：训练时随机丢弃，防止过拟合
        cat = self.dropout(cat)

        # 6. 全连接层：将特征映射到分类空间
        # 形状: (batch_size, total_filters) -> (batch_size, num_classes)
        logits = self.fc(cat)

        return logits

    def get_embeddings(self, tokens):
        """获取词向量，可用于可视化或特征提取"""
        return self.embedding(tokens)

if __name__ == "__main__":
    # 测试模型：验证模型结构是否正确，输入输出维度是否匹配
    vocab_size = 10000
    model = TextCNN(vocab_size=vocab_size)

    # 创建模拟输入：batch_size=4, seq_len=100
    batch_size = 4
    seq_len = 100
    x = torch.randint(1, vocab_size, (batch_size, seq_len))

    print("模型结构:")
    print(model)
    print(f"\n输入形状: {x.shape}")

    # 前向传播：测试模型是否能正常运行
    output = model(x)
    print(f"输出形状: {output.shape}")
    print(f"输出值:\n{output}")