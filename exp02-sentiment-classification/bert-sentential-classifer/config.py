class Config:
    """
    模型配置类，包含所有可配置参数
    """
    # 模型参数
    model_name = "bert-base-uncased"  # 预训练模型名称
    max_seq_length = 64  # 最大序列长度，超过此长度的文本将被截断
    num_classes = 2  # 分类类别数量，二分类为2
    
    # 训练参数
    batch_size = 1  # 批次大小，可根据GPU内存调整
    learning_rate = 2e-5  # 学习率
    num_epochs = 3  # 训练轮数
    
    # 路径配置
    #train_path = "dataset/train.csv"   
    train_path = "dataset/train_mini.csv"  # 使用小型训练集加速实验
    dev_path = "dataset/dev.csv"  # 验证集路径
    test_path = "dataset/test.csv"  # 测试集路径
    model_save_path = "saved_models/sentiment_model.pth"  # 模型保存路径