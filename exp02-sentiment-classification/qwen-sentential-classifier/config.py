# config.py
class Config:
    # 模型参数
    model_name = "https://hf-mirror.com/Qwen/Qwen2.5-0.5B"  # ← 改为 Qwen2.5-0.5B
    max_seq_length = 128
    num_classes = 2

    # 训练参数
    batch_size = 4      # Qwen 较大，建议从 4 开始（RTX 4060 8GB）
    learning_rate = 2e-6  # LLM 微调常用更小 lr
    num_epochs = 3

    # 路径
    train_path = "dataset/train_mini.csv"
    dev_path = "dataset/dev.csv"
    test_path = "dataset/test.csv"
    model_save_path = "saved_models/qwen_sentiment_model.pth"