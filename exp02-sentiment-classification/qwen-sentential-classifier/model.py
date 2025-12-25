import torch
import torch.nn as nn
from transformers import AutoModel

class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()

        # ★ 强制使用 BF16 + 自动分配设备，避免爆显存
        self.encoder = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,   # 更稳定，不会像 FP16 那样报溢出
            device_map="auto"             # 自动分配显存/CPU，防止卡住
        )

        hidden = self.encoder.config.hidden_size

        # 分类层仍然用 FP32（更稳定）
        self.classifier = nn.Linear(hidden, num_classes).to(torch.float32)

    def forward(self, ids, mask):
        # 编码部分为 BF16
        out = self.encoder(input_ids=ids, attention_mask=mask)

        # [CLS] 向量
        cls = out.last_hidden_state[:, 0, :]

        # 分类层为 FP32，需要转换
        cls = cls.to(torch.float32)

        return self.classifier(cls)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))
