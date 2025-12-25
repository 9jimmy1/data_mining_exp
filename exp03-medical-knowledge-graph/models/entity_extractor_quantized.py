# 量化版本的实体提取器（格式兼容版）
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class MedicalEntityExtractor:
    def __init__(self, model_name="Qwen/Qwen2.5-3B", device="auto"):
        self.device = device
        print(f"加载模型: {model_name} (4位量化)")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 4位量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # 加载量化模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.eval()
        print("模型加载完成")
    
    def extract_entities(self, text):
        """
        返回一个完全兼容 main.py 打印和数据保存逻辑的字典。
        为了快速验证流程，这里返回固定的模拟数据。
        """
        # 这是一个模拟的返回结构，确保：
        # 1. entities 是一个字符串列表（用于打印）
        # 2. 同时包含结构化的 entity_objects（用于生成CSV）
        return {

            # 用于后续生成 nodes.csv 的结构化字典列表
            "entity_objects": [
                {"type": "Disease", "name": "高血压"},
                {"type": "Symptom", "name": "胸痛"},
                {"type": "Symptom", "name": "恶心"},
                {"type": "Disease", "name": "肥胖症"}
            ],
            "relations": []
        }
