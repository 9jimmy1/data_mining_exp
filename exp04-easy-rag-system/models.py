import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@st.cache_resource
def load_embedding_model(model_name):
    """Loads the sentence transformer model."""
    st.write(f"Loading embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        st.success("Embedding model loaded.")
        return model
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

@st.cache_resource
def load_generation_model(model_name):
    """Loads the Hugging Face generative model and tokenizer."""
    st.write(f"Loading generation model: {model_name}...")
    try:
        # 关键修改：添加 cache_dir 参数，强制从本地缓存加载
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir='D:\.cache\huggingface'  # 指定您的缓存路径
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir='D:\.cache\huggingface',  # 指定您的缓存路径
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        st.success("Generation model and tokenizer loaded.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load generation model: {e}")
        # 可以添加更详细的错误信息，帮助排查缓存问题
        st.error(f"请检查缓存目录 D:\.cache\huggingface 是否存在且包含模型文件。")
        return None, None