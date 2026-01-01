# app.py - 适配 ChromaDB 版本
import streamlit as st
import time
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = 'D:\.cache\huggingface'
os.environ['TRANSFORMERS_CACHE'] = 'D:\.cache\huggingface'

# 导入配置和模块
from config import (
    DATA_FILE, EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, TOP_K,
    MAX_ARTICLES_TO_INDEX, COLLECTION_NAME, id_to_doc_map
)
from data_utils import load_data
from models import load_embedding_model, load_generation_model
# 导入 ChromaDB 工具函数
from chroma_utils import get_chroma_client, setup_chroma_collection, search_similar_documents
from rag_core import generate_answer

# --- Streamlit UI 设置 ---
st.set_page_config(layout="wide")
st.title("🏥 医疗 RAG 问答系统 (ChromaDB)")
st.markdown(f"**嵌入模型**: `{EMBEDDING_MODEL_NAME}` | **生成模型**: `{GENERATION_MODEL_NAME}`")

# --- 初始化与缓存 ---
# 获取 ChromaDB 客户端
chroma_client = get_chroma_client()

if chroma_client:
    # 加载模型
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    generation_model, tokenizer = load_generation_model(GENERATION_MODEL_NAME)
    
    models_loaded = embedding_model and generation_model and tokenizer
    
    if models_loaded:
        # 设置集合并索引数据（如果需要）
        with st.spinner("正在准备向量数据库..."):
            collection_ready = setup_chroma_collection(chroma_client, embedding_model)
        
        if collection_ready:
            st.divider()
            
            # --- RAG 交互界面 ---
            st.subheader("💬 医疗问答")
            query = st.text_input(
                "请输入您想了解的医疗问题：",
                placeholder="例如：感冒有什么症状？如何治疗？",
                key="query_input"
            )
            
            col1, col2, col3 = st.columns([1, 1, 6])
            with col1:
                search_btn = st.button("🔍 搜索答案", type="primary", use_container_width=True)
            with col2:
                clear_btn = st.button("🔄 清除", use_container_width=True)
            
            if clear_btn:
                st.rerun()
            
            if search_btn and query:
                start_time = time.time()
                
                # 1. 检索相关文档
                with st.spinner("正在从知识库中检索相关信息..."):
                    retrieved_ids, distances, retrieved_docs = search_similar_documents(
                        chroma_client, query, embedding_model
                    )
                
                if not retrieved_docs:
                    st.warning("⚠️ 未找到相关医学资料。请尝试其他提问方式。")
                else:
                    # 2. 显示检索结果
                    st.subheader("📄 检索到的参考内容")
                    for i, doc in enumerate(retrieved_docs):
                        with st.expander(f"📖 片段 {i+1}: {doc['title'][:50]}... (相关度: {1 - distances[i]:.2f})"):
                            st.caption(f"来源: {doc.get('source_file', '未知')} | 块索引: {doc.get('chunk_index', 'N/A')}")
                            st.markdown(f"**内容摘要:**")
                            st.info(doc['abstract'][:300] + ("..." if len(doc['abstract']) > 300 else ""))
                    
                    st.divider()
                    
                    # 3. 生成答案
                    st.subheader("🤖 AI 生成的回答")
                    with st.spinner("正在综合检索内容生成回答..."):
                        answer = generate_answer(query, retrieved_docs, generation_model, tokenizer)
                    
                    # 显示答案
                    st.success(answer)
                    
                    # 显示性能信息
                    end_time = time.time()
                    st.caption(f"⏱️ 本次查询耗时: {end_time - start_time:.2f}秒 | 检索文档数: {len(retrieved_docs)}")
            elif search_btn and not query:
                st.warning("请输入问题后再搜索。")
            
            # --- 系统信息侧边栏 ---
            st.sidebar.header("⚙️ 系统配置")
            st.sidebar.markdown(f"**向量数据库:** ChromaDB")
            st.sidebar.markdown(f"**数据文件:** `{DATA_FILE}`")
            st.sidebar.markdown(f"**集合名称:** `{COLLECTION_NAME}`")
            st.sidebar.markdown(f"**嵌入模型:** `{EMBEDDING_MODEL_NAME}`")
            st.sidebar.markdown(f"**生成模型:** `{GENERATION_MODEL_NAME}`")
            st.sidebar.markdown(f"**最大索引数:** `{MAX_ARTICLES_TO_INDEX}`")
            st.sidebar.markdown(f"**检索数量:** `{TOP_K}`")
            
            # 数据统计
            st.sidebar.divider()
            st.sidebar.subheader("📊 数据统计")
            if id_to_doc_map:
                st.sidebar.markdown(f"已加载文档块: **{len(id_to_doc_map)}**")
                # 显示前几个文档标题
                st.sidebar.caption("已索引的疾病主题:")
                for i, (doc_id, doc) in enumerate(list(id_to_doc_map.items())[:5]):
                    st.sidebar.markdown(f"- {doc['title'][:20]}...")
            
        else:
            st.error("向量数据库初始化失败，请检查数据文件。")
    else:
        st.error("模型加载失败，请检查配置和网络连接。")
else:
    st.error("ChromaDB 客户端初始化失败。")