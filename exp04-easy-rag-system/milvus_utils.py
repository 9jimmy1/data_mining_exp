# milvus_utils.py - 适配标准Milvus服务 (Docker) 版本
import streamlit as st
# 关键改动：导入标准Milvus的连接和集合类，而不是MilvusClient
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import time
import os

# 导入配置变量，包括全局映射
from config import (
    COLLECTION_NAME, EMBEDDING_DIM, 
    MAX_ARTICLES_TO_INDEX, INDEX_METRIC_TYPE, INDEX_TYPE, INDEX_PARAMS,
    SEARCH_PARAMS, TOP_K, id_to_doc_map
)

# ==================== 新增配置：Docker Milvus 服务地址 ====================
# 请确保此地址和端口与您运行 docker run -p 19530:19530 的配置一致
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
# ==========================================================================

@st.cache_resource
def get_milvus_connection():
    """初始化并返回一个到 Docker Milvus 服务的连接。"""
    try:
        st.write(f"正在连接 Milvus 服务: {MILVUS_HOST}:{MILVUS_PORT} ...")
        
        # 关键改动：建立网络连接，而不是本地文件
        connections.connect(
            alias="default",           # 连接别名
            host=MILVUS_HOST,          # 服务地址
            port=MILVUS_PORT           # 服务端口
        )
        st.success("成功连接到 Milvus 服务！")
        return True  # 返回连接成功标志
        
    except Exception as e:
        st.error(f"连接 Milvus 服务失败: {e}")
        import traceback
        st.code(traceback.format_exc())  # 显示详细错误
        return None

def setup_milvus_collection():
    """确保指定的集合存在，并在标准 Milvus 中正确设置。"""
    try:
        collection_name = COLLECTION_NAME
        dim = EMBEDDING_DIM

        # 检查集合是否存在
        has_collection = utility.has_collection(collection_name)

        if not has_collection:
            st.write(f"集合 '{collection_name}' 不存在。正在创建...")
            
            # 1. 定义字段模式
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields, description="Medical RAG Collection")

            # 2. 创建集合
            collection = Collection(name=collection_name, schema=schema)
            st.write(f"集合 '{collection_name}' 创建成功。")

            # 3. 创建索引
            st.write(f"正在为 '{collection_name}' 创建索引 ({INDEX_TYPE})...")
            index_params = {
                "index_type": INDEX_TYPE,
                "metric_type": INDEX_METRIC_TYPE,
                "params": INDEX_PARAMS,
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            st.success(f"集合 '{collection_name}' 的索引创建成功。")
            
            # 4. 加载集合到内存（以便搜索）
            collection.load()
            st.write(f"集合 '{collection_name}' 已加载到内存。")
        else:
            st.write(f"发现已存在的集合: '{collection_name}'。")
            # 获取现有集合并加载
            collection = Collection(collection_name)
            # 检查是否已加载，若未加载则加载
            if not collection.is_loaded:
                collection.load()
                st.write(f"集合 '{collection_name}' 已加载到内存。")

        # 获取实体数量
        current_count = collection.num_entities
        st.write(f"集合 '{collection_name}' 准备就绪。当前实体数量: {current_count}")
        
        # 返回集合对象，以便后续操作
        return collection
        
    except Exception as e:
        st.error(f"设置 Milvus 集合 '{COLLECTION_NAME}' 时出错: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def index_data_if_needed(data, embedding_model):
    """检查数据是否需要索引，并使用标准 Milvus API 执行。"""
    global id_to_doc_map # 修改全局映射

    # 首先建立连接并获取/创建集合
    if not get_milvus_connection():
        st.error("无法连接到 Milvus 服务，索引中止。")
        return False
    
    collection = setup_milvus_collection()
    if not collection:
        st.error("无法获取或创建 Milvus 集合，索引中止。")
        return False

    # 检查当前集合中的实体数量
    current_count = collection.num_entities
    st.write(f"Milvus 集合 '{COLLECTION_NAME}' 中现有实体: {current_count}")

    data_to_index = data[:MAX_ARTICLES_TO_INDEX] # 限制数据量用于演示
    needed_count = 0
    entities_to_insert = {
        "id": [],
        "embedding": [],
        "title": [],
        "abstract": [],
        "source_file": [],
        "chunk_index": [],
    }
    temp_id_map = {} # 构建临时映射

    # 准备数据
    with st.spinner("正在为索引准备数据..."):
        for i, doc in enumerate(data_to_index):
            title = doc.get('title', '') or ""
            abstract = doc.get('abstract', '') or ""  # 这就是文本块
            source_file = doc.get('source_file', '')
            chunk_idx = doc.get('chunk_index', i)
            
            if not abstract:  # 使用abstract字段作为主要内容
                continue

            doc_id = int(doc.get("id", i))  # 尝试使用原id，确保是整数
            needed_count += 1
            
            # 准备插入到Milvus的实体数据
            entities_to_insert["id"].append(doc_id)
            entities_to_insert["title"].append(title)
            entities_to_insert["abstract"].append(abstract)
            entities_to_insert["source_file"].append(source_file)
            entities_to_insert["chunk_index"].append(chunk_idx)
            # embedding 稍后填充
            
            # 存储到临时全局映射
            temp_id_map[doc_id] = {
                'title': title,
                'abstract': abstract,
                'content': abstract,  # 对于搜索，内容就是abstract文本块
                'source_file': source_file,
                'chunk_index': chunk_idx
            }

    # 只有当新数据量大于现有数据，且有待处理文档时才进行索引
    if current_count < needed_count and len(entities_to_insert["id"]) > 0:
        st.warning(f"需要执行索引 ({current_count}/{needed_count} 个文档已存在)。这可能需要一些时间...")

        st.write(f"正在为 {len(entities_to_insert['id'])} 个文档生成嵌入向量...")
        with st.spinner("正在生成嵌入向量..."):
            start_embed = time.time()
            # 使用abstract文本生成嵌入向量
            embeddings = embedding_model.encode(entities_to_insert["abstract"], show_progress_bar=True)
            end_embed = time.time()
            st.write(f"嵌入向量生成耗时 {end_embed - start_embed:.2f} 秒。")

        # 填充嵌入向量
        entities_to_insert["embedding"] = embeddings.tolist()

        st.write("正在将数据插入 Milvus ...")
        with st.spinner("正在插入..."):
            try:
                start_insert = time.time()
                # 使用标准 Collection.insert 方法
                insert_result = collection.insert(entities_to_insert)
                # 刷新以确保数据持久化
                collection.flush()
                end_insert = time.time()
                
                # insert_result 包含插入的ID，我们可以获取实际插入数量
                inserted_count = len(entities_to_insert["id"])
                st.success(f"成功索引 {inserted_count} 个文档。插入耗时 {end_insert - start_insert:.2f} 秒。")
                
                # 只在成功插入后更新全局映射
                id_to_doc_map.update(temp_id_map)
                return True
            except Exception as e:
                st.error(f"向 Milvus 插入数据时出错: {e}")
                import traceback
                st.code(traceback.format_exc())
                return False
    elif current_count >= needed_count:
        st.write("数据量显示索引已完成。")
        # 如果映射为空但不需要索引，仍然填充映射
        if not id_to_doc_map:
            id_to_doc_map.update(temp_id_map)
        return True
    else: # 没有找到可嵌入的文档
        st.error("在数据中未找到有效的文本内容以供索引。")
        return False


def search_similar_documents(query, embedding_model):
    """在标准 Milvus 中搜索与查询相似的文档。"""
    # 确保连接已建立
    if not get_milvus_connection():
        st.error("Milvus 连接不可用于搜索。")
        return [], [], []

    try:
        # 获取集合
        collection = Collection(COLLECTION_NAME)
        # 确保集合已加载
        if not collection.is_loaded:
            collection.load()

        # 生成查询的嵌入向量
        query_embedding = embedding_model.encode([query]).tolist()

        # 定义搜索参数
        search_params = SEARCH_PARAMS

        # 执行搜索
        results = collection.search(
            data=query_embedding, 
            anns_field="embedding", 
            param=search_params,
            limit=TOP_K,
            output_fields=["id", "title", "abstract", "source_file", "chunk_index"]  # 指定要返回的字段
        )

        # 处理结果
        if not results or len(results[0]) == 0:
            return [], [], []

        hit_ids = []
        distances = []
        retrieved_docs = []

        for hits in results:
            for hit in hits:
                hit_ids.append(hit.id)
                distances.append(hit.distance)
                
                # 直接从搜索结果中构建文档信息
                doc_info = {
                    'title': hit.entity.get('title', ''),
                    'abstract': hit.entity.get('abstract', ''),
                    'content': hit.entity.get('abstract', ''),  # 内容即abstract
                    'source_file': hit.entity.get('source_file', ''),
                    'chunk_index': hit.entity.get('chunk_index', 0)
                }
                retrieved_docs.append(doc_info)
                
                # 同时更新全局映射（如果尚不存在）
                if hit.id not in id_to_doc_map:
                    id_to_doc_map[hit.id] = doc_info

        return hit_ids, distances, retrieved_docs
        
    except Exception as e:
        st.error(f"Milvus 搜索过程中出错: {e}")
        import traceback
        st.code(traceback.format_exc())
        return [], [], []