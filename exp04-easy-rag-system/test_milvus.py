import sys
print("Python版本:", sys.version)
print("操作系统:", sys.platform)

try:
    # 注意：新版本pymilvus可能需要从pymilvus导入MilvusClient
    from pymilvus import MilvusClient
    print("✅ 成功导入 MilvusClient")
    
    # 尝试初始化客户端（使用和config.py中相同的路径）
    client = MilvusClient("./milvus_lite_data.db")
    print("✅ 成功初始化 Milvus Lite 客户端")
    
    # 列出已有的集合（刚开始应为空）
    collections = client.list_collections()
    print(f"现有集合: {collections}")
    
except Exception as e:
    print(f"❌ 测试失败，错误详情:")
    print(f"   类型: {type(e).__name__}")
    print(f"   信息: {e}")