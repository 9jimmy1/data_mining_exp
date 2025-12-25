# node2vec_custom.py
import pandas as pd
import networkx as nx
import numpy as np
import random
from gensim.models import Word2Vec
from collections import defaultdict
import re
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import time

# ==============================
# 配置参数
# ==============================
DATA_FILE = "postings.csv"
SKILLS = [
    "python", "java", "javascript", "sql", "machine learning", "deep learning",
    "data analysis", "cloud computing", "aws", "azure", "docker", "kubernetes",
    "react", "nodejs", "html", "css", "git", "agile", "scrum", "project management",
    "communication", "leadership", "problem solving", "teamwork", "analytical skills"
]
SKILLS_LOWER = {s.lower(): s for s in SKILLS}


# ==============================
# 自定义 Node2Vec 实现
# ==============================
class CustomNode2Vec:
    def __init__(self, graph, walk_length=30, num_walks=100, p=1.0, q=1.0, workers=1):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers

    def random_walk(self, start_node):
        """执行随机游走"""
        walk = [start_node]
        current_node = start_node

        for _ in range(self.walk_length - 1):
            neighbors = list(self.graph.neighbors(current_node))
            if not neighbors:
                break

            # 简单的随机游走（可以扩展为node2vec的biased walk）
            next_node = random.choice(neighbors)
            walk.append(next_node)
            current_node = next_node

        return walk

    def generate_walks_parallel(self):
        """并行生成随机游走序列"""
        nodes = list(self.graph.nodes())
        all_walks = []

        # 使用进程池并行处理
        with multiprocessing.Pool(processes=self.workers) as pool:
            # 为每个进程分配任务
            for walk_num in tqdm(range(self.num_walks), desc="随机游走进度"):
                random.shuffle(nodes)

                # 并行生成游走序列
                walks = pool.map(self.random_walk, nodes)
                all_walks.extend(walks)

        return all_walks

    def generate_walks(self):
        """为所有节点生成随机游走序列（单线程版本）"""
        all_walks = []
        nodes = list(self.graph.nodes())

        # 使用tqdm显示进度
        for walk_num in tqdm(range(self.num_walks), desc="随机游走进度"):
            random.shuffle(nodes)

            # 内部进度条
            for node in tqdm(nodes, desc=f"第{walk_num + 1}/{self.num_walks}轮游走", leave=False):
                walk = self.random_walk(node)
                all_walks.append(walk)

        return all_walks


# ==============================
# 数据预处理函数
# ==============================
def extract_company(text):
    """提取公司名称"""
    if pd.isna(text) or str(text).strip().lower() in ['nan', 'none', '']:
        return None
    text = str(text).strip()

    # 处理邮箱格式
    if '@' in text:
        try:
            domain = text.split('@')[1].split('.')[0]
            return domain.title().replace(' ', '')
        except:
            pass

    return text[:50].split('\n')[0].strip()


def extract_skills(text):
    """从文本中提取技能"""
    if pd.isna(text):
        return set()

    text_lower = str(text).lower()
    matched_skills = set()

    # 匹配预定义技能
    for skill_lower, skill_orig in SKILLS_LOWER.items():
        if skill_lower in text_lower:
            matched_skills.add(skill_orig)

    return matched_skills


# ==============================
# 可视化函数
# ==============================
def visualize_embeddings(model, nodes, max_nodes=200):
    """使用T-SNE进行降维可视化"""
    if len(nodes) > max_nodes:
        print(f"节点过多({len(nodes)})，随机选择{max_nodes}个进行可视化")
        nodes = np.random.choice(nodes, max_nodes, replace=False)

    # 获取嵌入向量
    embeddings = [model.wv[node] for node in nodes]
    embeddings_array = np.array(embeddings)

    # T-SNE降维
    print("正在进行T-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(nodes) - 1))
    embeddings_2d = tsne.fit_transform(embeddings_array)

    # 创建可视化
    plt.figure(figsize=(15, 10))

    # 区分公司和技能节点
    companies = [node for node in nodes if node not in SKILLS]
    skills = [node for node in nodes if node in SKILLS]

    # 绘制公司节点
    company_indices = [i for i, node in enumerate(nodes) if node in companies]
    if company_indices:
        plt.scatter(embeddings_2d[company_indices, 0],
                    embeddings_2d[company_indices, 1],
                    c='blue', alpha=0.6, label='Companies', s=50)

        # 标注一些公司
        for i in company_indices[:10]:  # 只标注前10个公司
            plt.annotate(nodes[i],
                         (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.8)

    # 绘制技能节点
    skill_indices = [i for i, node in enumerate(nodes) if node in skills]
    if skill_indices:
        plt.scatter(embeddings_2d[skill_indices, 0],
                    embeddings_2d[skill_indices, 1],
                    c='red', alpha=0.6, label='Skills', s=50)

        # 标注技能
        for i in skill_indices:
            plt.annotate(nodes[i],
                         (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.8)

    plt.title('Custom Node2Vec Embeddings Visualization (T-SNE)')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图像
    plt.savefig('custom_node2vec_tsne.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ 可视化已保存: custom_node2vec_tsne.png")


# ==============================
# 主程序
# ==============================
def main():
    print("🔍 正在加载 LinkedIn 职位数据...")

    try:
        # 尝试不同的编码方式
        try:
            df = pd.read_csv(DATA_FILE, on_bad_lines='skip', low_memory=False)
        except:
            try:
                df = pd.read_csv(DATA_FILE, encoding='latin-1', on_bad_lines='skip', low_memory=False)
            except:
                # 如果前两列是公司信息，直接读取前几列
                df = pd.read_csv(DATA_FILE, usecols=[0, 1, 2], on_bad_lines='skip', low_memory=False)

        print(f"✅ 数据加载成功: {df.shape}")
        if len(df.columns) < 3:
            print(f"数据列: {df.columns.tolist()}")
        else:
            print("使用前三列数据进行处理")

    except FileNotFoundError:
        print(f"❌ 文件 '{DATA_FILE}' 未找到!")
        print("请确保文件在当前目录，或从以下链接下载:")
        print("https://www.kaggle.com/datasets/arshkon/linkedin-job-postings")

        # 创建示例数据用于演示
        print("\n⚠️ 使用示例数据进行演示...")
        sample_data = {
            'company': ['Google', 'Microsoft', 'Amazon', 'Apple', 'Meta'],
            'description': [
                'Looking for python java machine learning experts',
                'Need java sql cloud computing professionals',
                'Hiring for aws docker kubernetes engineers',
                'Seeking react javascript html css developers',
                'Want python data analysis machine learning talent'
            ]
        }
        df = pd.DataFrame(sample_data)
        print("示例数据已创建")

    # 构建公司-技能关系图
    print("\n📊 正在构建公司-技能关系图...")
    edges = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理数据行"):
        # 提取公司名称
        company = None
        for col_idx in range(min(3, len(row))):
            company = extract_company(row.iloc[col_idx])  # 修复警告
            if company:
                break

        if not company:
            continue

        # 合并所有文本内容进行技能提取
        full_text = " ".join([str(x) for x in row if pd.notna(x)])
        skills = extract_skills(full_text)

        # 添加边
        for skill in skills:
            edges.append((company, skill))

    print(f"✅ 图构建完成: {len(edges)} 条边")

    if len(edges) == 0:
        print("⚠️ 未找到任何关系边，使用示例数据")
        # 添加一些示例边
        edges = [
            ('Google', 'python'), ('Google', 'java'), ('Google', 'machine learning'),
            ('Microsoft', 'java'), ('Microsoft', 'sql'), ('Microsoft', 'cloud computing'),
            ('Amazon', 'aws'), ('Amazon', 'docker'), ('Amazon', 'kubernetes'),
            ('Apple', 'react'), ('Apple', 'javascript'), ('Apple', 'html'),
            ('Meta', 'python'), ('Meta', 'data analysis'), ('Meta', 'machine learning')
        ]

    # 创建图
    edge_df = pd.DataFrame(edges, columns=["source", "target"])
    G = nx.from_pandas_edgelist(edge_df, "source", "target", create_using=nx.Graph())

    print(f"📈 图统计: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")

    # 使用自定义Node2Vec训练模型
    print("\n🧠 正在训练自定义 Node2Vec 模型...")

    # 根据数据大小调整参数
    if G.number_of_nodes() > 50000:
        walk_length = 30
        num_walks = 100
        workers = 20  # 使用20个worker

    print(f"参数设置: walk_length={walk_length}, num_walks={num_walks}, workers={workers}")

    custom_node2vec = CustomNode2Vec(
        graph=G,
        walk_length=walk_length,
        num_walks=num_walks,
        p=1.0,
        q=1.0,
        workers=workers
    )

    # 生成随机游走序列
    print("生成随机游走序列...")

    # 根据系统资源选择并行或串行版本
    if workers > 1:
        print("使用并行版本生成随机游走...")
        walks = custom_node2vec.generate_walks_parallel()
    else:
        print("使用串行版本生成随机游走...")
        walks = custom_node2vec.generate_walks()

    print(f"生成了 {len(walks)} 条随机游走序列")

    # 使用Word2Vec训练节点嵌入
    print("训练Word2Vec模型...")
    model = Word2Vec(
        sentences=walks,
        vector_size=64,  # 嵌入维度
        window=5,  # 上下文窗口
        min_count=1,  # 最小出现次数
        workers=workers,  # 使用20个worker
        epochs=10
    )

    # 保存模型
    model.save("custom_node2vec.model")
    print("✅ 模型已保存: custom_node2vec.model")

    # 相似度计算示例
    print("\n🔍 相似度计算示例:")

    # 查找有嵌入的节点进行测试
    nodes_with_embeddings = list(model.wv.key_to_index.keys())
    print(f"有嵌入向量的节点数量: {len(nodes_with_embeddings)}")

    if nodes_with_embeddings:
        # 测试公司节点
        company_nodes = [n for n in nodes_with_embeddings if n not in SKILLS]
        if company_nodes:
            test_node = company_nodes[0]
            print(f"\n与 '{test_node}' 最相似的节点:")
            try:
                similar_nodes = model.wv.most_similar(test_node, topn=5)
                for node, similarity in similar_nodes:
                    node_type = "技能" if node in SKILLS else "公司"
                    print(f"  {node}: {similarity:.4f} ({node_type})")
            except Exception as e:
                print(f"相似度计算失败: {e}")

        # 测试技能节点
        skill_nodes = [n for n in nodes_with_embeddings if n in SKILLS]
        if skill_nodes:
            test_skill = skill_nodes[0]
            print(f"\n与 '{test_skill}' 最相似的节点:")
            try:
                similar_nodes = model.wv.most_similar(test_skill, topn=5)
                for node, similarity in similar_nodes:
                    node_type = "技能" if node in SKILLS else "公司"
                    print(f"  {node}: {similarity:.4f} ({node_type})")
            except Exception as e:
                print(f"相似度计算失败: {e}")

    # T-SNE可视化
    print("\n🎨 正在进行T-SNE可视化...")
    visualize_embeddings(model, nodes_with_embeddings)


if __name__ == "__main__":
    main()