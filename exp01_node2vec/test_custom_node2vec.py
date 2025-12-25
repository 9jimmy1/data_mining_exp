# test_custom_node2vec.py
from gensim.models import Word2Vec
import sys


def main():
    model_path = "custom_node2vec.model"

    print("正在加载自定义 Node2Vec 模型...")
    try:
        model = Word2Vec.load(model_path)
    except FileNotFoundError:
        print(f"错误: 模型文件 '{model_path}' 未找到。")
        print("请先运行 node2vec_custom.py 训练模型")
        return
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    vocab_size = len(model.wv.key_to_index)
    vector_size = model.vector_size
    print(f"✅ 模型加载成功!")
    print(f"   词汇表大小: {vocab_size}")
    print(f"   向量维度: {vector_size}")
    print()

    # 显示一些节点示例
    nodes = list(model.wv.key_to_index.keys())
    print("节点示例:")
    companies = [n for n in nodes if n not in [
        "python", "java", "javascript", "sql", "machine learning", "deep learning",
        "data analysis", "cloud computing", "aws", "azure", "docker", "kubernetes",
        "react", "nodejs", "html", "css", "git", "agile", "scrum", "project management",
        "communication", "leadership", "problem solving", "teamwork", "analytical skills"
    ]][:5]
    skills = [n for n in nodes if n in [
        "python", "java", "javascript", "sql", "machine learning", "deep learning",
        "data analysis", "cloud computing", "aws", "azure", "docker", "kubernetes",
        "react", "nodejs", "html", "css", "git", "agile", "scrum", "project management",
        "communication", "leadership", "problem solving", "teamwork", "analytical skills"
    ]][:5]

    print("  公司:", ", ".join(companies))
    print("  技能:", ", ".join(skills))
    print()

    print("使用说明:")
    print("- 输入一个节点名称查看相似节点")
    print("- 输入两个节点计算相似度")
    print("- 输入 'vector 节点名' 查看向量")
    print("- 输入 'quit' 或 'exit' 退出")
    print()

    while True:
        try:
            user_input = input("请输入命令: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出...")
            break

        if user_input.lower() in ['quit', 'exit']:
            print("退出...")
            break

        if not user_input:
            continue

        # 处理 vector 命令
        if user_input.startswith("vector "):
            node = user_input[7:].strip()
            if node in model.wv:
                vec = model.wv[node]
                print(f"节点 '{node}' 的向量 (前10维):")
                print(f"{vec[:10]}\n")
            else:
                print(f"节点 '{node}' 不在模型中\n")
            continue

        words = user_input.split()

        if len(words) == 1:
            node = words[0]
            if node in model.wv:
                print(f"与 '{node}' 最相似的节点:")
                similar = model.wv.most_similar(node, topn=10)
                for i, (n, score) in enumerate(similar, 1):
                    node_type = "技能" if n in [
                        "python", "java", "javascript", "sql", "machine learning", "deep learning",
                        "data analysis", "cloud computing", "aws", "azure", "docker", "kubernetes",
                        "react", "nodejs", "html", "css", "git", "agile", "scrum", "project management",
                        "communication", "leadership", "problem solving", "teamwork", "analytical skills"
                    ] else "公司"
                    print(f"  {i}. {n:<20} ({node_type}) - {score:.4f}")
                print()
            else:
                print(f"节点 '{node}' 不在模型中\n")

        elif len(words) == 2:
            node1, node2 = words
            if node1 in model.wv and node2 in model.wv:
                similarity = model.wv.similarity(node1, node2)
                node1_type = "技能" if node1 in [
                    "python", "java", "javascript", "sql", "machine learning", "deep learning",
                    "data analysis", "cloud computing", "aws", "azure", "docker", "kubernetes",
                    "react", "nodejs", "html", "css", "git", "agile", "scrum", "project management",
                    "communication", "leadership", "problem solving", "teamwork", "analytical skills"
                ] else "公司"
                node2_type = "技能" if node2 in [
                    "python", "java", "javascript", "sql", "machine learning", "deep learning",
                    "data analysis", "cloud computing", "aws", "azure", "docker", "kubernetes",
                    "react", "nodejs", "html", "css", "git", "agile", "scrum", "project management",
                    "communication", "leadership", "problem solving", "teamwork", "analytical skills"
                ] else "公司"
                print(f"'{node1}' ({node1_type}) 和 '{node2}' ({node2_type}) 的相似度: {similarity:.4f}\n")
            else:
                missing = [n for n in [node1, node2] if n not in model.wv]
                print(f"以下节点不在模型中: {', '.join(missing)}\n")

        else:
            print("请输入1个或2个节点，或使用 'vector 节点名' 命令\n")


if __name__ == "__main__":
    main()