import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import re
from tqdm import tqdm  # ✅ 导入 tqdm

def preprocess_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    return tokens


def get_document_vector(text, model):
    tokens = preprocess_text(text)
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)


def main():
    file_path = 'dataset/dev.csv'

    print("正在加载数据...")
    df = pd.read_csv(file_path)
    print(f"数据形状: {df.shape}")

    # 合并 title 和 review 为 text
    title_series = df.iloc[:, 1].fillna('').astype(str)
    review_series = df.iloc[:, 2].fillna('').astype(str)
    df['text'] = title_series + " " + review_series

    labels = df.iloc[:, 0].values
    texts = df['text'].tolist()  # ✅ texts 在这里定义！

    # ✅ 正确位置：在 main() 内部使用 texts
    print("开始预处理文本（带进度条）...")
    corpus = [preprocess_text(text) for text in tqdm(texts, desc="预处理")]

    print("训练 Word2Vec 模型...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=50,
        window=3,
        min_count=5,
        workers=2,
        epochs=5
    )

    print("生成文档向量...")
    doc_vectors = []
    for text in tqdm(texts, desc="生成向量"):
        vec = get_document_vector(text, model)
        doc_vectors.append(vec)

    X = np.array(doc_vectors)
    y = labels

    print("\n✅ 完成!")
    print("文档向量形状:", X.shape)
    print("标签形状:", y.shape)

    # 保存模型
    model.save("word2vec_dev.model")

    # 测试相似词
    word = "great"
    if word in model.wv:
        similar = model.wv.most_similar(word, topn=5)
        print(f"\n与 '{word}' 最相似的词:")
        for w, score in similar:
            print(f"  {w}: {score:.4f}")


if __name__ == "__main__":
    main()