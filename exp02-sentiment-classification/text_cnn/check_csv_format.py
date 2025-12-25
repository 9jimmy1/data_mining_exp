# check_csv_format.py
import pandas as pd
import os


def check_csv_format(file_path):
    print(f"\n检查文件: {file_path}")
    if not os.path.exists(file_path):
        print(f"  文件不存在")
        return

    try:
        # 尝试读取前几行
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [next(f) for _ in range(5)]

        print(f"  文件前5行:")
        for i, line in enumerate(lines):
            print(f"  行{i + 1}: {line.strip()}")

        # 尝试用pandas读取
        print(f"\n  尝试用pandas读取...")
        try:
            df = pd.read_csv(file_path, header=None, names=['polarity', 'title', 'text'], quoting=3)
            print(f"  成功读取 {len(df)} 行")
            print(f"  前几行数据:")
            print(df.head())
            print(f"\n  列名: {df.columns.tolist()}")
            print(f"  极性值分布:")
            print(df[0].value_counts())
        except Exception as e:
            print(f"  pandas读取失败: {e}")

            # 尝试其他编码
            encodings = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, header=None, names=['polarity', 'title', 'text'],
                                     encoding=encoding, quoting=3)
                    print(f"  使用 {encoding} 编码成功读取 {len(df)} 行")
                    break
                except:
                    continue

    except Exception as e:
        print(f"  检查失败: {e}")


if __name__ == "__main__":
    base_path = r"D:\PycharmProjects\exp02-sentiment-classificationn\datasets"

    # 检查datasets文件夹是否存在
    if not os.path.exists(base_path):
        print(f"路径不存在: {base_path}")
        # 尝试另一个可能的文件夹名
        base_path = r"D:\PycharmProjects\exp02-sentiment-classificationn\dataset"
        if not os.path.exists(base_path):
            print(f"路径也不存在: {base_path}")
            exit(1)

    print(f"检查数据集文件夹: {base_path}")

    # 检查文件夹内容
    print(f"\n文件夹内容:")
    for f in os.listdir(base_path):
        print(f"  - {f}")

    # 检查各个CSV文件
    for file_name in ['train.csv', 'dev.csv', 'test.csv']:
        file_path = os.path.join(base_path, file_name)
        check_csv_format(file_path)