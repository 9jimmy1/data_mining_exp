# fix_csv_format.py
"""
修复CSV文件格式，使其统一
"""
import os
import csv


def fix_csv_file(input_path, output_path):
    """修复CSV文件格式"""
    print(f"修复文件: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    fixed_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # 使用csv.reader解析这一行
        try:
            # 尝试用不同的quoting方式解析
            reader = csv.reader([line], quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = next(reader)

            # 检查字段数量
            if len(row) < 3:
                # 如果字段不足，尝试不同的解析方式
                # 对于train.csv格式: "2","title","text"
                if line.startswith('"') and line.count('","') >= 2:
                    # 找到第二个和第三个引号的位置
                    first_quote_end = line.find('","', 1)
                    second_quote_end = line.find('","', first_quote_end + 3)

                    if second_quote_end != -1:
                        polarity = line[1:first_quote_end]
                        title = line[first_quote_end + 3:second_quote_end]
                        text = line[second_quote_end + 3:-1] if line.endswith('"') else line[second_quote_end + 3:]
                        row = [polarity, title, text]
                    else:
                        # 只有两个字段，可能是格式错误
                        print(f"警告: 行 {i + 1} 格式异常: {line}")
                        continue
                else:
                    print(f"警告: 行 {i + 1} 字段不足: {line}")
                    continue

            # 确保我们有3个字段
            if len(row) > 3:
                # 合并多余的字段到text中
                polarity = row[0]
                title = row[1]
                text = ','.join(row[2:])
                row = [polarity, title, text]

            # 清理字段
            polarity = str(row[0]).strip('" ')
            title = str(row[1]).strip('" ')
            text = str(row[2]).strip('" ')

            # 重新格式化为标准CSV（所有字段都用引号包围）
            fixed_line = f'"{polarity}","{title}","{text}"'
            fixed_lines.append(fixed_line)

        except Exception as e:
            print(f"解析行 {i + 1} 失败: {e}")
            print(f"  原始行: {line}")
            continue

    # 写入修复后的文件
    with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
        outfile.write('\n'.join(fixed_lines))

    print(f"修复完成: {len(fixed_lines)} 行写入 {output_path}")


def main():
    base_path = r"D:\PycharmProjects\exp02-sentiment-classificationn\datasets"

    # 创建修复后的文件夹
    fixed_path = os.path.join(base_path, "fixed")
    os.makedirs(fixed_path, exist_ok=True)

    # 修复所有CSV文件
    for file_name in ['train.csv', 'dev.csv', 'test.csv']:
        input_file = os.path.join(base_path, file_name)
        output_file = os.path.join(fixed_path, file_name)

        if os.path.exists(input_file):
            fix_csv_file(input_file, output_file)
        else:
            print(f"文件不存在: {input_file}")


if __name__ == "__main__":
    main()