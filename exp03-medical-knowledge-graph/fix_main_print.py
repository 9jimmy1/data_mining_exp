import re

# 读取 main.py 内容
with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到并替换有问题的打印行
# 原代码大约是这样的：print(f"- {key}: {', '.join(values)}")
# 我们将其替换为能处理字符串和字典的版本
old_line = r'print\(f\"- \{key\}: \{', '\.join\(values\)\}\"\)'
new_code = '''
    # 修复打印：同时兼容字符串列表和字典列表
    display_values = []
    for item in values:
        if isinstance(item, dict):
            # 如果是字典，格式化为 "类型:名称"
            display_values.append(f"{item.get('type', 'Unknown')}:{item.get('name', 'Unknown')}")
        else:
            # 如果是字符串，直接使用
            display_values.append(str(item))
    print(f"- {key}: {', '.join(display_values)}")
'''

# 使用正则表达式进行替换，确保精确找到目标行
# 注意：如果原代码格式有差异，这里可能需要调整。我们先尝试精确匹配。
# 更安全的方法：直接替换整个代码块
pattern = r'(\s*)print\(f\"-\s\{key\}:\s\{', '\.join\(values\)\}\"\)'
replacement = r'\1# 修复打印：同时兼容字符串列表和字典列表\n\1display_values = []\n\1for item in values:\n\1    if isinstance(item, dict):\n\1        # 如果是字典，格式化为 "类型:名称"\n\1        display_values.append(f"{item.get(\"type\", \"Unknown\")}:{item.get(\"name\", \"Unknown\")}")\n\1    else:\n\1        # 如果是字符串，直接使用\n\1        display_values.append(str(item))\n\1print(f"- {key}: {', '.join(display_values)}")'

# 执行替换
new_content = re.sub(pattern, replacement, content)

# 保存修改
with open('main.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ 已修复 main.py 中的打印逻辑。")
