import os
import json
import re
import random

def process_json_files(folder_path, output_file):
    # 打开输出文件准备写入
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # 遍历文件夹中的所有文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)

                    # 提取需要的字段并去除所有空白字符
                    review_data = data.get('review_data', '')
                    review_data = re.sub(r'\s+', ' ', review_data).strip()

                    img_paths = data.get('img_path', [])
                    if not isinstance(img_paths, list):
                        img_paths = [img_paths]

                    # 处理img_path字段，去掉换行符并移除'..\\img\\'
                    processed_img_paths = [re.sub(r'\s+', '', path).replace('..\\img\\', '') for path in img_paths]

                    # 将processed_img_paths列表转换为JSON数组格式的字符串
                    img_paths_str = json.dumps(processed_img_paths, ensure_ascii=False)

                    # 过滤 "http_undefined" 之后再将字符串解析为列表
                    filtered_img_paths = json.loads(img_paths_str)
                    filtered_img_paths = [path for path in filtered_img_paths if path != "http_undefined"]

                    # 如果没有图像，则跳过这组数据
                    if not filtered_img_paths:
                        continue

                    img_paths_str = json.dumps(filtered_img_paths, ensure_ascii=False)

                    items_score = data.get('items_score', {})

                    # 生成输出内容
                    for item, score in items_score.items():
                        if score is not None:  # 跳过值为null的项
                            outfile.write(f"{review_data}\n")
                            outfile.write(f"{img_paths_str}\n")
                            outfile.write(f"{item}\n")
                            outfile.write(f"{score}\n")

def split_data(input_file, train_file, dev_file, test_file, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # 每四行一组
    data = [lines[i:i + 4] for i in range(0, len(lines), 4)]

    # 随机打乱分组
    random.shuffle(data)

    # 计算各个集合的大小
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    dev_size = int(total_size * dev_ratio)
    test_size = total_size - train_size - dev_size  # 余下的分给测试集

    train_data = data[:train_size]
    dev_data = data[train_size:train_size + dev_size]
    test_data = data[train_size + dev_size:]

    # 写入训练集文件
    with open(train_file, 'w', encoding='utf-8') as train_out:
        for group in train_data:
            train_out.writelines(group)

    # 写入验证集文件
    with open(dev_file, 'w', encoding='utf-8') as dev_out:
        for group in dev_data:
            dev_out.writelines(group)

    # 写入测试集文件
    with open(test_file, 'w', encoding='utf-8') as test_out:
        for group in test_data:
            test_out.writelines(group)

# 处理JSON文件
# 文件夹路径和输出文件名
folder_path = './multi_jsonData'
output_file = 'output.txt'
# 处理JSON文件并生成输出
process_json_files(folder_path, output_file)

# 输入文件名和输出文件名
input_file = 'output.txt'
train_file = './datasets/train.txt'
dev_file = './datasets/dev.txt'
test_file = './datasets/test.txt'

# 按8:1:1的比例划分数据集
split_data(input_file, train_file, dev_file, test_file)
