import random


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


# 输入文件名和输出文件名
input_file = 'output.txt'
train_file = 'train.txt'
dev_file = 'dev.txt'
test_file = 'test.txt'

# 按8:1:1的比例划分数据集
split_data(input_file, train_file, dev_file, test_file)
