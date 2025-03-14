import random
import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import dok_matrix  # 使用稀疏矩阵减少内存占用


class MyDataSet(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        return self.features[index].toarray(), self.labels[index]  # 转换为普通数组

    def __len__(self):
        return len(self.features)


def get_data_set(data_path, window_width, window_step, negative_sample_num):
    with open(data_path, 'r', encoding='utf-8') as file:
        document = file.read().replace(",", "").replace("?", "").replace(".", "").replace('"', '')
        data = document.split(" ")
        print(f"数据中共有 {len(data)} 个单词")

        # 生成词典
        vocabulary = list(set(data))
        print(f"词典大小为 {len(vocabulary)}")

        # 词典索引映射
        index_dict = {word: idx for idx, word in enumerate(vocabulary)}

        features = []
        labels = []
        neighbor_dict = {}

        vocab_size = len(vocabulary)

        for start_index in range(0, len(data), window_step):
            if start_index + window_width - 1 < len(data):
                mid_index = (start_index + start_index + window_width - 1) // 2
                for index in range(start_index, start_index + window_width):
                    if index != mid_index:
                        # 计算上下文单词与目标单词之间的距离
                        distance = abs(index - mid_index)
                        weight = 1 / distance

                        # 使用稀疏矩阵替代全零矩阵
                        feature = dok_matrix((vocab_size, vocab_size), dtype=np.float32)
                        feature[index_dict[data[mid_index]], index_dict[data[index]]] = weight

                        features.append(feature)
                        labels.append(1)

                        neighbor_dict.setdefault(data[mid_index], set()).add(data[index])

        # 负采样
        vocab_indices = list(range(vocab_size))  # 预生成索引列表，避免每次 `random.randint`
        for _ in range(negative_sample_num):
            random_word = vocabulary[random.choice(vocab_indices)]  # 确保索引合法
            for word in vocabulary:
                if word not in neighbor_dict.get(random_word, set()):
                    feature = dok_matrix((vocab_size, vocab_size))
                    feature[index_dict[random_word], index_dict[random_word]] = 1
                    feature[index_dict[word], index_dict[word]] = 1
                    features.append(feature)
                    labels.append(0)
                    break  # 确保每次只添加一个负样本

        return MyDataSet(features, labels), vocabulary, index_dict
