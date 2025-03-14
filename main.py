import random
from math import sqrt

import numpy as np
import torch
from torch.utils.data import DataLoader

# from dataset import get_data_set
from dataset_cbow import *
from model import DNN


def same_seed(seed):
    """
    Fixes random number generator seeds for reproducibility
    固定时间种子。由于cuDNN会自动从几种算法中寻找最适合当前配置的算法，为了使选择的算法固定，所以固定时间种子
    :param seed: 时间种子
    :return: None
    """
    # torch.backends.cudnn.deterministic 用于控制cuDNN（深度神经网络库），确保在使用gpu进行计算的训练和推理过程是确定的
    # 因为在运行过程中，cuDNN会自动选择一些更好的优化算法，这可能会导致复现过程中出现一些不可复现的实验结果
    # 所以一般会选择true
    torch.backends.cudnn.deterministic = True  # 解决算法本身的不确定性，设置为True 保证每次结果是一致的
    # 这个也是，这个是判断是否要动态选择最优的卷积算法
    # 会导致不确定性，所以选false
    # 不介意不确定性影响可以选true
    torch.backends.cudnn.benchmark = True  # 解决了算法选择的不确定性，方便复现，提升训练速度
    np.random.seed(seed)  # 按顺序产生固定的数组，如果使用相同的seed，则生成的随机数相同， 注意每次生成都要调用一次
    torch.manual_seed(seed)  # 手动设置torch的随机种子，使每次运行的随机数都一致
    random.seed(seed)
    if torch.cuda.is_available():
        # 为GPU设置唯一的时间种子
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


import logging
from tqdm import tqdm  # 导入进度条库

# 设置日志记录
logging.basicConfig(filename="training_cbow.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train(model, train_loader, config):
    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hyper_paras'])

    device = config['device']
    epoch = 0
    while epoch < config['n_epochs']:
        model.train()  # 设置模型为训练模式
        loss_arr = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['n_epochs']}", unit="batch")  # 添加进度条

        for x, y in progress_bar:
            optimizer.zero_grad()  # 清除梯度
            x, y = x.to(device).to(torch.float32), y.to(device).to(torch.float32)  # 传输到设备
            pred = model(x)  # 前向传播
            mse_loss = model.cal_loss(pred, y)  # 计算损失
            mse_loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            loss_arr.append(mse_loss.item())

            # 更新进度条显示
            progress_bar.set_postfix(loss=mse_loss.item())

        avg_loss = np.mean(loss_arr)
        print(f"Epoch {epoch+1}/{config['n_epochs']} - Loss: {avg_loss}")
        logging.info(f"Epoch {epoch+1}/{config['n_epochs']} - Loss: {avg_loss}")  # 记录日志

        epoch += 1

    print('Finished training after {} epochs'.format(epoch))
    logging.info('Finished training after {} epochs'.format(epoch))


def find_min_distance_word_vector(cur_i, vector, embeddings, vocabulary):
    # 用来计算相似度
    def calc_distance(v1, v2):
        # 计算欧式距离
        distance = 0
        for i in range(len(v1)):
            distance += sqrt(pow(v1[i] - v2[i], 2))
        return distance

    min_distance = None
    min_i = -1
    for i, word in enumerate(vocabulary):
        if cur_i != i:
            distance = calc_distance(vector, embeddings[i].tolist())
            if min_distance is None or min_distance > distance:
                min_distance = distance
                min_i = i
    return min_i


if __name__ == '__main__':
    data_path = 'train.txt'
    config = {
        'seed': 3407,  # Your seed number, you can pick your lucky number. :)
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_epochs': 10,  # Number of epochs.
        'batch_size': 16,
        'optimizer': 'AdamW',
        'optim_hyper_paras': {  # hyper-parameters for the optimizer (depends on which optimizer you are using)
            'lr': 0.001,  # learning rate of optimizer
        },
        'embedding_dim': 6,  # 词向量长度
        'window_width': 5,  # 窗口的宽度
        'window_step': 2,  # 窗口滑动的步长
        'negative_sample_num': 10  # 要增加的负样本个数
    }

    same_seed(config['seed'])

    data_set, vocabulary, index_dict = get_data_set(data_path, config['window_width'], config['window_step'],
                                                    config['negative_sample_num'])
    # train_loader = DataLoader(data_set, config['batch_size'], shuffle=True, drop_last=False, pin_memory=True)
    train_loader = DataLoader(data_set, config['batch_size'], shuffle=True, drop_last=False, pin_memory=True, num_workers=2)



    model = DNN(len(vocabulary), config['embedding_dim']).to(config['device'])

    train(model, train_loader, config)

    # 训练完，看看embeddings，展示部分词的词向量，并找到离它最近的词的词向量
    embeddings = torch.t(model.embedding.weight)
    for i in range(10):
        print('%-50s%s' % (f"{vocabulary[i]} 的词向量为 :", str(embeddings[i].tolist())))
        min_i = find_min_distance_word_vector(i, embeddings[i].tolist(), embeddings, vocabulary)
        print('%-45s%s' % (
            f"离 {vocabulary[i]} 最近的词为 {vocabulary[min_i]} , 它的词向量为 :", str(embeddings[min_i].tolist())))
        print('-' * 200)
