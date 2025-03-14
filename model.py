from torch import nn


class DNN(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim):
        super(DNN, self).__init__()

        self.embedding = nn.Linear(vocabulary_size, embedding_dim, bias=False)
        print("embedding_size:", list(self.embedding.weight.size()))

        self.layers = nn.Sequential(
            # vocabulary_size * embedding_dim可以理解为，每个词要用好几个维度表示，所以综合出来的矩阵尺寸就是这样的
            # 为什么最后输出的维度要小，因为是为了提取特征，减小计算，为后面做准备
            nn.Linear(vocabulary_size * embedding_dim, embedding_dim // 2),
            # 改进版的relu激活函数
            # 解决标准 ReLU 在负输入区域梯度为零（即“死亡 ReLU”问题）的缺陷。它是一种改进版的 ReLU，允许负输入有一个非零的斜率。
            nn.LeakyReLU(),
            nn.Linear(embedding_dim // 2, 4),
            nn.LeakyReLU(),
            # 是个回归问题，因此只需要一个预测值
            # 结果的维度是多少应该是要看解决的问题的实际需求
            nn.Linear(4, 1),
        )

        # Mean squared error loss
        # 均方误差损失（Mean Squared Error Loss，简称 MSE）
        # 计算损失函数
        self.criterion = nn.MSELoss()
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size()[0], -1)
        x = self.layers(x)
        x = x.squeeze(1)
        return x

    def cal_loss(self, pred, target):
        """ Calculate loss """
        return self.criterion(pred, target)
