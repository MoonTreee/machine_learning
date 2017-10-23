import numpy as np
import scrapy


class Perceptron(object):
    """
    eta:学习率
    n_iter:权重向量的训练次数
    w_:神经分叉权重的向量
    errors_:用户记录神经元判断的出错次数
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        pass

    def fit(self, X, y):
        """
        输入训练数据，培训神经元
        :param x: 输入样本的向量
        :param y: 对应的样本分类
        :return:
        X:shape[n_samples, n_features]
        X:[[1,2,3], [4,5,6]]
        n_samples:2
        n_features:3

        y:[1:-1]
        """
        # 初始化权重向量为零,1+是因为要引入w0，也就是步调函数的阈值
        self.w_ = np.zero(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            """
            X:[[1,2,3], [4,5,6]]
            y:[1,-1]
            zip(X, y)=[[1,2,3,1], [4,5,6,-1]]
            """
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)
                pass
            pass

    def net_input(self, X):
        """
       进行点积的运算
       :param self:
       :param X:
       :return:
       """
        return np.dot(X, self.w_[1:]+self.w_[0])

    def predict(self, X):
        return np.where(self.net_imput(X) >= 0.0, 1, -1)

