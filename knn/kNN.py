import numpy as np
from os import listdir
import operator


# 图像矩阵转换为二进制向量
def img2vector(filename):
    return_vector = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            return_vector[0, 32 * i + j] = int(line[j])
    return return_vector


# knn算法:
# in_X 用于分类的输入向量
# data 训练样本集
# labels 标签向量
# k 选择最近邻的数量
def knnClassify(in_X, data, labels, k):
    data_size = data.shape[0]
    diff_mat = np.tile(in_X, (data_size, 1)) - data
    sq_diff = diff_mat ** 2
    sq_distance = sq_diff.sum(axis=1)
    distance = sq_distance ** 0.5
    sorted_dist = distance.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


# 手写数字识别
def handwritingTest():
    hw_labels = []
    training_list = listdir("trainingDigits")
    num_files = len(training_list)
    training = np.zeros((num_files, 1024))
    for i in range(num_files):
        # 文件名 0_1.txt，从文件名中解析分类数字，此例为0
        file = training_list[i]
        file_string = file.split(".")[0]
        number = int(file_string.split("_")[0])
        hw_labels.append(number)
        training[i, :] = img2vector("trainingDigits/%s" % file)
    testing_list = listdir('testDigits')
    error_count = 0.0
    num_test = len(testing_list)
    for i in range(num_test):
        t_file = testing_list[i]
        t_file_string = t_file.split('.')[0]
        t_number = int(t_file_string.strip('_')[0])
        t_vector = img2vector('testDigits/%s' % t_file)
        result = knnClassify(t_vector, training, hw_labels, 3)
        print("分类结果： %d , 实际值： %d" % (result, t_number))
        if result != t_number:
            error_count += 1
    print("\n 识别错误的数目：%d" % error_count)
    print("\n 错误率： %f" % (error_count/float(t_number)))


if __name__ == '__main__':
    handwritingTest()
