from math import log
import operator


# 计算信息熵
def clacShannon(data):
    num = len(data)
    # 用于计算每个类别出现的次数，配合num就可以计算该类别的概率了
    label_count = {}
    for feat_vec in data:
        # 向量的最后一个为标签（类别）
        current_label = feat_vec[-1]
        if current_label in label_count.keys():
            label_count[current_label] += 1
        else:
            label_count[current_label] = 1
    entropy = 0.0
    for key in label_count.keys():
        prob = float(label_count[key]) / num
        entropy -= prob * log(prob, 2)
    return entropy


# 划分数据集
# data 需要划分的数据，双重列表[[],[],……,[]]
# axis 划分的特征
# value 上述axis的值
def splitData(data, axis, value):
    result = []
    for feature_vec in data:
        if feature_vec[axis] == value:
            # 将符合条件的实例加入到result中（并去除了相应的特征）
            # result.append(feature_vec[:axis].extend(feature_vec[axis + 1:]))
            reduce_feat = feature_vec[:axis]
            reduce_feat.extend(feature_vec[axis+1:])
            result.append(reduce_feat)
    return result


# 选择分类效果最好的特征
def chooseFeature(data):
    num_data = len(data)
    num_feature = len(data[0]) - 1
    # 原始数据的香农熵
    base_entropy = clacShannon(data)
    # 信息增益
    best_info_gain = 0.0
    # 分类效果最好的特征
    best_feature = -1
    # 计算各个特征的信息增益
    for i in range(num_feature):
        # 获取特征i下所有可能的取值,并去重
        feature_list = [example[i] for example in data]
        values = set(feature_list)
        # 新的香农熵
        new_entropy = 0.0
        for value in values:
            sub_data = splitData(data, i, value)
            prob = len(sub_data) / float(num_data)
            new_entropy += prob * clacShannon(sub_data)
        info_gain = base_entropy - new_entropy
        # 选择分类效果最好--信息增益最大
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


# 使用投票机制确定节点的类别
def majorityCnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    # 选取票数最多的作为分类作为该节点的最终类别
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reversed=True)
    return sorted_class_count[0][0]


# 创建决策树
# data 数据集，每条记录的最后一项为该实例的类别
# labels 为了增加结果的可解释性，设定标签
def createTree(data, labels):
    # data中每条记录的最后一项为该实例的类别
    class_list = [example[-1] for example in data]
    # 结束条件一：该分支下所有记录的类别相同，则为叶子节点，停止分类
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 结束条件二：所有特征使用完毕，该节点为叶子节点，节点类别投票决定
    if len(data[0]) == 1:
        return majorityCnt(class_list)
    best_feature = chooseFeature(data)
    best_label = labels[best_feature]
    my_tree = {best_label: {}}
    del(labels[best_feature])
    feature_values = [example[best_feature] for example in data]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_lables = labels[:]
        # 递归
        my_tree[best_label][value] = createTree(splitData(data, best_feature, value), sub_lables)
    return my_tree


# 使用决策树进行分类
# 参数说明：决策树， 标签， 待分类数据
def classify(input_tree, feature_labels, test_vec):
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    # 得到第特征的索引，用于后续根据此特征的分类任务
    feature_index = feature_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                classLabel = classify(second_dict[key], feature_labels, test_vec)
            # 达到叶子节点，返回递归调用，得到分类
            else:
                classLabel = second_dict[key]
    return classLabel


# 决策树的存储
# 决策树的构造是一个很耗时的过程，因此需要将构造好的树保存起来以备后用
# 使用pickle序列化对象
def storeTree(input_tree, filename):
    import pickle
    fw = open(filename, "w")
    pickle.dump(input_tree, fw)
    fw.close()


# 读取文件中的决策树
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)