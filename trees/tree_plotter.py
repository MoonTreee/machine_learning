import matplotlib.pyplot as plt

# 用字典进行存储
# boxstyle为文本框属性， 'sawtooth'：锯齿型；fc为边框粗细
decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


# node_txt 要注解的文本，center_pt文本中心点，箭头指向的点，parent_pt箭头的起点
def plotNode(node_txt, center_pt, parent_pt, node_type):
    createPlot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                            xytext=center_pt, textcoords='axes fraction',
                            va="center", ha="center", bbox=node_type, arrowprops=arrow_args)


# 创建画板
def createPlot(in_tree):
    # figure创建画板，‘1’表示第一个图，背景为白色
    fig = plt.figure(1, facecolor='white')
    # 清空画板
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # subplot(x*y*z）,表示把画板分割成x*y的网格，z是画板的标号，
    # frameon=False表示不绘制坐标轴矩形
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # plotNode('decision_node', (0.5, 0.1), (0.1, 0.5), decision_node)
    # plotNode('leaf_node', (0.8, 0.1), (0.8, 0.3),  leaf_node)
    # plt.show()
    # 存储树的宽度
    plotTree.totalW = float(getNumLeafs(in_tree))
    # 存储树的深度
    plotTree.totalD = float(getTreeDepth(in_tree))
    # xOff用于追踪已经绘制的节点的x轴位置信息，为下一个节点的绘制提供参考
    plotTree.xOff = -0.5/plotTree.totalW
    # yOff用于追踪已经绘制的节点y轴的位置信息，为下一个节点的绘制提供参考
    plotTree.yOff = 1.0
    plotTree(in_tree, (0.5, 1.0), '')
    plt.show()


# 为了绘制树，要先清楚叶子节点的数量以及树的深度--以便确定x轴的长度和y轴的高度
# 下面就分别定义这两个方法
def getNumLeafs(my_tree):
    num_leafs = 0
    first_str = next(iter(my_tree))  # 找到第一个节点
    second_dic = my_tree[first_str]
    # 测试节点数据是否为字典类型，叶子节点不是字典类型
    for key in list(second_dic.keys()):
        # 如果节点为字典类型，则递归使用getNumLeafs()
        if type(second_dic[key]).__name__ == 'dict':
            num_leafs += getNumLeafs(second_dic[key])
        else:
            num_leafs += 1
    return num_leafs


def getTreeDepth(my_tree):
    max_depth = 0
    first_str = next(iter(my_tree))
    second_dic = my_tree[first_str]
    # 测试节点数据是否为字典类型，叶子节点不是字典类型
    for key in list(second_dic.keys()):
        # 如果节点为字典类型，递归使用getTreeDepth()
        if type(second_dic[key]).__name__ == 'dict':
            this_depth = 1 + getTreeDepth(second_dic[key])
        else:
            # 当节点不为字典型，为叶子节点，深度遍历结束
            # 从递归中调用返回，且深度加1
            this_depth = 1
        # 最大的深度存储在max_depth中
        if this_depth > max_depth:
            max_depth = this_depth

    return max_depth


# 在父子节点之间填充文本信息进行标注
# 在决策树中此处应是对应父节点的属性值
def plotMidText(center_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - center_pt[0])/2.0 + center_pt[0]
    y_mid = (parent_pt[1] - center_pt[1])/2.0 + center_pt[1]
    createPlot.ax1.text(x_mid, y_mid, txt_string)


def plotTree(my_tree, parent_pt, node_txt):
    num_leafs = getNumLeafs(my_tree)
    depth = getTreeDepth(my_tree)
    first_str = list(my_tree.keys())[0]
    # 以第一次调用为例说明
    # 此时 绘制的为根节点，根节点的x轴：-0.5/plotTree.totalW + (1.0 + float(num_leafs))/2.0/plotTree.totalW
    # 假设整个树中叶子节点的数目为6 则上述根节点的x轴：-0.5/6 + (1 + 6)/2.0/6 = 0.5
    # 实际上，对于根节点而言，下式的值始终是0.5
    center_pt = (plotTree.xOff + (1.0 + float(num_leafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(center_pt, parent_pt, node_txt)
    plotNode(first_str, center_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]
    # y轴的偏移--深度优先的绘制策略
    plotTree.yOff -= 1.0 / plotTree.totalD
    for key in list(second_dict.keys()):
        if type(second_dict[key]).__name__ == 'dict':
            plotTree(second_dict[key], center_pt, str(key))
        else:
            plotTree.xOff += 1.0 / plotTree.totalW
            plotNode(second_dict[key], (plotTree.xOff, plotTree.yOff), center_pt, leaf_node)
            plotMidText((plotTree.xOff, plotTree.yOff), center_pt, str(key))
    plotTree.yOff += 1.0 / plotTree.totalD
