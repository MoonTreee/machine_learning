import shannon
import tree_plotter
from sklearn import tree

if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_labels = ['age', 'prescript', 'astigmatic', 'tear_rate']
    lenses_tree = shannon.createTree(lenses, lenses_labels)
    print(lenses_tree)
    tree_plotter.createPlot(lenses_tree)

    # d = {'tear_rate': {'reduced': 'no lenses', 'normal': {'astigmatic': {'no': {'age': {'pre': 'soft', 'young': 'soft',
    #      'presbyopic': {'prescript': {'hyper': 'soft', 'myope': 'no lenses'}}}}, 'yes': {'prescript': {'hyper': {'age':
    #     {'pre': 'no lenses', 'young': 'hard', 'presbyopic': 'no lenses'}}, 'myope': 'hard'}}}}}}
    # num = tree_plotter.getNumLeafs(d)
    # print(num)

    # clf = tree.DecisionTreeClassifier()
    # lenses = clf.fit(lenses, lenses_labels)


