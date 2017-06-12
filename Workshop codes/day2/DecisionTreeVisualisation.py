from sklearn import datasets
from sklearn import tree
import pydotplus


dataset = datasets.load_iris()
clf = tree.DecisionTreeClassifier()
clf.fit(dataset.data, dataset.target)


dot_data = dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=dataset.feature_names,
                         class_names=dataset.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")