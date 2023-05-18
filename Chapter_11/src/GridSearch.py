from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
# kernel：核函数   C:惩罚项，默认为1.0，C越大容错空间越小
parameters = {'kernel':('linear','poly','rbf'), 'C':[1, 10, 100],'gamma': [0.001, 0.01, 0.1, 1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters,cv=3,iid=False)
search = clf.fit(iris.data, iris.target)
print(search.best_params_)
print("Best parameters: {}".format(search.best_params_))
