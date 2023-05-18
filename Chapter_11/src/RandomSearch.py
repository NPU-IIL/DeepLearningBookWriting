from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import uniform
iris = load_iris()
# tol：停止训练的误差精度  max_iter：最大迭代次数，int类型，默认为-1，不限制   random_state ：数据洗牌时的种子值，int类型
logistic = LogisticRegression(solver='saga', tol=1e-2, multi_class='auto',max_iter=200, random_state=0)
# scipy.stats.uniform 均匀分布，属于连续性概率分布函数，参数 loc 和 scale 确定均匀分布的范围为 [loc, loc + scale]
distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])  # penalty 惩罚
clf = RandomizedSearchCV(logistic, distributions, cv=3, random_state=0)
search = clf.fit(iris.data, iris.target)
print(search.best_params_)
