import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pylab as plt
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn import svm
import  datetime

# 导入数据集
dataframe0 = pd.read_csv('.\CW_Data.csv')
dataframe = pd.read_csv('finaldata.csv')
# 删除空行
dataframe0.drop(dataframe0[dataframe0['Programme'].isnull()].index,inplace=True)
# 观察数据之后发现ID一列没有实际意义删除
dataframe0.drop(labels='ID',axis=1,inplace=True)
# x = np.array(dataframe0.drop(labels="Programme", axis=1))
# 考虑后删除program0
# dataframe[(dataframe.Programme == 0)].index.tolist()
dataframe0.drop(dataframe0[(dataframe0.Programme == 0)].index,inplace=True)
#去掉重复数据保留后者
dataframe0.drop_duplicates(subset=[ 'Q1', 'Q2', "Q3" , "Q4" , "Q5"],inplace = True)
# x = np.array(dataframe.drop(labels="Programme", axis=1)
x = np.array(dataframe)
y = np.array(dataframe0["Programme"])
data = dataframe.values

stats = []
label = []
for i in range(len(data)):
    stats.append(data[i,:-1])
for i in range(len(data)):
    label.append(data[i,-1:])

#建立高斯，分类，补体贝叶斯分类器因为别的都有不适合的原因，并尝试优化
x_train, x_test, y_train, y_test = train_test_split(stats,label,test_size = 0.4,random_state=26)
GNB_estimate = GaussianNB()
GNB_estimate.fit(x_train,y_train)
predict_prgm = GNB_estimate.predict(x_test)
correctness_rate = GNB_estimate.score(x_test,y_test)
print('GaussianNB的准确率为：\n',correctness_rate)
print(predict_prgm)
cvscore = cross_val_score(GNB_estimate,stats,label,scoring = 'accuracy',cv=5)
print('GaussianNB的交叉验证得分：{}'.format(cvscore))
print('GaussianNB的交叉验证平均得分：{:.3f}'.format(cvscore.mean()))

# #筛选随机种子数对简单traintestsplit结果影响的曲线
# GNBtests = []
# randoms = []
# for rs in range(1,31,1):
#     randoms.append(rs)
#     gnbclf = GaussianNB()
#     x_train, x_test, y_train, y_test = train_test_split(stats, label, test_size=0.6, random_state=rs)
#     gnbclf.fit(x_train,y_train)
#     GNBp1 =gnbclf.predict(x_test)
#     GNBtest = gnbclf.score(x_test,y_test)
#     GNBtests.append(GNBtest)
# plt.plot(randoms,GNBtests)
# plt.xlabel('Values of random state')
# plt.ylabel('Accurate rate')
# plt.show()

# CmplmtNB_estimate = ComplementNB()
# CmplmtNB_estimate.fit(x_train,y_train)
# predict_prgm = CmplmtNB_estimate.predict(x_test)
# correctness_rate = CmplmtNB_estimate.score(x_test,y_test)
# print('ComplementNB的准确率为：\n',correctness_rate)
# print(predict_prgm)
# cvscore = cross_val_score(CmplmtNB_estimate,stats,label,scoring = 'accuracy',cv=5)
# print('ComplementNB的交叉验证得分：{}'.format(cvscore))
# print('ComplementNB的交叉验证平均得分：{:.3f}'.format(cvscore.mean()))
#
# Catgr_estimate = CategoricalNB()
# Catgr_estimate.fit(x_train,y_train)
# predict_prgm = Catgr_estimate.predict(x_test)
# correctness_rate = Catgr_estimate.score(x_test,y_test)
# print('CategoricalNB的准确率为：\n',correctness_rate)
# print(predict_prgm)
# cvscore = cross_val_score(Catgr_estimate,stats,label,scoring = 'accuracy',cv=5)
# print('CategoricalNB的交叉验证得分：{}'.format(cvscore))
# print('CategoricalNB的交叉验证平均得分：{:.3f}'.format(cvscore.mean()))