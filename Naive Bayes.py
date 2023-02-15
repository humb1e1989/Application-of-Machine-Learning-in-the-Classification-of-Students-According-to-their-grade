# import warnings
# warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import  datetime
# 开始时间
begin = datetime.datetime.now()
# # 高斯朴素贝叶斯
# # 导入数据集
# dataframe = pd.read_csv('.\CW_Data.csv')
# # 删除空行
# dataframe.drop(dataframe[dataframe['Programme'].isnull()].index,inplace=True)
# x = np.array(dataframe.drop(labels="Programme", axis=1))
# y = np.array(dataframe["Programme"])
#
# x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=100)
# # 假设符合高斯分布，使用高斯贝叶斯进行计算
# clf = GaussianNB(priors=None)
# clf.fit(x_train,y_train)
# GaussianNB(priors=None)
# # 验证
# x_pred = clf.predict(x_test)
# acc = sum(x_pred == y_test) / x_pred.shape[0]
# print("准确率ACC: %.3f", acc)
# # 交叉验证
# scores = cross_val_score(clf, x, y, cv = 5)
# print(scores)
# print("高斯朴素贝叶斯预测的分类结果:",x_pred)
# 根据此类分类结果 发现只有 0 / 1 / 2 / 3 没有 4 猜测不符合高斯分布，而且结合准确率在30到60
# 可视化结果
# programme_count = x_pred['0'].value_counts()
# cross_validation.train_test_split(train_data,train_target,test_size=0.4, random_state=0)
# ---------------------------------------------------------------------------------------------------------------------
# 分类特征朴素贝叶斯
# 导入数据集
dataframe = pd.read_csv('.\CW_Data.csv')
# 删除空行
dataframe.drop(dataframe[dataframe['Programme'].isnull()].index,inplace=True)
# 观察数据之后发现ID一列没有实际意义删除
dataframe.drop(labels='ID',axis=1,inplace=True)
#去掉重复数据保留后者
dataframe.drop_duplicates(subset=[ 'Q1', 'Q2', "Q3" , "Q4" , "Q5"],inplace = True)
# 考虑后删除program0
dataframe[(dataframe.Programme == 0)].index.tolist()
dataframe.drop(dataframe[(dataframe.Programme == 0)].index,inplace=True)

x = np.array(dataframe.drop(labels="Programme", axis=1))
y = np.array(dataframe["Programme"])

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=100)
# 假设符合分布，分类贝叶斯
# clf = CategoricalNB()
# clf.fit(x_train,y_train)
# CategoricalNB()
# # 验证
# x_pred = clf.predict(x_test)
# acc = sum(x_pred == y_test) / x_pred.shape[0]
# print("准确率ACC: %.3f", acc)
# # 交叉验证
# scores = cross_val_score(clf, x, y, cv = 5)
# print(scores)
# print("分类朴素贝叶斯预测的分类结果:",x_pred)
# 交叉验证结果 print 出nan 准确率在0.45-0.60之间 结果也没有出现 3/4 的programme 只有1和2 结合前面的programme的counter发现这个分类偷懒了
# 因为program1/2的人数最多
# ---------------------------------------------------------------------------------------------------------------------
# # 多项式模型朴素贝叶斯
# # 导入数据集
# dataframe = pd.read_csv('.\CW_Data.csv')
# # 删除空行
# dataframe.drop(dataframe[dataframe['Programme'].isnull()].index,inplace=True)
# x = np.array(dataframe.drop(labels="Programme", axis=1))
# y = np.array(dataframe["Programme"])
#
# x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=100)
# # 假设符合分布，使用高斯贝叶斯进行计算
# clf = MultinomialNB()
# clf.fit(x_train,y_train)
# MultinomialNB()
# # 验证
# x_pred = clf.predict(x_test)
# acc = sum(x_pred == y_test) / x_pred.shape[0]
# print("准确率ACC: %.3f", acc)
# # 交叉验证
# scores = cross_val_score(clf, x, y, cv = 5)
# print(scores)
# print("多项式朴素贝叶斯预测的分类结果:",x_pred)

clf = GaussianNB(priors=None)
clf.fit(x_train,y_train)
GaussianNB(priors=None)
# 验证
x_pred = clf.predict(x_test)
acc = sum(x_pred == y_test) / x_pred.shape[0]
print("准确率ACC: %.3f", acc)
# 交叉验证
scores = cross_val_score(clf, x, y, cv = 5)
print(scores)
print("高斯朴素贝叶斯预测的分类结果:",x_pred)
# 将结果计数
programmecount0 = np.array(x_pred)
# count = programmecount0.value_counts()
programmecount0 = pd.Series(programmecount0)
count = programmecount0.value_counts()
print(count)

end = datetime.datetime.now()
runningtime = end - begin
print(runningtime)