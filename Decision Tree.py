import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn. discriminant_analysis import LinearDiscriminantAnalysis as LQAl
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import NMF
import seaborn as sns
from mpl_toolkits.mplot3d import  Axes3D
from  sklearn.tree import DecisionTreeClassifier
# # data extraction
dataframe = pd.read_csv('.\CW_Data.csv ', sep=',')
data = dataframe.values
# # print(dataframe)
# 观察数据之后发现ID一列没有实际意义删除
dataframe.drop(labels='ID',axis=1,inplace=True)
# #查找给定数组中的是否具有空值行并清除
# x0=np.isnan(data.data)
# print(np.any(x0))
# 删除空行
dataframe.drop(dataframe[dataframe['Programme'].isnull()].index,inplace=True)
# x1=np.isnan(data.data)
# print(dataframe.isna().sum())
# # print(data.shape)
# # print(dataframe)
# # print(data)
# # 观察数据之后发现Programme一列没有实际意义删除(做program的计数的时候需要把这一行注释掉)
# dataframe.drop(labels='Programme',axis=1,level=None,inplace=True,errors="raise")
# # dataframe.drop(labels='',axis=1,level=None,inplace=True,errors="raise")
# # 数据简化：删除重复的部分
#去掉重复数据保留后者
# dataframe.duplicate()
dataframe.drop_duplicates(subset=[ 'Q1', 'Q2', "Q3" , "Q4" , "Q5"],inplace = True)
# 决策树算法
x = np.array(dataframe.drop(labels="Programme", axis=1))
y = np.array(dataframe["Programme"])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2,random_state= 7)
# 建模，训练
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_predict = clf.predict(x_test)
# 交叉验证
scores = cross_val_score(clf, x, y, cv = 5).mean()
print(scores)
print(y_predict)
print("Acc:",clf.score(x_test,y_test))
# 将结果计数
programmecount0 = np.array(y_predict)
# count = programmecount0.value_counts()
programmecount0 = pd.Series(programmecount0)
count = programmecount0.value_counts()
print(count)
# 现在的问题：为什么设置了随机种子还是会随机