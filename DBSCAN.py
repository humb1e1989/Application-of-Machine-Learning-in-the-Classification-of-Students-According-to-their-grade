from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
# 数据预处理
# 读取文件
from sklearn.preprocessing import StandardScaler

dataframe = pd.read_csv('.\CW_Data.csv ', sep=',')
data = dataframe.values
# dataframe.duplicate()
dataframe.drop_duplicates(subset=[ 'Q1', 'Q2', "Q3" , "Q4" , "Q5"],inplace = True)
# 删除空行
dataframe.drop(dataframe[dataframe['Programme'].isnull()].index,inplace=True)
# 使用原始数据
x = np.array(dataframe.drop(labels="Programme", axis=1))
y = np.array(dataframe["Programme"])
# 观察数据之后发现Programme一列没有实际意义删除(做program的计数的时候需要把这一行注释掉)
dataframe.drop(labels='Programme',axis=1,level=None,inplace=True,errors="raise")

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=2.5, min_samples=2).fit(x)
# DBSCAN
x_pred = db.fit_predict(x)
print(x_pred)
# 将结果计数
programmecount0 = np.array(x_pred)
# count = programmecount0.value_counts()
programmecount0 = pd.Series(programmecount0)
count = programmecount0.value_counts()
print(count)
# 可视化 PCA
# pca2=PCA(n_components=2)
# pca2.fit(x)
# X2=pca2.fit_transform(x)
tsne =TSNE(n_components=2)
scaler = StandardScaler()
X2 = TSNE.fit_transform(x)
X2 = tsne.embedding_
estimator=DBSCAN(eps=2.5, min_samples=2)
y_predicted=estimator.fit_predict(X2)
print(y_predicted)
plt.scatter(X2[:,0],X2[:,1],marker='*',c=y_predicted)
plt.show()
# 将结果计数
programmecount0 = np.array(x_pred)
# count = programmecount0.value_counts()
programmecount0 = pd.Series(programmecount0)
count = programmecount0.value_counts()