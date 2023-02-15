from sklearn import datasets

from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
# 数据预处理
# 读取文件
dataframe = pd.read_csv('.\CW_Data.csv ', sep=',')
data = dataframe.values
# dataframe.duplicate()
dataframe.drop_duplicates(subset=[ 'Q1', 'Q2', "Q3" , "Q4" , "Q5"],inplace = True)
# 删除空行
dataframe.drop(dataframe[dataframe['Programme'].isnull()].index,inplace=True)
x = np.array(dataframe.drop(labels="Programme", axis=1))
y = np.array(dataframe["Programme"])
# 观察数据之后发现Programme一列没有实际意义删除(做program的计数的时候需要把这一行注释掉)
dataframe.drop(labels='Programme',axis=1,level=None,inplace=True,errors="raise")


# # AGNES聚类
clustering = AgglomerativeClustering(linkage='ward', n_clusters=5)
res = clustering.fit(dataframe)
print ("各个簇的样本数目：")
print (pd.Series(clustering.labels_).value_counts())
print ("聚类结果：")
print (confusion_matrix(y, clustering.labels_))

#
# d0 = dataframe[clustering.labels_ == 0]
# plt.plot(d0[:, 0], d0[:, 1], 'r.')
# d1 = dataframe[clustering.labels_ == 1]
# plt.plot(d1[:, 0], d1[:, 1], 'go')
# d2 = dataframe[clustering.labels_ == 2]
# plt.plot(d2[:, 0], d2[:, 1], 'b*')
# d3 = dataframe[clustering.labels_ == 3]
# plt.plot(d0[:, 0], d0[:, 1], 'r.')
# d4 = dataframe[clustering.labels_ == 4]
# plt.plot(d0[:, 0], d0[:, 1], 'r.')
#
# plt.xlabel("Sepal.Length")
#
# plt.ylabel("Sepal.Width")
#
# plt.title("AGNES Clustering")
#
# plt.show()

#   File "pandas\_libs\index.pyx", line 82, in pandas._libs.index.IndexEngine.get_loc
# TypeError: '(slice(None, None, None), 0)' is an invalid key(同KMeans聚类画图的结果)
# 画图
plt.figure()
pca2=PCA(n_components=2)
pca2.fit(x)
X2=pca2.fit_transform(x)
estimator=AgglomerativeClustering(n_clusters=5)
y_predicted=estimator.fit_predict(X2)
print(y_predicted)
plt.scatter(X2[:,0],X2[:,1],marker='*',c=y_predicted)
plt.show()
