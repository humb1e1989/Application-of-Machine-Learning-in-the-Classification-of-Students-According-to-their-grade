from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

import tensorflow123 as tf

from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LQAl
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import NMF
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

dataframe = pd.read_csv('.\CW_Data.csv ', sep=',')
data = dataframe.values
# 观察数据之后发现ID一列没有实际意义删除
dataframe.drop(labels='ID', axis=1, inplace=True)
# 删除空行
dataframe.drop(dataframe[dataframe['Programme'].isnull()].index, inplace=True)
# # 删除重复
dataframe.drop_duplicates(subset=[ 'Q1', 'Q2', "Q3", "Q4" , "Q5"],inplace = True)

x = np.array(dataframe.drop(labels="Programme", axis=1))
truelabel = np.array(dataframe["Programme"])

# 找到最佳的K值，先用轮廓系数和SSE确定K 然后用肘方法和NMI验证

# K means
KMs = KMeans(init='k-means++', n_clusters=4, random_state=41).fit(x)
x_pred = KMs.predict(x)
y_pred = KMs.labels_
print(x_pred)
# 将结果计数
programmecount0 = np.array(x_pred)
# count = programmecount0.value_counts()
programmecount0 = pd.Series(programmecount0)
count = programmecount0.value_counts()
print(count)
# K-means聚类
from sklearn.cluster import KMeans

# -----------------------------------------------------------------------------------------------------------
# 将PCA作为坐标系进行可视化
# pca2 = PCA(n_components=2)
# pca2.fit(x)
# X2 = pca2.fit_transform(x)
# estimator = KMeans(n_clusters=5)
# y_predicted = estimator.fit_predict(X2)
# print(y_predicted)
# -----------------------------------------------------------------------------------------------------------
# t-SNE 作为坐标系进行可视化
tsne =TSNE(n_components=2)
X2 = tsne.fit_transform(x)
newdataset = tsne.embedding_
estimator=KMeans(n_clusters=4)
y_predicted=estimator.fit_predict(x)
# 画出簇心
centroid = estimator.cluster_centers_
# print(centroid)
plt.scatter(centroid[:, 0], centroid[:, 1], marker='o', s=260, linewidths=3, color='pink', label='centroid')
# 聚类结果
plt.title('Kmeans clustering with t-SNE')
plt.xlabel('Silhouettes Score: 0.2741')

plt.scatter(X2[:, 0], X2[:, 1], marker='*', c=x_pred)
plt.show()
# -----------------------------------------------------------------------------------------
# 检测准确率
NMI_scores = normalized_mutual_info_score(truelabel, y_pred)
# 可视化，尝试找到最适合的K值
# range = np.array(2,10)
nmi_scores = []
# 存放设置不同簇数时的SSE值
sse_list = []
# 轮廓系数
silhouettes = []
range = np.arange(2, 15, 1)
for i in range:
    kmeans = KMeans(init='k-means++', n_clusters=i, random_state=2)
    kmeans.fit(x)
    y_prednew = kmeans.labels_
    nmi_score = normalized_mutual_info_score(truelabel, y_prednew)
    nmi_scores.append(nmi_score)
    sse_list.append(kmeans.inertia_)
    silhouette = metrics.silhouette_score(x, y_prednew, metric='euclidean')
    silhouettes.append(silhouette)


plt.figure()
# subplot()
plt.plot(range, silhouettes, marker='*', c='orange')
plt.title('Silhouettes Score')
plt.xlabel('Cluster')
plt.ylabel('Silhouettes Score')
plt.show()

plt.figure()
plt.plot(range, sse_list, marker='x', c='orange')
plt.title('SSE Score')
plt.xlabel('Cluster')
plt.ylabel('SSE Score')
plt.show()

plt.figure()
plt.plot(range, nmi_scores, marker='o', c='orange')
plt.title('NMI Score')
plt.xlabel('Cluster')
plt.ylabel('NMI Score')
plt.show()

# 轮廓系数
# 自动选择欧式距离
score = metrics.silhouette_score(x, y_pred)
print(score)
# # NMI检测
# import math
# from sklearn import metrics
#
# def NMI(A,B):
#     # A = x_pred
#     # B = np.array(dataframe["Programme"])
#     A = A.tolist()
#     B = B.tolist()
#     print(A)
#     print(B)
#     # 样本点数
#     total = len(A)
#     B_ids = set(B)
#     A_ids = set(A)
#
#     # 互信息计算
#     MI = 0
#     eps = 1.4e-5
#     for idA in A_ids:
#         for idB in B_ids:
#             idAoccur = np.where(A == idA)
#             idBoccur = np.where(B == idB)
#             idABOccur = np.intersect1d(idAoccur, idBoccur)
#             px = 1.0 * len(idAoccur[0]) / total
#             py = 1.0 * len(idBoccur[0]) / total
#             pxy = 1.0 * len(idABOccur) / total
#             MI = MI + pxy * math.log(pxy / (px * py + eps) + eps, 2)
#     # 标准化互信息
#     Hx = 0
#     for idA in A_ids:
#         idAoccurcount = 1.0 * len(np.where(A == idA)[0])
#         Hx = Hx - (idAoccurcount / total) * math.log(idAoccurcount / total + eps, 2)
#     Hy = 0
#     for idB in B_ids:
#         idBoccurcount = 1.0 * len(np.where(B == idB)[0])
#     Hy = Hy - (idBoccurcount / total) * math.log(idBoccurcount / total + eps, 2)
#     MIhat = 2.0 * MI / (Hx + Hy +eps)
#     return MIhat
# # if __name__ == 'K MEANS':
# if __name__ == '__main__':
#     A = x_pred
#     B = np.array(dataframe["Programme"])
#     A = A.tolist()
#     B = B.tolist()
#     print(NMI(A,B))
#     print(metrics.normalized_mutual_info_score(A,B))
# # print(NMI(x_pred,x))


# if __name__ == '__main__':
