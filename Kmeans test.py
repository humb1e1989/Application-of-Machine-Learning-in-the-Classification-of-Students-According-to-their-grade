from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, calinski_harabasz_score, silhouette_score


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
# 观察数据之后发现ID一列没有实际意义删除
dataframe.drop(labels='ID', axis=1, inplace=True)
# 删除空行
dataframe.drop(dataframe[dataframe['Programme'].isnull()].index, inplace=True)
# # 删除重复
dataframe.drop_duplicates(subset=[ 'Q1', 'Q2', "Q3", "Q4" , "Q5"],inplace = True)

x = np.array(dataframe.drop(labels="Programme", axis=1))
truelabel = np.array(dataframe["Programme"])
data = dataframe.values
# 找到最佳的K值，先用轮廓系数和SSE确定K 然后用肘方法和NMI验证

# K means
KMs = KMeans(init='k-means++', n_clusters=4, random_state=41).fit(x)
x_pred = KMs.predict(x)
y_pred = KMs.labels_
print(x_pred)
# t-SNE 作为坐标系进行可视化
tsne =TSNE(n_components=2)
X2 = tsne.fit_transform(x)
newdataset = tsne.embedding_
estimator=KMeans(n_clusters=4)
y_predicted=estimator.fit_predict(X2)