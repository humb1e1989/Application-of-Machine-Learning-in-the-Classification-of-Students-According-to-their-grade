import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn. discriminant_analysis import LinearDiscriminantAnalysis as LQAl
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import NMF
import seaborn as sns
from mpl_toolkits.mplot3d import  Axes3D
# data extraction
dataframe = pd.read_csv('.\CW_Data.csv ', sep=',')
data = dataframe.values
# print(dataframe)
# 观察数据之后发现ID一列没有实际意义删除
dataframe.drop(labels='ID',axis=1,inplace=True)
#查找给定数组中的是否具有空值行并清除
x=np.isnan(data.data)
print(np.any(x))
# 删除空行
dataframe.drop(dataframe[dataframe['Programme'].isnull()].index,inplace=True)
x1=np.isnan(data.data)
print(dataframe.isna().sum())
# print(data.shape)
# print(dataframe)
# print(data)
# 观察数据之后发现Programme一列没有实际意义删除(做program的计数的时候需要把这一行注释掉)
dataframe.drop(labels='Programme',axis=1,level=None,inplace=True,errors="raise")
# dataframe.drop(labels='',axis=1,level=None,inplace=True,errors="raise")
# 数据简化：删除重复的部分
#去掉重复数据保留后者
# dataframe.duplicate()
dataframe.drop_duplicates(subset=[ 'Q1', 'Q2', "Q3" , "Q4" , "Q5"],inplace = True)

print(data.shape)
print(dataframe)
print(data)
