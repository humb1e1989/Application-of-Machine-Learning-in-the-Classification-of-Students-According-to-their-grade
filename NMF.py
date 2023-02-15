import pandas as pd
import matplotlib.pylab as plt
import numpy as np
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

# 数据简化：删除重复的部分
#去掉重复数据保留后者
# dataframe.duplicate()
dataframe.drop_duplicates(subset=[ 'Q1', 'Q2', "Q3" , "Q4" , "Q5"],inplace = True)
# 观察数据之后发现Programme一列没有实际意义删除(做program的计数的时候需要把这一行注释掉)
dataframe.drop(labels='Programme',axis=1,level=None,inplace=True,errors="raise")
# 导入数据
Program0 = []
Program1 = []
Program2 = []
Program3 = []
Program4 = []
for i in range (len(data)):
    if data[i,-1] == 0:
        Program0.append(data[i,1:-1])
    elif data[i,-1] == 1:
        Program1.append(data[i,1:-1])
    elif data[i,-1] ==2:
        Program2.append(data[i,1:-1])
    elif data[i,-1] ==3:
        Program3.append(data[i,1:-1])
    elif data[i,-1]==4:
        Program4.append(data[i,1:-1])
# ----------------------------------------------------------------------------------------------------------------------
# NMF
plt.figure()
plt.title('NMF')
# init中的“”字符串可以改变去改变NMF的结果和方法
nmf = NMF(n_components = 2 ,init= "nndsvd")
dataset = np.concatenate([Program0,Program1,Program2,Program3,Program4],axis=0)
scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)
# print(dataset)
newnmf = nmf.fit_transform(dataset)
# print(newnmf)
for i in range(len(Program0)):
    plt.scatter(newnmf[i][0],newnmf[i][1],alpha=0.5,c='m')
for i in range(len(Program0),len(Program0)+len(Program1),1):
    plt.scatter(newnmf [i][0],newnmf [i][1],alpha=0.5,c='blue')
for i in range(len(Program0)+len(Program1),len(Program0)+len(Program1)+len(Program2),1):
    plt.scatter(newnmf [i][0],newnmf [i][1],alpha=0.5,c='c')
for i in range(len(Program0)+len(Program1)+len(Program2),len(Program0)+len(Program1)+len(Program2)+len(Program3),1):
    plt.scatter(newnmf [i][0],newnmf [i][1],alpha=0.5,c='g')
for i in range(len(Program0)+len(Program1)+len(Program2)+len(Program3),len(Program0)+len(Program1)+len(Program2)+len(Program3)+len(Program4),1):
    plt.scatter(newnmf [i][0],newnmf [i][1],alpha=0.5,c='k')
plt.show()