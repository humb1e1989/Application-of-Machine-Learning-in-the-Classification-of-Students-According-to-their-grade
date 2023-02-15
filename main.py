import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn. discriminant_analysis import LinearDiscriminantAnalysis as LQAl
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import NMF
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------------------------------------------------------------------------------------
# 通过pandas导入csv文件
# 数据处理：1.回归/异常值检测并清除 2.将missing data换成0 3.数据简化，删除相似/相同的data，清除不感兴趣的部分
#task1(现在还需要考虑的问题 task0要不要删掉)
# data extraction
dataframe = pd.read_csv('.\CW_Data.csv ', sep=',')
print(dataframe.shape)
# 观察数据之后发现ID一列没有实际意义删除
dataframe.drop(labels='ID',axis=1,inplace=True)
# dataframe[(dataframe.Programme == 0)].index.tolist()
# 删除空行
dataframe.drop(dataframe[dataframe['Programme'].isnull()].index,inplace=True)
# 考虑后删除program0
dataframe.drop(dataframe[(dataframe.Programme == 0)].index,inplace=True)
# 数据简化：删除重复的部分
#去掉重复数据保留后者
dataframe.drop_duplicates(subset=[ 'Q1', 'Q2', "Q3" , "Q4" , "Q5"],inplace = True)
data = dataframe.values

print(data.shape)
print(dataframe)
print(data)
# 输出的结果证明只有一行且是唯一一行有全部的Nan值
# dataframe.dropna(axis=0,how = 'all')
# dataframe.to_csv('finaldata.csv')


# ----------------------------------------------------------------------------------------------------------------------
# 统计每个programme的数量
# programme_count = dataframe['Programme'].value_counts()
# plt.figure()
# plt.title('Programme Count')
# for i in range(0, 5):
#     plt.bar(str(i), programme_count[i])
# plt.show()
# plt.savefig('INT 104 CW2 原始数据按照program分布图.jpg')      #保存图片
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 相关性矩阵
correlations = dataframe.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations,vmin = -1, vmax =1 )

labels ='Q1','Q3','Q4','Q2','Q5','Programme'
fig.colorbar(cax)
ticks = np.arange(0,6,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.show()

# ----------------------------------------------------------------------------------------------------------------------

# 观察数据之后发现Programme一列没有实际意义删除(做program的计数的时候需要把这一行注释掉)
dataframe.drop(labels='Programme',axis=1,level=None,inplace=True,errors="raise")

# print(data.shape)
# print(dataframe)
# print(data)
# print(type(dataframe))
# print(data.isna.sum())


# ----------------------------------------------------------------------------------------------------------------------
# 逐qs来观察mean,median，std,range
print("逐qs来观察mean,median，std,range")
print(dataframe.describe())
mean = dataframe.mean()
median = dataframe.median()
std = dataframe.std()
range1 = dataframe.max() - dataframe.min()
# print()
q = ["Q1","Q2","Q3","Q4","Q5"]
# q = [1,2,3,4,5,6]
plt.plot(q, mean , label="mean")
plt.plot(q, range1, label="range")
plt.plot(q, median , label="median")
plt.plot(q, std, label="std")
# plt.hist(dataframe,alpha = 0.7)
plt.title("basic data visualization")
plt.legend()
plt.show()
# 图P2中可以看出Q1和Q4的特征在各项数据的差别上会比较大，可能作为一个能够分别的feature
# ----------------------------------------------------------------------------------------------------------------------


# 逐渐programme开始观察数据
# 导入数据
# Program0 = []
Program1 = []
Program2 = []
Program3 = []
Program4 = []
for i in range (len(data)):
    if data[i,-1] == 1:
        Program1.append(data[i,:-1])
    elif data[i,-1] == 2:
        Program2.append(data[i,:-1])
    elif data[i,-1] == 3:
        Program3.append(data[i,:-1])
    elif data[i,-1] ==4:
        Program4.append(data[i,:-1])
    # elif data[i,-1] == 0:
    #     Program0.append(data[i,1:-1])
# print(Program1)
# ----------------------------------------------------------------------------------------------------------------------


# 对于mean/range/median做boxplot
# programme 0
# plt.figure()
# plt.title('Programme 0')
# data0 = np.array(Program0)
# label0 ='Q1','Q2','Q3','Q4','Q5'
# plt.boxplot(data0, labels=label0, capprops={'color': 'orange'})
# plt.xlabel("marks by Qs for Programme")
# plt.show()
#
# programme 1
plt.figure()
plt.title('Programme 1')
data_1 = np.array(Program1)
label1 = '1','2','3','4','5'
plt.boxplot(data_1,labels=label1,capprops={'color':'black'})
plt.xlabel("marks by Qs for Programme")
plt.show()

plt.figure()
plt.title('Programme 2')
data_2 = np.array(Program2)
label2 = '1','2','3','4','5'
plt.boxplot(data_2,labels=label2,capprops={'color':'c'})
plt.xlabel("marks by Qs for Programme")
plt.show()

plt.figure()
plt.title('Programme 3')
data_3 = np.array(Program3)
label3 = '1','2','3','4','5'
plt.boxplot(data_3,labels=label3,capprops={'color':'purple'})
plt.xlabel("marks by Qs for Programme")
plt.show()

plt.figure()
plt.title('Programme 4')
data_4 = np.array(Program4)
label4 = '1','2','3','4','5'
plt.boxplot(data_4,labels=label4,capprops={'color':'blue'})
plt.xlabel("marks by Qs for Programme")
plt.show()
# ----------------------------------------------------------------------------------------------------------------------

# PCA

plt.figure()
plt.title('PCA')
pca = KernelPCA(n_components=2, kernel='rbf')

dataset = np.concatenate([Program1,Program2,Program3,Program4],axis=0)
# 归一化
minMax = MinMaxScaler()
newpca = minMax.fit_transform(dataset)
scaler = StandardScaler()
dataset = scaler.fit_transform(newpca)
newpca = pca.fit_transform(newpca)


for i in range(len(Program1)):
    plt.scatter(newpca[i][0],newpca[i][1],alpha=0.5,c='black')
for i in range(len(Program1),len(Program1)+len(Program2),1):
    plt.scatter(newpca[i][0],newpca[i][1],alpha=0.5,c='blue')
for i in range(len(Program1)+len(Program2),len(Program1)+len(Program2)+len(Program3),1):
    plt.scatter(newpca[i][0],newpca[i][1],alpha=0.5,c='m')
for i in range(len(Program1)+len(Program2)+len(Program3),len(Program1)+len(Program2)+len(Program3)+len(Program4),1):
    plt.scatter(newpca[i][0],newpca[i][1],alpha=0.5,c='red')
# for i in range(len(Program0)+len(Program1)+len(Program2)+len(Program3),len(Program0)+len(Program1)+len(Program2)+len(Program3)+len(Program4),1):
#     plt.scatter(newpca[i][0],newpca[i][1],alpha=0.5,c='k')
plt.show()
# # 由于二维的PCA的分解数据看不太清，只能看出绿色和粉色的区别，其他颜色都杂糅在一起，所以我们二维的PCA分解意义不大
# # 所以采用三维的试一试
# #
# PCA 三维图
fig = plt.figure()
plt.title("3D PCA")
pca = PCA(n_components=3)
dataset = np.concatenate([Program1,Program2,Program3,Program4],axis=0)
scaler = StandardScaler()
dataset = scaler.fit_transform(dataset)
newpca = pca.fit_transform(dataset)
ax = Axes3D(fig)
for i in range(len(Program1)):
    ax.scatter(newpca[i][0],newpca[i][1],alpha=0.5,c='m')
for i in range(len(Program1),len(Program1)+len(Program2),1):
    ax.scatter(newpca[i][0],newpca[i][1],alpha=0.5,c='blue')
for i in range(len(Program1)+len(Program2),len(Program1)+len(Program2)+len(Program3),1):
    ax.scatter(newpca[i][0],newpca[i][1],alpha=0.5,c='c')
for i in range(len(Program1)+len(Program2)+len(Program3),len(Program1)+len(Program2)+len(Program3)+len(Program4),1):
    ax.scatter(newpca[i][0],newpca[i][1],alpha=0.5,c='g')
# for i in range(len(Program0)+len(Program1)+len(Program2)+len(Program3),len(Program0)+len(Program1)+len(Program2)+len(Program3)+len(Program4),1):
#     ax.scatter(newpca[i][0],newpca[i][1],alpha=0.5,c='k')
ax.set_xlabel("X",fontdict={'size': 15,'c':'b'})
ax.set_ylabel("Y",fontdict={'size': 15,'c':'b'})
ax.set_zlabel("Z",fontdict={'size': 15,'c':'b'})
plt.show()
# # ----------------------------------------------------------------------------------------------------------------------
# # LDA
# plt.figure()
# plt.title('LDA')
# lda =LDA(n_components=2)
#
#
# train_x=np.concatenate([Program0,Program1,Program2,Program3,Program4],axis=0)
# scaler = StandardScaler()
# train_x = scaler.fit_transform(train_x)
# Program0_y=np.zeros(len(Program0))+0
# Program1_y=np.zeros(len(Program1))+1
# Program2_y=np.zeros(len(Program2))+2
# Program3_y=np.zeros(len(Program3))+3
# Program4_y=np.zeros(len(Program4))+4
# train_y=np.concatenate([Program0_y,Program1_y,Program2_y,Program3_y,Program4_y],axis=0)
# afterlda=lda.fit_transform(train_x,train_y)
# for i in range(len(Program0)):
#     plt.scatter(afterlda[i][0],afterlda[i][1],alpha=0.5,c='blue')
# for i in range(len(Program0),len(Program0)+len(Program1),1):
#     plt.scatter(afterlda[i][0],afterlda[i][1],alpha=0.5,c='c')
# for i in range(len(Program0)+len(Program1),len(Program0)+len(Program1)+len(Program2),1):
#     plt.scatter(afterlda[i][0],afterlda[i][1],alpha=0.5,c='g')
# for i in range(len(Program0)+len(Program1)+len(Program2),len(Program0)+len(Program1)+len(Program2)+len(Program3),1):
#     plt.scatter(afterlda[i][0],afterlda[i][1],alpha=0.5,c='k')
# for i in range(len(Program0)+len(Program1)+len(Program2)+len(Program3),len(Program0)+len(Program1)+len(Program2)+len(Program3)+len(Program4),1):
#     plt.scatter(afterlda[i][0],afterlda[i][1],alpha=0.5,c='r')
# plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# t-SNE
plt.figure()
plt.title('t-SNE')
tsne =TSNE(n_components=2)


dataset=np.concatenate([Program1,Program2,Program3,Program4],axis=0)
scaler = StandardScaler()
dataset = scaler.fit_transform(dataset)
tsne.fit_transform(dataset)
newdataset = tsne.embedding_
for i in range(len(Program1)):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5,c='blue')
for i in range(len(Program1),len(Program1)+len(Program2),1):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5,c='c')
for i in range(len(Program1)+len(Program2),len(Program1)+len(Program2)+len(Program3),1):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5,c='g')
for i in range(len(Program1)+len(Program2)+len(Program3),len(Program1)+len(Program2)+len(Program3)+len(Program4),1):
    plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5,c='k')
# for i in range(len(Program0)+len(Program1)+len(Program2)+len(Program3),len(Program0)+len(Program1)+len(Program2)+len(Program3)+len(Program4),1):
#     plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5,c='m')
plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# t-SNE 3D


# ----------------------------------------------------------------------------------------------------------------------

# NMF
# plt.figure()
# plt.title('NMF')
# # init中的“”字符串可以改变去改变NMF的结果和方法
# nmf = NMF(n_components = 2 ,init= "nndsvd")
# dataset = np.concatenate([Program0,Program1,Program2,Program3,Program4],axis=0)
# scaler = MinMaxScaler()
# dataset = scaler.fit_transform(dataset)
# # print(dataset)
# newnmf = nmf.fit_transform(dataset)
# # print(newnmf)
# for i in range(len(Program0)):
#     plt.scatter(newnmf[i][0],newnmf[i][1],alpha=0.5,c='m')
# for i in range(len(Program0),len(Program0)+len(Program1),1):
#     plt.scatter(newnmf [i][0],newnmf [i][1],alpha=0.5,c='blue')
# for i in range(len(Program0)+len(Program1),len(Program0)+len(Program1)+len(Program2),1):
#     plt.scatter(newnmf [i][0],newnmf [i][1],alpha=0.5,c='c')
# for i in range(len(Program0)+len(Program1)+len(Program2),len(Program0)+len(Program1)+len(Program2)+len(Program3),1):
#     plt.scatter(newnmf [i][0],newnmf [i][1],alpha=0.5,c='g')
# for i in range(len(Program0)+len(Program1)+len(Program2)+len(Program3),len(Program0)+len(Program1)+len(Program2)+len(Program3)+len(Program4),1):
#     plt.scatter(newnmf [i][0],newnmf [i][1],alpha=0.5,c='k')
# plt.show()
# # ----------------------------------------------------------------------------------------------------------------------
# 拼接结果
# testnmf = np.array(newnmf)
testtsne = np.array(newdataset)
testpca = np.array(newpca)
# newdata = np.array(dataframe['Q5'],axis=1)
dataframe.drop(labels='Q1',axis=1,level=None,inplace=True,errors="raise")
dataframe.drop(labels='Q2',axis=1,level=None,inplace=True,errors="raise")
dataframe.drop(labels='Q3',axis=1,level=None,inplace=True,errors="raise")
dataframe.drop(labels='Q4',axis=1,level=None,inplace=True,errors="raise")
dataframe = np.array(dataframe)
print(dataframe.shape)
print(testpca.shape)
print(testtsne.shape)
print(dataframe)
data_final = np.concatenate((dataframe,testpca,testtsne),axis= 1)
# data_final = np.array(data_final)
# print(data_final)
data_final = pd.DataFrame(data_final)
# data_final.to_csv("finaldata1.csv")
print(data_final)



# def get_NMF(dataframe,steps=2000,alpha=0.0002,beta=0.02):
#     N=len(dataframe)
#     M =len(dataframe[0])
#     K = 2
#     P = np.random. rand(N,K)
#     Q= np.random.rand(M,K)
#     Q=Q.T
#     for step in range(steps):
#         for i in range(len(dataframe)):
#             for j in range(len(dataframe[i])):
#                 if dataframe[i][j] >0:
#                     eij = dataframe[i][j] - np.dot(P[i,:], Q[:, j])
#                     for k in range(K):
#                         P[i][k] =P[i][k] + alpha * (2 *eij * Q[k][j] - beta * P[i][k])
#                         dataframe[k][i] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
#         e = 0
#         for i in range(len(dataframe)):
#             for j in range(len(dataframe[j])):
#                 if dataframe[i][j] >0:
#                     e= e+ pow(dataframe[i][j] - np.dot(P[i,:],Q[:,j]),2)
#                     for k in range(K):
#                         e = e + (beta / 2) * (pow(P[i][k],2) + pow(Q[k][j],2))
#         if e< 0.001:
#             break
#     print(P)
# print(" ---------------------------------")
# print(Q.T)
# return P

# plt.figur
# 创建一个model，注意NMF必须规定降到的维数 n_components ，但是PCA不必要
# model = NMF(n_components = 2)
# 训练模型
# model.fit(dataframe)
# 降维
# nmf_features = model.transform(data)
# print(nmf_features)
# print(Program0)


# task1 结束 找出feature就行
# task2开始



# range = []
# for i in range[0,5]:
#     range.append(data['0'+str(i)].max() - data['0'+str(i)].min())
# 开始画图
# plt.figure()
# plt.title( 'basic data')
# plt.bar([ '0','1','2','3','4'],
# plt.show()
# dataframe.drop(labels='Programme',axis=1,level=None,inplace=False,errors="raise")



# programme_count = dataframe['Programme'].value_counts()
# plt.figure()
# plt.title('Programme Count')
# for i in range(0, 6):
#     plt.bar(str(i), programme_count[i])
# plt.show()
# plt.bar(["Programme0.0","Programme1.0","Programme2.0","Programme3.0","Programme4.0"],np.mean(dataframe, axis=0),label='meanbyprograme');

# program classification
# 逐programme的分析每个program 的不同的feature
# dataframe.drop(labels='Programme',axis=1,level=None,inplace=False,errors="raise")

# range() = dataframe.max() - dataframe.min()
# meanofpro = dataframe.groupby("Programme").mean()
# medianofpro = dataframe.groupby("Programme").median()
# # rangeofpro = dataframe.groupby("Programme").range()
# stdofpro = dataframe.groupby("Programme").std()
# # # sns.barplot(x="Programme", y='Q1', data=meanofpro, color= "darkblue",label = "Q1")
# print("mean by programme:")
# print(meanofpro)

# sns.set(style= "whitegrid")
# plt.bar(["Q1","Q2", "Q3", "Q40", "Q5"],[15.833333,0.25000,4.500000,0.66667,2.583333])
# plt.ylabel("mean")
# plt.title("Mean of Programme0.0")
# # plt.legend()
# plt.show()
# print("median by programe:")
# print(medianofpro)
# sns.set(style= "whitegrid")
# plt.bar(["Q1","Q2", "Q3", "Q40", "Q5"],[14.0, 0.0,2.0,0.0,5.0],color = "pink")
# plt.bar(["Q8","Q2", "Q3", "Q40", "Q5"],[7.0, 70.0,20.0,0.7,0.5],color = "black")
#
# plt.ylabel("mean")
# plt.title("Mean of Programme0.0")
# # plt.legend()
# plt.show()
# print("range by programme:")
#
# print("std by programme:")
# print(stdofpro)
# [9.437193,0.452267,4.908249,0.984732,3.369685]

# 观察之后能得出q4 和 q5的区分性比较大(Q4，Q5，作为feartures)
# 继续观察，然后手动输入得到直方图
# 考虑到直方图可能存在覆盖/不好对比的情况下，用散点图画
# sns.barplot(x = "Programme", y = "meanogpro", data= data)
# mean
#散点图：
# sns.boxplot(x="Programme0.0", y="Q1", data=meanofpro)


# p = ["Programme0.0", "Programme1.0", "Programme2.0","Programme3.0", "Programme4.0"]
# plt.plot(p, mean , label="mean")
# plt.plot(q, range, label="range")
# plt.plot(p, median , label="median")
# plt.plot(p, std, label="std")

# 逐program的将每个专业的数组录进去预设组里面 计算PCA 和 TSNE的

# print(type(data))
# l = len(data)
# for i in range(l):
#    if data[i,-1]== 0:
#        Program0.append(data[i,-1])
#    elif data[i,-1] == 1:
#        Program1.append(data[i,1:-1])
#    elif data[i,-1] == 2:
#        Program2.append(data[i,1:-1])
#    elif data[i,-1] == 3:
#        Program3.append(data[i,1:-1])
#    elif data[i,-1]==4:
#        Program4.append(data[i,1:-1])
# p=["Programme0.0" , "Programme1.0", "Programme2.0", "Programme3.0","Programme4.0"]
# plt.plot(mean,p, Label= "meanGroupbyProgramme")
# plt.legend()
# plt.show()
#  #      print(df.decrible)
#
# # data visulization
# # bar graph

# plt.figure()
# plt.title( 'data number')
# plt.bar([ '0','1','2','3','4'],[len(Program0),len(Program1), len(Program2),len(Program3),len(Program4)])
# plt.show()

# #t-SNE
# #####
# plt.figure()
# plt.title('t-SNE ' )
# tsne = TSNE(n_components=2)#determine dimension
# dataset = np.concatenate([Program0,Program1,Program2,Program3,Program4],axis=0)
# scaler = StandardScaler()
# dataset = scaler.fit_transform(dataset)
# tsne.fit_transform(dataset)
# newdataset = tsne.embedding_
# for i in range(len(Program0)):
#     plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5,c='blue')
# for i in range(len(Program0),len(Program0)+len(Program1),1):
#     plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5,c='C')
# for i in range(len(Program0)+len(Program1),len(Program0)+len(Program1)+len(Program2),1):
#     plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5,c='g')
# for i in range(len(Program0)+len(Program1)+len(Program2),len(Program0)+len(Program1)+len(Program2)+len(Program3),1):
#     plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5,c='k')
# for i in range(len(Program0)+len(Program1)+len(Program2)+len(Program3),len(Program0)+len(Program1)+len(Program2)+len(Program3)+len(Program4),1):
#     plt.scatter(newdataset[i][0],newdataset[i][1],alpha=0.5,C='m')
# plt.show()

#散点图
# plt.scatter(data.index,data,color = 'k ' , marker=' . ' ,alpha = 0.3)
# plt.scatter(error.index,error, color = 'r ',marker='. ' ,alpha = 0.5)
# plt.xlim([-10,10010])
# plt.grid()

