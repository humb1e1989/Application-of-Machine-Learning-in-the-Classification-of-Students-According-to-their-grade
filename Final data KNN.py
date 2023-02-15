# KNN
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pylab as plt
from sklearn.metrics import f1_score
import  datetime
# 开始时间
begin = datetime.datetime.now()
# 导入数据集
dataframe0 = pd.read_csv('.\CW_Data.csv')
dataframe = pd.read_csv('finaldata1.csv')
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
# 归一化
minMax = MinMaxScaler()
x = minMax.fit_transform(x)
# y = minMax.fit_transform(y)
print(x.shape)
print(y.shape)
# 得到训练集合和验证集
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=77)
# 检测K值在什么时候最好(使用网格搜索)
testKNN = KNeighborsClassifier(p = 2)
# # parameter = dataframe
param_dict={"n_neighbors":[i for i in range(3,25)]}
Gridsearch = GridSearchCV(testKNN,cv = 5,scoring= 'accuracy',param_grid=param_dict)
Gridsearch.fit(x_train,y_train)
print("最佳参数：\n",Gridsearch.best_params_)
print("最佳结果：\n",Gridsearch.best_score_)
# size可分 什么size 都可以 只是要确定比例就好
# 训练模型（定义）
clf = KNeighborsClassifier(n_neighbors= 11, p = 2)
# 使用欧氏距离（距离会影响什么？）
clf.fit(x_train, y_train)
KNeighborsClassifier()
# 评估
y_predict = clf.predict(x)
# 循环做平均就是交叉验证
x_pred = clf.predict(x_test)
acc = sum(x_pred == y_test) / x_pred.shape[0]
print("准确率ACC:", acc)
# 交叉验证
scoresmean = cross_val_score(clf, x, y, cv = 5,scoring = 'accuracy').mean()
scores = cross_val_score(clf, x, y, cv = 5,scoring = 'accuracy')
scoresstd = cross_val_score(clf, x, y, cv = 5,scoring = 'accuracy').std()
usedscore = scoresstd/scoresmean
# F1
# 这里需要注意，如果是二分类问题则选择参数‘binary’；
# 如果考虑类别的不平衡性，需要计算类别的加权平均，则使用‘weighted’；
# 如果不考虑类别的不平衡性，计算宏平均，则使用‘macro’。：
f1_scores = f1_score(y, y_predict, average='weighted')

print(scores)
print(scoresmean)
print(scoresstd)
print(usedscore)
print(f1_scores)

print(x_pred)
print(x_pred.shape)
# print(y_test)
# y_proba = clf.predict_proba(x_test[:1])
# print(clf.predict(x_test[:1]))
# print('预计的概率：',y_proba)
# 将结果计数
programmecount0 = np.array(x_pred)
# count = programmecount0.value_counts()
programmecount0 = pd.Series(programmecount0)
count = programmecount0.value_counts()
print(count)
# 准确率在归一化之后波动减少,一直没有出现4和0证明用Q1/Q2/Q3/Q4/Q5作为feature不能很好的用KNN将其分类 从print的数据也可以看出，1和2 有50，3只有3个数据 和原始数据相比太过于少了 少了太多的的数据了
# 检测K值在什么时候最好(可视化方法)
neig = np.arange(1, 25, 1)
train_accuracy = []
test_accuracy = []
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='auto')
    # Fit with knn
    knn.fit(x_train,y_train)
    # #train accuracy
    # train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    testscores = cross_val_score(knn, x_train, y_train, cv = 5,scoring = 'accuracy').mean()
    test_accuracy.append(testscores)
# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
# plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.show()

end = datetime.datetime.now()
runningtime = end - begin
print(runningtime)