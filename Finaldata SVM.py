import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pylab as plt
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn import svm
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

# 分类
clf = OneVsRestClassifier(SVC(C = 13,kernel = 'rbf', decision_function_shape = 'ovr',degree=5,gamma= 'scale'))
# 得到训练集合和验证集
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=3)
# # # 检测C值在什么时候最好(使用网格搜索)
testSVM = svm.SVC(decision_function_shape='ovr')
# # parameter = dataframe
param_dict={"C":[i for i in range(1,100)]}
# param_dict={"gamma":[i for i in range{'scale', 'auto'}]}
Gridsearch = GridSearchCV(testSVM,cv = 5,scoring= 'accuracy',param_grid=param_dict)
Gridsearch.fit(x_train,y_train)
print("最佳参数：\n",Gridsearch.best_params_)
print("最佳结果：\n",Gridsearch.best_score_)
# 评估
clf.fit(x,y)
y_predict = clf.predict(x)
clf.fit(x_train, y_train)
x_pred = clf.predict(x_test)
acc = sum(x_pred == y_test) / x_pred.shape[0]
print("准确率ACC:", acc)
# 交叉验证
scores = cross_val_score(clf, x_train, y_train, cv = 5,scoring = 'accuracy')
scoresmean = cross_val_score(clf, x_train, y_train, cv = 5,scoring = 'accuracy').mean()
scoresstd = cross_val_score(clf, x_train, y_train, cv = 5,scoring = 'accuracy').std()
usedscore = scoresstd/scoresmean
print(x_pred)
f1_scores = f1_score(y, y_predict, average='weighted')

print(scores)
print(scoresmean)
print(scoresstd)
print(usedscore)
print(f1_scores)
# 将结果计数
programmecount0 = np.array(x_pred)
# count = programmecount0.value_counts()
programmecount0 = pd.Series(programmecount0)
count = programmecount0.value_counts()
print(count)

end = datetime.datetime.now()
runningtime = end - begin
print(runningtime)