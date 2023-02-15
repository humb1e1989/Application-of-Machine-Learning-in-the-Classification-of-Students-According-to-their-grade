import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn. discriminant_analysis import LinearDiscriminantAnalysis as LQAl
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import NMF
import seaborn as sns
from mpl_toolkits.mplot3d import  Axes3D
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import  datetime

# # data extraction
# dataframe = pd.read_csv('.\CW_Data.csv ', sep=',')
# data = dataframe.values
# # 观察数据之后发现ID一列没有实际意义删除
# dataframe.drop(labels='ID',axis=1,inplace=True)
# # 删除空行
# dataframe.drop(dataframe[dataframe['Programme'].isnull()].index,inplace=True)
# x = np.array(dataframe.drop(labels="Programme", axis=1))
# y = np.array(dataframe["Programme"])
#
# # 得到训练集合和验证集
# x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=3)
# # 分类
# # clf = make_pipeline(StandardScaler(),SVC(gamma='auto'))
# clf = svm.SVC(C = 3,kernel = 'rbf', decision_function_shape = 'ovr',degree=5)
# clf.fit(x_train, y_train)
# x_pred = clf.predict(x_test)
# acc = sum(x_pred == y_test) / x_pred.shape[0]
# print("准确率ACC:", acc)
# # 交叉验证
# scores = cross_val_score(clf, x, y, cv = 5).mean()
# print(scores)
# print(x_pred)
# 重新定义SVC下面的参数可以自己设置
# 见下面的classifier
# 开始时间
begin = datetime.datetime.now()
# data extraction
dataframe = pd.read_csv('.\CW_Data.csv ', sep=',')
data = dataframe.values
# 观察数据之后发现ID一列没有实际意义删除
dataframe.drop(labels='ID',axis=1,inplace=True)
# 删除空行
dataframe.drop(dataframe[dataframe['Programme'].isnull()].index,inplace=True)
# 考虑后删除program0
dataframe.drop(dataframe[(dataframe.Programme == 0)].index,inplace=True)
x = np.array(dataframe.drop(labels="Programme", axis=1))
y = np.array(dataframe["Programme"])

# 归一化
minMax = MinMaxScaler()
x = minMax.fit_transform(x)

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
# clf = make_pipeline(StandardScaler(),SVC(gamma='auto'))
# 分类
clf = OneVsRestClassifier(SVC(C = 13,kernel = 'rbf', decision_function_shape = 'ovr',degree=5,gamma= 'scale'))
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
# print(usedscore)
# 将结果计数
programmecount0 = np.array(x_pred)
# count = programmecount0.value_counts()
programmecount0 = pd.Series(programmecount0)
count = programmecount0.value_counts()
print(count)

cvscorelist = []
cvalues = []
for cvalue in range(1,20,1):
    cvalues.append(cvalue)
    choosec = SVC(C=cvalue,kernel='rbf',gamma='scale',decision_function_shape='ovr')
    choosecvs = cross_val_score(choosec,x_train,y_train,scoring = 'accuracy',cv = 5)
    cvscorelist.append(choosecvs.mean())
plt.plot(cvalues,cvscorelist)
plt.xlabel('Values of c')
plt.ylabel('Accurate rate')
plt.show()

end = datetime.datetime.now()
runningtime = end - begin
print(runningtime)
