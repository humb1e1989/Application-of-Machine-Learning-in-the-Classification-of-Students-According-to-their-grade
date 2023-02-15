import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn. discriminant_analysis import LinearDiscriminantAnalysis as LQAl
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, validation_curve
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


# data extraction
dataframe = pd.read_csv('.\CW_Data.csv ', sep=',')
data = dataframe.values
# 观察数据之后发现ID一列没有实际意义删除
dataframe.drop(labels='ID',axis=1,inplace=True)
# 删除空行
dataframe.drop(dataframe[dataframe['Programme'].isnull()].index,inplace=True)

x = np.array(dataframe.drop(labels="Programme", axis=1))
y = np.array(dataframe["Programme"])

# 得到训练集合和验证集
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=3)
# 检测C值在什么时候最好(使用网格搜索)
testSVM = svm.SVC(decision_function_shape='ovr')
# # parameter = dataframe
param_dict={"C":[i for i in range(1,100)]}
Gridsearch = GridSearchCV(testSVM,cv = 5,scoring= 'accuracy',param_grid=param_dict)
Gridsearch.fit(x_train,y_train)
print("最佳参数：\n",Gridsearch.best_params_)
print("最佳结果：\n",Gridsearch.best_score_)

# clf = make_pipeline(StandardScaler(),SVC(gamma='auto'))
# 3.训练svm分类器
classifier = svm.SVC(C=2, kernel='rbf', gamma=10, decision_function_shape='ovr')  # ovr:一对多策略
classifier.fit(x_train,y_train)  # ravel函数在降维时默认是行序优先

# 4.计算svc分类器的准确率
print("训练集：", classifier.score(x_train, y_train))
print("测试集：", classifier.score(x_test, y_test))


# 查看决策函数
print('train_decision_function:\n', classifier.decision_function(x_train))
print('predict_result:\n', classifier.predict(x_train))


y_predict=classifier.predict(x_test)
print(y_predict)
print("直接比对真实值和预测值:\n",y_test==y_predict)

score=classifier.score(x_test,y_test)
print("准确率为：\n",score)

#Learning curve 检视过拟合
train_sizes, train_loss, test_loss = learning_curve(
    svm.SVC(gamma=0.001), x, y, cv=10, scoring='neg_mean_squared_error',
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1])

#平均每一轮所得到的平均方差(共5轮，分别为样本10%、25%、50%、75%、100%)
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
        label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()


param_range = np.logspace(-6,-2.3,5)

#使用validation_curve快速找出参数对模型的影响
train_loss, test_loss = validation_curve(
    svm.SVC(), x_train, y_train, param_name='gamma', param_range=param_range, cv=10, scoring='neg_mean_squared_error')

#平均每一轮的平均方差
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

#可视化图形
plt.plot(param_range, train_loss_mean, 'o-', color="r",
         label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g",
        label="Cross-validation")

plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()