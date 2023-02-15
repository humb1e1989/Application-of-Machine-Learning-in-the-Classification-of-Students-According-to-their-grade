import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import  datetime
# 开始时间
begin = datetime.datetime.now()
# 导入数据集
dataframe = pd.read_csv('.\CW_Data.csv')
# 删除空行
dataframe.drop(dataframe[dataframe['Programme'].isnull()].index,inplace=True)
# 观察数据之后发现ID一列没有实际意义删除
dataframe.drop(labels='ID',axis=1,inplace=True)
#去掉重复数据保留后者
dataframe.drop_duplicates(subset=[ 'Q1', 'Q2', "Q3" , "Q4" , "Q5"],inplace = True)
# 考虑后删除program0
dataframe[(dataframe.Programme == 0)].index.tolist()
dataframe.drop(dataframe[(dataframe.Programme == 0)].index,inplace=True)

x = np.array(dataframe.drop(labels="Programme", axis=1))
y = np.array(dataframe["Programme"])
# # 多项式模型朴素贝叶斯
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=100)
# 假设符合分布，使用高斯贝叶斯进行计算
clf = MultinomialNB()
clf.fit(x_train,y_train)
MultinomialNB()
# 验证
y_predict = clf.predict(x)
x_pred = clf.predict(x_test)
acc = sum(x_pred == y_test) / x_pred.shape[0]
print("准确率ACC: %.3f", acc)
# 交叉验证
scores = cross_val_score(clf, x, y, cv = 5)
print(scores)
print("多项式朴素贝叶斯预测的分类结果:",x_pred)

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