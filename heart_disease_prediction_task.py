# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

df=pd.read_csv('/content/heart.csv')
df

df.info()

df.describe()

df.isna().sum()

df.head()

df.tail()

df.shape

df.columns

df['target'].value_counts()

df['chol'].value_counts()

df['sex'].value_counts()

df=df.drop([df.columns[3],df.columns[6],df.columns[7],df.columns[8],df.columns[9],df.columns[11],df.columns[12]],axis=1)
df.head()

import matplotlib.pyplot as plt
import plotly.express as px
fig=px.histogram(df,x='age')
fig.show()

x=df.drop(['target'],axis=1)
y=df['target']
print(x.head())
print(y.head())

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=2)

print(train_x.head())

print(train_y.head())

"""# Linear Regression"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
model_lr=LinearRegression()
model_lr.fit(train_x,train_y)

"""## **LinearRegression** **PerformanceMetrics**"""

y_pred_train_lr=model_lr.predict(train_x)
y_pred_test_lr=model_lr.predict(test_x)
train_mse_lr=mean_squared_error(train_y,y_pred_train_lr)
test_mse_lr=mean_squared_error(test_y,y_pred_test_lr)
train_r2_lr=r2_score(train_y,y_pred_train_lr)
test_r2_lr=r2_score(test_y,y_pred_test_lr)

print(train_mse_lr)
print(test_mse_lr)
print(train_r2_lr)
print(test_r2_lr)

"""# **Logistic Regression**"""

from sklearn.linear_model import LogisticRegression
model_lr=LogisticRegression(max_iter=1000)
model_lr.fit(train_x,train_y)
y_pred_train=model_lr.predict(train_x)
y_pred_test=model_lr.predict(test_x)

print(y_pred_test)

print(y_pred_train)

"""# **LogisticRegression** **PerformanceMetrics**"""

from sklearn.metrics import accuracy_score,confusion_matrix

accuracy_train=accuracy_score(train_y,y_pred_train)
print("Accuracy of the Logistic Regression model on your train",accuracy_train)

accuracy_test=accuracy_score(test_y,y_pred_test)
print("Accuracy of the Logistic Regression model on your test dataset",accuracy_test)

confusion_matrix(test_y,y_pred_test)

tn,fp,fn,tp=confusion_matrix(test_y,y_pred_test).ravel()
print("TN:",tn)
print("FN:",fn)
print("FP:",fp)
print("TP:",tp)

accuracy=(tp+tn)/(tp+tn+fp+fn)
accuracy

confusion_matrix(train_y,y_pred_train)

from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score,f1_score

Recall=tp/(tp+fn)
print("Recall of Logistic Regression is:",Recall)
Precision=tp/(tp+fp)
print("Precision of Logistic Regression is",Precision)
F1_score=(2*Recall+Precision)/(Recall+Precision)
print("F1_score of Logistic Regression is:",F1_score)

"""# **DecisionTree**"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(train_x,train_y)
y_pred_test_dt=dt.predict(test_x)
y_pred_train_dt=dt.predict(train_x)

accuracy_train=accuracy_score(train_y,y_pred_train)
print("Accuracy of the Decision Tree model on your train",accuracy_train)

accuracy_test=accuracy_score(test_y,y_pred_test)
print("Accuracy of the Decision Tree model on your test dataset",accuracy_test)

"""# **KNN**"""

from sklearn.neighbors import KNeighborsClassifier
model_knn=KNeighborsClassifier(n_neighbors=5)
model_knn.fit(train_x,train_y)
y_pred_train_knn=model_knn.predict(train_x)
y_pred_test_knn=model_knn.predict(test_x)

print(y_pred_train_knn)
print(y_pred_test_knn)

accuracy_test=accuracy_score(test_y,y_pred_test_knn)
print("Accuracy of the KNN model on your test",accuracy_test)
precision_test=precision_score(test_y,y_pred_test_knn)
print("Precision of the KNN model on your test",precision_test)
recall_test=recall_score(test_y,y_pred_test_knn)
print("Recall of the KNN model on your test",recall_test)
f1score_test=f1_score(test_y,y_pred_test_knn)
print("F1 score of the KNN model on your test",f1score_test)

"""# **KMeans**"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

k=3
model_Kmeans=KMeans(n_clusters=k,random_state=42)
columns_for_clustering=['age','chol']
df_for_clustering=df[columns_for_clustering]
model_Kmeans.fit(df_for_clustering)

cluster_labels=model_Kmeans.labels_
cluster_centers=model_Kmeans.cluster_centers_

plt.figure(figsize=(8,6))
plt.scatter(df_for_clustering['age'],df_for_clustering['chol'],c=cluster_labels,cmap='viridis',alpha=0.7)
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],c='red',marker='x',s=100,label='centroids')
plt.xlabel('age')
plt.ylabel('cholesterol')
plt.title('K-means Clustering')
plt.legend()
plt.show()

"""### **KMeans** **PerformanceMetrics**"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, homogeneity_score, completeness_score, v_measure_score
import matplotlib.pyplot as plt

k=2
model_Kmeans=KMeans(n_clusters=k,random_state=42)
model_Kmeans.fit(x)
cluster_labels=model_Kmeans.labels_
inertia=model_Kmeans.inertia_
silhouette=silhouette_score(x,cluster_labels)
davies_bouldin=davies_bouldin_score(x,cluster_labels)
calinski_harabasz=calinski_harabasz_score(x,cluster_labels)
homogeneity=homogeneity_score(y,cluster_labels)
completeness=completeness_score(y,cluster_labels)
v_measure=v_measure_score(y,cluster_labels)

print("inertia",inertia)
print("silhouette_score",silhouette)
print("davies_bouldin_score",davies_bouldin)
print("calinski_harabasz_score",calinski_harabasz)
print("homogeneity_score",homogeneity)
print("completeness_score",completeness)
print("v_measure_score",v_measure)

plt.figure(figsize=(8,6))
colors=['navy','turquoise']
lw=2
plt.figure(figsize=(8,6))
plt.scatter(df_for_clustering['age'],df_for_clustering['chol'],c=cluster_labels,cmap='viridis',alpha=0.7)
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],c='red',marker='x',s=100,label='centroids')
plt.xlabel('age')
plt.ylabel('cholesterol')
plt.title('K-means Clustering')
plt.legend()
plt.show()

"""# **LDA**"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np
lda = LDA(n_components=1)
x_lda=lda.fit_transform(x,y)

plt.figure(figsize=(8,6))
colors=['navy','turquoise']
lw=2
for color,i,target_name in zip(colors,[0,1],['no heart disease','heart disease']):
  plt.scatter(x_lda[y==i,0],np.zeros_like(x_lda[y==i,0]),color=color,alpha=0.8,lw=lw,label=target_name)
  plt.legend(loc='best',shadow=False,scatterpoints=1)
plt.title('LDA of heart disease dataset')
plt.xlabel('LDA1')
plt.show()
