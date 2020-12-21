# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 22:58:31 2020

@author: lijiaojiao
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

from sklearn.cluster import KMeans
wcss = [] #组间距离
#组数1:10
for i in range(1,11):  
    kmeans = KMeans(n_clusters = i,max_iter=300,n_init=10,init = "k-means++",
                    random_state=0)  #init 参数是初始值的选取
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("the elbow methond")
plt.xlabel("number of cluster")
plt.ylabel("wcss")
plt.show()

#kmeans算法#
kmeans =KMeans(n_clusters = 5, max_iter = 300,n_init = 10,init = "k-means++",random_state = 0 )
y_kmeans = kmeans.fit_predict(x)

#可视化各个类
plt.scatter(x[y_kmeans == 0,0],x[y_kmeans == 0,1],s=100,c = "red",label="Cluster0")
plt.scatter(x[y_kmeans == 1,0],x[y_kmeans == 1,1],s=100,c = "blue",label="Cluster0")
plt.scatter(x[y_kmeans == 2,0],x[y_kmeans == 2,1],s=100,c = "green",label="Cluster0")
plt.scatter(x[y_kmeans == 3,0],x[y_kmeans == 3,1],s=100,c = "cyan",label="Cluster0")
plt.scatter(x[y_kmeans == 4,0],x[y_kmeans == 4,1],s=100,c = "black",label="Cluster0")

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s =300,
            c = "yellow",label = "centers")
plt.title("")
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.legend()
plt.show()


