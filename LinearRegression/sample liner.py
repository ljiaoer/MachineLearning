# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 23:15:02 2020

@author: dell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#read datasets
dataset = pd.read_csv("Salary_Data.csv")

x = dataset.iloc[:,:-1].values #提取 自变量
y = dataset.iloc[:,:1].values #提取 因变量

#将数据集分为测试集和训练集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 1/3,random_state = 0)

#创建简单回归分类器
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train) #fit()做拟合

y_pred = regressor.predict(x_test)

#可视化
plt.scatter(x_train,y_train,color = "red")  #实际数据
plt.plot(x_train,regressor.predict(x_train),color = "blue") #训练集的预测数据
plt.title("salary(train set)")
plt.xlabel("year of experience ")
plt.ylabel("salary")
plt.show()



