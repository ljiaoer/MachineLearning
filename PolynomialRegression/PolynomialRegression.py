# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:11:09 2020

@author: lijiaojiao
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#read datasets
dataset = pd.read_csv("Position_Salaries.csv")

x = dataset.iloc[:,1:2].values #提取 自变量(只取第一列，为了是矩阵，所以写为1：2)
y = dataset.iloc[:,2].values #提取 因变量

# =============================================================================
# #对自变量数据处理
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# labellencoder_x = LabelEncoder()
# x[:,3] = labellencoder_x.fit_transform(x[:,3])
# onehotencoder = OneHotEncoder(categorical_features = [3])
# x = onehotencoder.fit_transform(x).toarray()
# 
# #避免虚拟变量（dummy variable）
# x = x[:,1:]
# 
# #将数据集分为测试集和训练集
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2,random_state = 0)
# =============================================================================

#拟合线性回归
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)  #拟合好的线性模型


#拟合多项式回归
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =3)    #degree为自变量的最高次数，默认是2，可以通过调整degree的值拟合不同的多项式
x_poly = poly_reg.fit_transform(x)   #多项式参数的拟合
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

#线性回归拟合的可视化
plt.scatter(x,y,color = "red") #原始数据散点图
plt.plot(x,lin_reg.predict(x),color = "blue")
plt.title("truth or Bluff(LinearRegression)")
plt.xlabel("position level")
plt.ylabel("salary")

#多项式回归拟合的可视化
plt.scatter(x,y,color = "red") #原始数据散点图
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color = "blue")
plt.title("truth or Bluff(PolynomialRegression)")
plt.xlabel("position level")
plt.ylabel("salary")


#多项式回归拟合的可视化，使线条更平滑
x_grid = np.arange(min(x),max(x),0.1) #使线条更平滑
x_grid = x_grid.reshape(len(x_grid),1)    #x_grid转化为矩阵
plt.scatter(x,y,color = "red") #原始数据散点图
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color = "blue")
plt.title("truth or Bluff(PolynomialRegression)")
plt.xlabel("position level")
plt.ylabel("salary")

#实际预测
#线性预测
lin_reg.predict(6.5) 

#多项式模型预测
lin_reg_2.predict(poly_reg.fit_transform(6.5))


