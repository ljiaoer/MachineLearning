# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 23:07:15 2020

@author: dell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#read datasets
dataset = pd.read_csv("50_startups.csv")

x = dataset.iloc[:,:-1].values #提取 自变量
y = dataset.iloc[:,4].values #提取 因变量

#对自变量数据处理
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labellencoder_x = LabelEncoder()
x[:,3] = labellencoder_x.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#避免虚拟变量（dummy variable）
x = x[:,1:]

#将数据集分为测试集和训练集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2,random_state = 0)


#创建回归分类器
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train) #拟合方法

#利用测试集预测
y_pred = regressor.predict(x_test)


#建立选择模型筛选自变量，剔除无用自变量（利用backward eliminate算法）
import statsmodels.api as sm
x_train = np.append(arr = np.ones((40,1)),values =x_train,axis = 1)
x_opt = x_train[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog= y_train,exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x_train[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog= y_train,exog = x_opt).fit()
regressor_OLS.summary()


x_opt = x_train[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog= y_train,exog = x_opt).fit()
regressor_OLS.summary()

# =============================================================================
# plt.scatter(x_train,y_train,color = "red")  #实际数据
# plt.plot(x_train,regressor.predict(x_train),color = "blue") #训练集的预测数据
# plt.title("salary(train set)")
# plt.xlabel("year of experience ")
# plt.ylabel("salary")
# plt.show()
# =============================================================================



#对因变量数据处理
#labellencoder_y = LabelEncoder()
#y[:,0] = labellencoder_y.fit_transform(y[:,0] )
