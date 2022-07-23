#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 18:41:03 2022

@author: lijiao
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing datasets
# read the datasets segmented by ”tab“  and  delete the ” ” “
dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter = "\t",quoting= 3)



#cleaning data（先取少量数据测试）
import re
import nltk
nltk.download("stopwords") #下载虚词表
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = [ ]


for i in range(0,1000):

    # 1. 去掉数字和标点，即留下所有的字母（大小写）（利用sub函数）；去掉标点符号后用空格分割。
    review = re.sub("[^a-zA-Z]"," ",dataset["Review"][i])
    
    # 2. 大写字母转换成小写字母
    review = review.lower()
    
    # 3. 去掉虚词
    # import nltk
    # nltk.download("stopwords") #下载虚词表
    # from nltk.corpus import stopwords
    ## 将字符串转换为list （split）
    review = review.split()
    ## 通过for循环删除虚词(set(stopwords.words("english"))为了提升效率)
    review = [word for word in review if not word in set(stopwords.words("english"))]
    
    # 4. 词根化
    # from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    
    # 将list转化为字符串
    review =  " ".join(review)
    
    corpus.append(review)
    
    

# 创建词袋模型
## 文本向量化CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=(1500))  # 选取出现频次最多前1500个词语，max_features可依据实际情况调整
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[ : ,1].values



# 采用贝叶斯分类预测

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

    
    



























        
        
