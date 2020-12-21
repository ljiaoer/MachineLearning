# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:48:36 2020

@author: lijiaojiao
"""


import numpy as np
import matplotlib as plt
import pandas as pd

dataset = pd.read_csv("Market_Basket_Optimisation.csv",header = None)
#python 中共7500行？
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

#Aprior模型训练
from apyori import  apriori
rules = apriori(transactions,min_support = 0.03,min_confidence = 0.2, min_lift =3,min_length = 2)

results = list(rules)
myResults = [list(x) for x in results]


