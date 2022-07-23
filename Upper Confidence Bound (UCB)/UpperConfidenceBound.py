#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 13:25:33 2022

@author: lijiao
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# import datasets

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")


#置信区间算法


d = 10 #广告数量
N = 10000 #用户数


numbers_of_selections = [0]* d   #广告被投放的总次数
sum_of_clicks = [0]*d            # 广告i被点击的数量
ads_selected = []
total_clicks = 0

for n in range(0,N):            #n为被投放的用户
    ad =0
    max_upper_bound = 0
    for i in range(0,d):        #d广告序号
        if(numbers_of_selections[i]>0):
            average_click =sum_of_clicks[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/numbers_of_selections[i])
            upper_bound = average_click + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    click = dataset.values[n,ad]
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    sum_of_clicks[ad] =  sum_of_clicks[ad]+click
    total_clicks = total_clicks + click
    
            
    
# visulaization

plt.hist(ads_selected)
plt.xlabel("Ads")
plt.ylabel("number of times each ad was selected")
plt.show()




















