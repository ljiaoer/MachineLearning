#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 23:28:04 2022

@author: lijiao
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# import datasets

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")


#thompson算法


d = 10 #广告数量
N = 10000 #用户数



numbers_of_rewards_1=[0] * d
numbers_of_rewards_0=[0] * d

ads_selected = []
total_rewards = 0

for n in range(0,N):            #n为被投放的用户
    ad =0
    max_random = 0
    for i in range(0,d):        #d广告序号
        random_beta = random.betavariate(numbers_of_rewards_1[i]+1, numbers_of_rewards_1[0]+1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] +1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] +1
    
    total_rewards = total_rewards + reward
    
            
    
# visulaization

plt.hist(ads_selected)
plt.xlabel("Ads")
plt.ylabel("number of times each ad was selected")
plt.show()










