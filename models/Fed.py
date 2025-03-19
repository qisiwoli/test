#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w,sizes):
    #聚合
    all_sizes = sum(sizes)
    #w_avg是一个字典
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        weighted_sum = 0
        for i in range(len(w)):
            weighted_sum += (sizes[i] / all_sizes) * w[i][k]
        w_avg[k] = weighted_sum
    return w_avg


def avg_mid(w_add,w_g,sizes, id1, num,per):
    all_sizes = 0
    for i in range(num+1):
        all_sizes+=sizes[per[i]]
    if num==0:
        for k in w_g.keys():
            w_g[k] = w_add[k]
    else:
        for k in w_g.keys():
            w_g[k] = (sizes[id1]/all_sizes)*w_add[k] + w_g[k]*(1-(sizes[id1]/all_sizes))
    return w_g





