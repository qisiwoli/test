#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import csv
from random import random

import matplotlib

from utils.shapley_calculation import shapley_calculation

#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import  math

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, num_best_provide
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, avg_mid
from models.test import test_img, test_acc

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    ri_base = [0.3, 0.25, 0.2]  # 应该和策略相关，数量越大 成本越高
    q = 1
    #每个参与者共用一个概率列表 表示选择数量策略的概率
    probabilities = [1/3,1/3,1/3]
    probabilities_dic = [[],[],[]]
    #策略列表表示可选择的三种策略
    strategy = [300,200,100]
    sigma_1 = 0.004 #控制收敛速度的参数
    #记录每个参与者在每一轮次的收益
    u_dict = [[],[],[]]
    alphe1 = 30000
    alphe2 = 25000
    alphe3 = 0.01
    beta1 = 0.0001
    beta2 =0.001
    u_d_list = [] #记录购买者收益
    shapleys_sum_list = []  # 存储每轮的shapleys比例
    shap = []

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape


    # all_num记录每个轮次各参与者提供数量
    all_num = []
    user_data_sizes = [0,0,0]

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print("初始模型性能准确率为：", test_acc(net_glob, dataset_test, args).item())
    net_glob.train()

    # 复制全局模型net_glob的权重
    w_glob = net_glob.state_dict()

    # training
    # 训练过程中每一轮次的损失函数列表
    loss_train = []
    accuracy_test = []
    # 交叉验证的损失和准确率
    cv_loss, cv_acc = [], []
    # 上一次迭代验证机的损失值
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    shapleys = np.zeros(len(user_data_sizes))
    average_list=[]
    ri=[0, 0, 0]
    every_u=[[],[],[]]

    if args.all_clients:
        print("Aggregation over all clients")
    for iter in range(20):

        # 每一轮次都先计算当前轮次应该提供的数量------------------------------------------------
        num = 0
        for i in range(len(probabilities)):
            num += probabilities[i] * strategy[i]
        for i in range(len(probabilities)):
            if num < 0:
                user_data_sizes[i] = 1
            else:
                user_data_sizes[i] = int(num)
        print(iter,user_data_sizes)  # 初始结果应为【200，200，200】
        # all_num记录每个轮次各参与者提供数量
        all_num.append(copy.deepcopy(user_data_sizes))
        best_num = num_best_provide(dict_users, args.num_users, user_data_sizes)
        for y in range(len(probabilities)):
            probabilities_dic[y].append(probabilities[y])
        print(probabilities_dic)
        #--------------------------------------------------------------------------------

        w_locals = [w_glob for i in range(args.num_users)]
        # 存储每个参与者的局部损失值
        loss_locals = []
        for idx in range(args.num_users):
            # 遍历被选中的参与者 进行本地训练
            # 每轮次更新的最佳传参时应该把数据提供量传入
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=best_num[idx])
            # 获得更新后的权重和局部损失
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals[idx] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))

        # 计算当前轮次shapley值---------------------------------
        shapley, v = shapley_calculation(user_data_sizes, dataset_test, args, net_glob, w_glob, w_locals, iter)
        shapleys += shapley
        # 计算shapley比例结果
        shap_sum = sum(shapleys)
        shapleys_sum = [i / shap_sum for i in shapleys]
        shapleys_sum_list.append(shapleys_sum)  # 新增：保存当前轮次的比例
        shap.append(shapleys)
        print("目前轮次的shapleys:", shapleys)
        print("目前轮次的shapleys比例:", shapleys_sum)
        #-----------------------------------------------------
        # update global weights
        w_glob = FedAvg(w_locals, user_data_sizes)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # 更新计算成本
        for z in range(len(strategy)):
            ri[z] = round(ri_base[z] * (10*probabilities[z])**(1/2) , 8)
        print(ri)
        total_num = sum(user_data_sizes)
        sum_u = [0,0,0] #记录每种策略的总收益
        # 每个参与者在所有策略下进行训练，以计算每个策略的收益--------------------------------------------------------------------
        for j in range(args.num_users):  # 表示第j个参与者
            u_ = [0, 0, 0]  # 记录参与者在三种策略下的收益
            every = 0
            for z in range(len(strategy)):
                #每种策略的shapley可以用决策数量乘概率 / 总数量
                u_[z] = (shapleys_sum[j] * ((strategy[z]*probabilities[z])/user_data_sizes[j])) * alphe3 * q * v * total_num - ri[z] * (strategy[z]*probabilities[z])
                sum_u[z] += u_[z]
                every += probabilities[z] * u_[z]
            every_u[j].append(every)
        print("每个参与者的收益",every_u)
        for i in range(len(strategy)):
            u_dict[i].append(sum_u[i])
        print("每一轮次提供者的三种策略总收益",sum_u)


        # 计算平均收益
        average_u = 0
        for zi in range(len(u_)):
            average_u += probabilities[zi] * sum_u[zi]
        average_list.append(average_u)
        print("三种策略平均收益", average_u)

        # 演化博弈更新概率
        for xc in range(len(probabilities)):
            probabilities[xc] = probabilities[xc] + sigma_1 * probabilities[xc] * (
                    sum_u[xc] - average_u)  # 应该加一个归一化处理，保证概率加起来等于1
            if probabilities[xc] < 0:
                probabilities[xc] = 0
        print("未归一化probabilities:", probabilities)
        # 归一化处理
        summ = 0
        for y in range(3):
            summ += probabilities[y]
        for y in range(3):
            probabilities[y] = probabilities[y] / summ
        print("probabilities:",probabilities)


        u_d = alphe1 * math.log(1+beta1*v) + alphe2*math.log(1+beta2*total_num) - alphe3 *  q * v * total_num
        u_d_list.append(u_d)
        print("模型购买者的收益变化是",u_d_list)  #结果应该是上升后平稳
        # ---------------------------------------------------------------------------------------------------------------


        # 在每一轮次都评估模型在测试集上的准确率，表示当前模型性能
        net_glob.eval()
        acc_test = test_acc(net_glob, dataset_test, args)
        print('Round {:3d}, Testing accuracy: {:.2f}'.format(iter, acc_test))
        accuracy_test.append(acc_test)
        net_glob.train()




    # 指定CSV文件的文件名 记录结每个参与者效用结果和平均收益  购买者收益
    csv_name = "D:/project/PythonProject1/save/u_i__sigma_{}_acc_{}.csv".format(sigma_1, v)
    # 打开文件，使用 'w' 模式表示写入，如果文件不存在会创建新文件，如果存在则覆盖原有内容
    try:
        with open(csv_name, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(3):
                writer.writerow([round(j, 3) for j in u_dict[i]])
            for i in range(3):
                writer.writerow([round(j, 3) for j in every_u[i]])
            writer.writerow([j for j in average_list])
            writer.writerow([j for j in u_d_list])
    except Exception as e:
        print(f"写入文件时出现错误: {e}")
    # 设置画布大小
    plt.figure(figsize=(10, 6))

    # 定义不同参与者的颜色和标签
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']  # 颜色!!!
    labels = [f'参与者{i + 1}' for i in range(4)]  # !!!
    every_u.append(average_list)
    # 遍历每个参与者的数据并绘图
    for i in range(4):  # !!!
        # x轴为轮次（从1开始），y轴为收益
        rounds = range(1, len(every_u[i]) + 1)  # 轮次：1,2,...,n
        earnings = every_u[i]  # 对应参与者的收益列表
        plt.plot(rounds, earnings,
                 color=colors[i],
                 label=labels[i],
                 marker='o',  # 数据点用圆点标记
                 markersize=5,
                 linewidth=2)

    # 添加图表细节
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('收益', fontsize=12)
    plt.title('各参与者每轮收益变化', fontsize=14)
    plt.legend(fontsize=10)  # 显示图例（区分参与者）
    plt.grid(linestyle='--', alpha=0.6)  # 显示网格线
    plt.xticks(rounds)  # x轴刻度与轮次一一对应

    # 显示图像
    plt.show()

    #模型购买者图像
    # 创建画布
    plt.figure(figsize=(10, 6))

    # 绘制线型图（x轴为索引/序号，y轴为 u_d_list 的值）
    plt.plot(u_d_list,
             color='blue',  # 线条颜色
             marker='o',  # 数据点标记
             markersize=5,  # 标记大小
             linewidth=2)  # 线条宽度

    # 添加标签和标题
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('收益', fontsize=12)
    plt.title('model buyer shouyi', fontsize=14)
    plt.grid(linestyle='--', alpha=0.7)  # 网格线

    # 显示图像
    plt.show()

    # 绘制三种策略的总收益趋势图
    plt.figure(figsize=(10, 6))

    # 定义策略的颜色和标签
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色，与策略比例图保持一致
    labels = [f'策略{i + 1}（数量{strategy[i]}）' for i in range(3)]  # 标签包含策略数量

    # 遍历三种策略的总收益数据
    for i in range(3):
        rounds = range(1, len(u_dict[i]) + 1)  # 轮次从1开始
        plt.plot(rounds, u_dict[i],
                 color=colors[i],
                 label=labels[i],
                 linewidth=2,
                 marker='o',
                 markersize=5)

    # 添加图表细节
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('总收益', fontsize=12)
    plt.title('三种策略的总收益变化趋势', fontsize=14)
    plt.legend(fontsize=10)  # 显示图例区分不同策略
    plt.grid(linestyle='--', alpha=0.6)  # 网格线
    plt.xticks(rounds)  # x轴刻度与轮次对应
    plt.tight_layout()  # 自动调整布局

    # 显示图像
    plt.show()

    # 指定CSV文件的文件名 记录probabilities和数量变化
    csv_name = "D:/project/PythonProject1/save/probabilities__sigma_{}_ri_base_{}.csv".format(sigma_1, ri_base)
    # 打开文件，使用 'w' 模式表示写入，如果文件不存在会创建新文件，如果存在则覆盖原有内容
    try:
          with open(csv_name, 'w', newline='', encoding='utf-8') as csvfile:
              writer = csv.writer(csvfile)
              for i in range(3):
                  writer.writerow([round(j, 2) for j in probabilities_dic[i]])
              for i in range(len(all_num)):
                  writer.writerow([j for j in all_num[i]])
    except Exception as e:
        print(f"写入文件时出现错误: {e}")
    # 设置画布大小
    plt.figure(figsize=(10, 6))

    # 定义不同策略的颜色和标签
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色（可自定义）
    labels = [f'策略{i + 1}' for i in range(3)]

    # 绘制每条策略的曲线
    for i in range(3):
        # x轴为轮次（从1开始），y轴为比例
        rounds = range(1, len(probabilities_dic[i]) + 1)
        plt.plot(rounds, probabilities_dic[i],
                 color=colors[i],
                 label=labels[i],
                 linewidth=2,  # 线宽
                 marker='o',  # 数据点标记
                 markersize=5,  # 标记大小
                 alpha=0.8)  # 透明度

    # 添加图表信息
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('策略比例', fontsize=12)
    plt.title('各策略在所有轮次中的比例变化', fontsize=14)
    plt.legend(fontsize=10)  # 显示图例
    plt.grid(linestyle='--', alpha=0.5)  # 网格线
    plt.ylim(0, 1.05)  # 比例范围限制在0-1.05（确保比例显示完整）
    plt.tight_layout()  # 自动调整布局

    # 显示图像
    plt.show()
    #print(all_num)
    '''# 画出参与者最佳提供量趋势
    p1 = []
    p2 = []
    p3 = []
    for i in all_num:
        p1.append(i[0])
        p2.append(i[1])
        p3.append(i[2])
    data = [p1, p2, p3]
    # 生成时间序列作为横坐标，这里简单用数字表示时期，范围根据内层列表长度确定
    time_periods = np.arange(len(data[0]))
    colors = ['r', 'g', 'b']
    line_styles = ['-'] * len(data)
    markers = [None] * len(data)
    for i in range(len(data)):
        plt.plot(time_periods, data[i], label=f'Provider {i + 1}', color=colors[i], linestyle=line_styles[i],
                 marker=markers[i])
    plt.title('Best Volume of Data')
    plt.xlabel('FL Iterative Times')
    plt.ylabel('Data Volume')
    plt.legend()
    plt.savefig('./save/best_volume_ri_{}_init_{}-{}-{}_q-{}_acc-{}.png'.format(ri[0], all_num[0][0], all_num[0][1],
                                                                                all_num[0][2], q, v))

    # plot loss curve
    plt.figure()
    plt.plot(range(len(accuracy_test)), accuracy_test)
    plt.ylabel('test_accuracy')
    plt.savefig('./save/test_accuracy.png')'''
    print(net_glob)
    # testing

    # 绘制shapleys比例趋势图
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']  # 颜色'#d62728'!!!
    labels = [f'参与者{i + 1}' for i in range(3)]  # !!!

    # 遍历每个参与者的shapleys比例数据
    for i in range(3):#!!!
        # 提取每轮中第i个参与者的比例
        values = [round(ss[i], 4) for ss in shapleys_sum_list]
        rounds = range(1, len(values) + 1)
        plt.plot(rounds, values,
                 color=colors[i],
                 label=labels[i],
                 marker='s',  # 用正方形标记
                 markersize=5,
                 linewidth=2)

    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('Shapley值比例', fontsize=12)
    plt.title('各参与者Shapley值比例变化趋势', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(linestyle='--', alpha=0.6)
    plt.xticks(rounds)
    plt.ylim(0, 1.0)  # 比例范围限制在0-1之间
    plt.show()
    # 指定CSV文件的文件名 记录shap
    csv_name = "D:/project/PythonProject1/save/shap__sigma_{}_acc_{}.csv".format(sigma_1, v)
    # 打开文件，使用 'w' 模式表示写入，如果文件不存在会创建新文件，如果存在则覆盖原有内容
    try:
        with open(csv_name, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(3):
                writer.writerow([round(ss[i], 4) for ss in shapleys_sum_list])
            for i in range(3):
                writer.writerow([round(ss[i], 4) for ss in shap])
    except Exception as e:
        print(f"写入文件时出现错误: {e}")


