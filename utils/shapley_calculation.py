import copy
import random
import numpy as np
import csv
import itertools

from models.test import test_acc
from models.Fed import avg_mid, FedAvg


def shapley_calculation(user_data_sizes,datatest,args,net_glob,w_glob,w_locals,count):
    #在每轮次的联邦学习中，每个参与者的权重已经保存，只需要进行聚合，然后评估新模型的准确率，不需要重新进行训练
    #参数含义：参与者列表，全局模型准确率，轮内轮间早停的阈值
    num = len(user_data_sizes)
    #初始化shapley值为0
    shapleys = np.zeros(num)
    final_shap=[]
    #存储每次迭代计算得到的中间shapley
    iter_time = 0
    participants = list(range(num))
    permutations_list = list(itertools.permutations(participants))


    #指定迭代次数，平均值近似计算真实值     可以通过判断是否收敛！！！！！！！！！！！！！！！！！！
    for i in permutations_list:
        # 计算完shapley后要恢复原来的模型和参数,对原来模型进行深拷贝
        w_g = copy.deepcopy(w_glob)
        net = copy.deepcopy(net_glob).to(args.device)
        v = test_acc(net, datatest, args)

        shap=np.zeros(num)
        iter_time += 1
        #随机排列组合 得到索引列表
        permutation = i
        v_before = copy.deepcopy(v)
        print("迭代次数为",iter_time)
        for j in range(num):
            #聚合新加入的客户端的权重 更新当前模型权重
            user_id = permutation[j]
            w_add = w_locals[user_id]
            w_g = avg_mid(w_add,w_g,user_data_sizes,user_id,j,permutation)
            net.load_state_dict(w_g)
            # 计算当前模型的准确率
            v_new = test_acc(net,datatest, args)
            #判断截断条件
            #if v_new - v_before >= 0.03:
            shap[permutation[j]] += v_new - v_before
            shapleys[permutation[j]] += v_new - v_before
            v_before = copy.deepcopy(v_new)
            #else:
                #break
        final_shap.append(shap)


    # 指定CSV文件的文件名
    csv_name = "C:/Users/user/PycharmProjects/federated-learning/save/shapley_values/fed_{}.csv".format(count)
    # 打开文件，使用 'w' 模式表示写入，如果文件不存在会创建新文件，如果存在则覆盖原有内容
    try:
        with open(csv_name, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [f'参与者{i + 1}' for i in range(num)]
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)
            for i in range(iter_time):
                writer.writerow([round(i,2) for i in final_shap[i]])
            writer.writerow([round(i/len(permutations_list),2) for i in shapleys])
    except Exception as e:
        print(f"写入文件时出现错误: {e}")

    shapleys = [i /len(permutations_list) for i in shapleys]
    return shapleys, v_new.item()


















