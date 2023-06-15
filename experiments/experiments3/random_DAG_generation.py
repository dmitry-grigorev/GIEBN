## original https://github.com/Livioni/DAG_Generator

import random, math, argparse
import numpy as np
from numpy.random.mtrand import sample
from matplotlib import pyplot as plt
import networkx as nx


def DAGs_generate(n=10, max_out=2, alpha=1, beta=1.0, random_state=42):
    ##############################################initialize###########################################

    np.random.seed(random_state)
    random.seed(random_state)

    length = math.floor(math.sqrt(n) / alpha)
    mean_value = n / length
    random_num = np.random.normal(loc=mean_value, scale=beta, size=(length, 1))
    ###############################################division############################################
    position = dict()
    generate_num = 0
    dag_num = 1
    dag_list = []
    for i in range(len(random_num)):
        dag_list.append([])
        for j in range(math.ceil(random_num[i])):
            dag_list[i].append(j)
        generate_num += math.ceil(random_num[i])

    if generate_num != n:
        if generate_num < n:
            for i in range(n - generate_num):
                index = random.randrange(0, length, 1)
                dag_list[index].append(len(dag_list[index]))
        if generate_num > n:
            i = 0
            while i < generate_num - n:
                index = random.randrange(0, length, 1)
                if len(dag_list[index]) == 1:
                    i = i - 1 if i != 0 else 0
                else:
                    del dag_list[index][-1]
                i += 1

    dag_list_update = []
    pos = 1
    max_pos = 0
    for i in range(length):
        dag_list_update.append(list(range(dag_num, dag_num + len(dag_list[i]))))
        dag_num += len(dag_list_update[i])
        pos = 1
        for j in dag_list_update[i]:
            position[j] = (3 * (i + 1), pos)
            pos += 5
        max_pos = pos if pos > max_pos else max_pos
        position['Start'] = (0, max_pos / 2)
        position['Exit'] = (3 * (length + 1), max_pos / 2)

    ############################################link###################################################
    into_degree = [0] * n
    out_degree = [0] * n
    edges = []
    pred = 0

    for i in range(length - 1):
        sample_list = list(range(len(dag_list_update[i + 1])))
        for j in range(len(dag_list_update[i])):
            od = random.randrange(1, max_out + 1, 1)
            od = len(dag_list_update[i + 1]) if len(dag_list_update[i + 1]) < od else od
            bridge = random.sample(sample_list, od)
            for k in bridge:
                edges.append((dag_list_update[i][j], dag_list_update[i + 1][k]))
                #into_degree[pred + len(dag_list_update[i]) + k] += 1
                out_degree[pred + j] += 1
        pred += len(dag_list_update[i])

    ######################################create start node and exit node################################
    # for node,id in enumerate(into_degree):#给所有没有入边的节点添加入口节点作父亲
    #    if id ==0:
    #        edges.append(('Start',node+1))
    #        into_degree[node]+=1

    # for node,od in enumerate(out_degree):#给所有没有出边的节点添加出口节点作儿子
    #    if od ==0:
    #        edges.append((node+1,'Exit'))
    #        out_degree[node]+=1

    # if np.random.uniform(size=1) > 0.5:
    #    edges.append(('Start', 'Exit'))

    #############################################plot##################################################
    return edges, into_degree, out_degree, position


def plot_DAG(edges, position):
    g1 = nx.DiGraph()
    g1.add_edges_from(edges)
    nx.draw_networkx(g1, arrows=True, pos=position)
