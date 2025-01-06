# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 10:24:13 2023

@author: XJM
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def dpli_transform(conn, thr):
    ## 将mne_connectivity跑出的dpli矩阵转换为 0.5-1的矩阵
    num_node = conn.shape[0]
    temp = np.ones([num_node,num_node])
    temp = np.triu(temp,k=1)
    conn_T = temp - conn.T
    conn = conn + conn_T
    
    conn[conn < thr] = 0
    return conn


def psl_transform(conn,vec,thr_percentage):
    ## 将PSI算法跑出的矩阵 进行转化为只有正向连接的网络
    num_node = conn.shape[0]
    conn_T = conn.T
    
    conn = conn - conn_T
    
    vec = np.abs(vec).squeeze()
    sorted_vec = np.sort(vec)[::-1]
    index = int((num_node * (num_node-1))/2 * thr_percentage)
    thr_val = sorted_vec[index]
    conn[conn<thr_val] = 0
    return conn
 

def module_7net_hemi(roi, left_idx, right_idx):
    ## 分左右半球计算稀疏模式的brain paterns在每个ICNs的统计 （绝对值数量） 适用于 先左后右的半球模板
    net_val = np.unique(left_idx)
    num_net = net_val.shape[0]
    num_hemi_node = left_idx.shape[0]
    roi_left, roi_right = roi[0:num_hemi_node], roi[num_hemi_node:]
    
    roi_net = np.zeros([2*num_net])
    for i in range(num_net):
        index_left = np.where(left_idx == net_val[i])
        temp_left = roi_left[index_left]
        temp_left[temp_left>0] = 1
        roi_net[i] = np.sum(temp_left)
        
        index_right = np.where(right_idx == net_val[i])
        temp_right = roi_right[index_right]
        temp_right[temp_right>0] = 1
        roi_net[i+num_net] = np.sum(temp_right)
    return roi_net
    
    
def directed_net_summary(net, label):
    ## 对于一个有向网络，将节点划分为不同的社团，统计社团内部的连接和社团之间的连接。最后返回社团之间的有向连接网络
    label = np.array(label)
    label_val = np.unique(label)
    num_val = label_val.shape[0]
    module_net = np.zeros([num_val, num_val])
    
    for i in range(num_val):
        for j in range(num_val):
            index_i = np.where(label == label_val[i])[0]
            index_j = np.where(label == label_val[j])[0]            
            mat_temp = net[index_i,:]
            mat_temp = mat_temp[:,index_j]
            module_net[i,j] = np.sum(mat_temp)
            
    return module_net



def conn_density(A,node_idx):
    ## 对于一个邻接矩阵A 计算特定node_idx的平均连接强度
    connections = []
        
    for i in node_idx:
        for j in node_idx:
            if i != j:
                connections.append(A[i][j])
    return np.mean(connections)

def random_conn_density(A,rate,N):
    ## 计算一个邻接矩阵A 给定一定比例节点连接强度的随即水平
    n = A.shape[0]
    connections_avg = []
    for _ in range(N):
        selected_nodes = np.random.choice(n, int(n * rate), replace=False)
        connections_avg.append(conn_density(A,selected_nodes))
    return connections_avg

def overlap_rate(vec, mat):
    ## 计算重叠比例，vec和mat都是 0 1数组和矩阵
    ## vec 是0 1 数组， mat是0 1矩阵， 计算vec对mat每列的重叠度（矩阵的点乘 求和），再标准化
    num_mat = mat.shape[1]
    rate = []
    summary = np.sum(vec)
    for i in range(num_mat):
        temp = np.sum(np.multiply(vec,mat[:,i]))
        rate.append(temp / summary)
    rate = np.array(rate)
    return rate
        

def plot_node_edge(nodes,matrix,node_order,node_colors,save_name,
                   node_size=1000,node_linewidths=1.5,edge_width=1.5,arrowsize=30,rad=0.25):
    ## 画具有节点和边的有向网络连通图
    G = nx.DiGraph()
    # 添加节点
    G.add_nodes_from(nodes)
    # 添加有向边
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if matrix[i][j] != 0:
                G.add_edge(nodes[i], nodes[j], weight=matrix[i][j])
    
    # 自定义节点排序
    #pos = nx.shell_layout(G, nlist=[custom_order])
    pos = nx.circular_layout(G)  
    # 调整节点位置以满足自定义顺序
    angle = 2 * np.pi / len(node_order)
    for i, node in enumerate(node_order):
        pos[node] = np.array([np.cos(i * angle), np.sin(i * angle)])

    
    plt.figure(figsize=(8, 8))
    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=[node_colors[node] for node in nodes], edgecolors='black', linewidths=node_linewidths)
    # 绘制边
    nx.draw_networkx_edges(G, pos, width=edge_width, arrows=True,arrowstyle='-|>', arrowsize=arrowsize, connectionstyle=("Arc3,rad="+str(rad)))

    # 添加标签
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

    # 显示图
    plt.axis('off')
    
    #plt.tight_layout()
    plt.savefig(save_name, format='svg')
    plt.show()

    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
   