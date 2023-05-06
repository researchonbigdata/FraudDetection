# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:19:21 2022

@author: Administrator
"""


import torch
import dgl
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, precision_score

def normalize_row(mx):
	"""Row-normalize sparse matrix"""
	rowsum = np.array(mx.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx

def normalize_col(mx):
    """Row-normalize sparse matrix"""
    colmean = np.array(mx.mean(0))
    c_inv = np.power(colmean, -1).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = torch.tensor(np.diag(c_inv))
    mx = np.array(mx).dot(c_mat_inv)
    return mx

def test_model(idx_test, labels, model):
    gnn_prob = model.to_prob(idx_test)
    
    auc_gnn = roc_auc_score(labels, gnn_prob.data.cpu().numpy()[:,1].tolist())
    precision_gnn = precision_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    a_p = average_precision_score(labels, gnn_prob.data.cpu().numpy()[:,1].tolist())
    recall_gnn = recall_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    f1 = f1_score(labels, gnn_prob.data.cpu().numpy().argmax(axis=1), average="macro")
    
    print(f"GNN auc: {auc_gnn:.4f}")
    print(f"GNN precision: {precision_gnn:.4f}")
    print(f"GNN a_precision: {a_p:.4f}")
    print(f"GNN Recall: {recall_gnn:.4f}")
    print(f"GNN f1_score: {f1:.4f}")
    
    return auc_gnn, precision_gnn, a_p, recall_gnn, f1

def node_probability(g1, g2, g3, idx_train, y_train, w1, w2):
    node_degree = (g1.in_degrees() + g2.in_degrees() + g3.in_degrees()).clamp(min=1)
    node_degree = torch.pow(node_degree, w1)
    node_degree = np.array(node_degree.cpu())
    fraud_rate = torch.tensor(np.sum(y_train)/len(y_train))
    fraud_rate = torch.pow(fraud_rate, w2)
    norm_rate = 1 - fraud_rate
    node_prob = np.zeros(len(y_train))
    for i in range(len(y_train)):
        if y_train[i] == 1:
            node_prob[i] = node_degree[i]/fraud_rate
        elif y_train[i] == 0:
            node_prob[i] = node_degree[i]/norm_rate
    node_prob = node_prob/np.sum(node_prob)
    return node_prob
    
def edge_probability(g1, idx_train, labels, w1, w2):
    node_degree = g1.in_degrees().clamp(min=1)
    node_degree = np.array(torch.pow(node_degree, w1).cpu())
    edges_source = np.array(g1.edges()[0].cpu())
    edges_target = np.array(g1.edges()[1].cpu())
    
    index_mask = np.zeros(len(labels))
    index_mask[idx_train]=1
    edges_index = []
    edges_label = np.zeros(len(edges_source))
    edges_label_1 = []
    T11, T00, Y01=0.,0.,0.

    edges_probability =[]

    for i in range(len(edges_source)):
        source_temp = edges_source[i]
        target_temp = edges_target[i]
        if (index_mask[source_temp]==1) and (index_mask[target_temp]==1):
            edges_index.append(i)
            edges_probability.append(node_degree[source_temp]*node_degree[target_temp])
            if labels[source_temp]^labels[target_temp]:
                edges_label_1.append(0)
                Y01 = Y01 + 1
            elif labels[source_temp] == 1:
                edges_label[i] = 1
                edges_label_1.append(1)
                T11 = T11 + 1
            elif labels[source_temp] == 0:
                edges_label[i] = 1
                T00 = T00 + 1
                edges_label_1.append(2)
            else:
                print('edges_label is wrong!')
    T11, T00, Y01 = np.power(T11, w2), np.power(T00, w2), np.power(Y01, w2) 
    T11_rate, T00_rate, Y01_rate= T11/(Y01+T11+T00), T00/(Y01+T11+T00), Y01/(Y01+T11+T00)
    
    for i in range(len(edges_probability)):
        if edges_label_1[i] == 0:  #Y01_rate
            edges_probability[i] = edges_probability[i]/Y01_rate
        elif edges_label_1[i] == 1:  #T11_rate
            edges_probability[i] = edges_probability[i]/T11_rate
        elif edges_label_1[i] ==2:
            edges_probability[i] = edges_probability[i]/T00_rate
    
    edges_probability = edges_probability/np.sum(edges_probability)
    edges_label = torch.LongTensor(edges_label)
    return edges_index, edges_label, edges_probability
    
            
            
    
            
    
    
                
                
                
                
                