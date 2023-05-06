# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:57:27 2022

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
import numpy as np
from new_utlis_1 import *
    
class Layer(nn.Module):
    def __init__(self, g, in_dim, dropout_rate):
        super(Layer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(dropout_rate)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        a = torch.tanh(self.gate(h2)).squeeze()
        e = a * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)
        return {'e': e, 'm': a}

    #h表示为特征向量——feature
    def forward(self, h):
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']       
        
class Model(nn.Module):
    def __init__(self, g1, g2, g3, h, in_dim, hidden_dim, out_dim, dropout_rate, eps):
        #                graph,       25/32,     32,        2,        0.5,       0.5
        super(Model, self).__init__()
        self.g1 = g1
        self.g2 = g2
        self.g3 = g3
        
        self.h = h
        self.eps = eps
        self.dropout_rate = dropout_rate
        self.xent = nn.CrossEntropyLoss()       # 交叉熵损失函数
        self.fun = nn.LeakyReLU(0.3)            # 激活函数
        self.dropout = nn.Dropout(dropout_rate) # dropout函数
        
        # For layer1
        self.layer1_1 = Layer(self.g1, hidden_dim, dropout_rate)
        self.layer1_2 = Layer(self.g2, hidden_dim, dropout_rate)
        self.layer1_3 = Layer(self.g3, hidden_dim, dropout_rate)
        self.hw1_1 = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        self.hw1_2 = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        self.hw1_3 = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        nn.init.xavier_normal_(self.hw1_1, gain=1.414)
        nn.init.xavier_normal_(self.hw1_2, gain=1.414)
        nn.init.xavier_normal_(self.hw1_3, gain=1.414)
        
        # For layer2
        self.layer2_1 = Layer(self.g1, hidden_dim*3, dropout_rate)
        self.layer2_2 = Layer(self.g2, hidden_dim*3, dropout_rate)
        self.layer2_3 = Layer(self.g3, hidden_dim*3, dropout_rate)
        self.hw2_1 = nn.Parameter(torch.FloatTensor(hidden_dim*3, hidden_dim))
        self.hw2_2 = nn.Parameter(torch.FloatTensor(hidden_dim*3, hidden_dim))
        self.hw2_3 = nn.Parameter(torch.FloatTensor(hidden_dim*3, hidden_dim))
        nn.init.xavier_normal_(self.hw2_1, gain=1.414)
        nn.init.xavier_normal_(self.hw2_2, gain=1.414)
        nn.init.xavier_normal_(self.hw2_3, gain=1.414)
        
        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.t2 = nn.Linear(7*hidden_dim + in_dim, 64)
        self.t3 = nn.Linear(64, out_dim)
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)
        nn.init.xavier_normal_(self.t3.weight, gain=1.414)
        
        
    def forward(self, nodes):
#        h = F.dropout(h, p=self.dropout, training=self.training)
#        h = torch.relu(self.t1(h))
#        h = F.dropout(h, p=self.dropout, training=self.training)
        # 需要修改
        h = self.h
        raw0 = h
        h = self.dropout(h)
        h = self.fun(self.t1(h))
#        h = self.dropout(h)  
        
        raw1 = h
        
        # for the layer1  
        h1_1 = self.fun(torch.mm(self.eps*raw1 + self.layer1_1(h), self.hw1_1))
        h1_2 = self.fun(torch.mm(self.eps*raw1 + self.layer1_2(h), self.hw1_2))
        h1_3 = self.fun(torch.mm(self.eps*raw1 + self.layer1_3(h), self.hw1_3))
        
        # aggregation
        h = torch.cat((h1_1, h1_2, h1_3), dim = 1)
        raw2 = h
        
        # for the layer2
        h2_1 = self.fun(torch.mm(self.eps*raw2 + self.layer2_1(h), self.hw2_1))
        h2_2 = self.fun(torch.mm(self.eps*raw2 + self.layer2_2(h), self.hw2_2))
        h2_3 = self.fun(torch.mm(self.eps*raw2 + self.layer2_3(h), self.hw2_3))
        
        # aggregation and classification
#        h = torch.cat((h2_1, h2_2, h2_3, raw0, raw1, raw2), dim = 1)
        h = torch.cat((h2_1, h2_2, h2_3, raw0, raw1, raw2), dim = 1)
        
        if isinstance(nodes, list):
            node_index = torch.LongTensor(nodes)
        else:
            print('The type of node_index is wrong!')
        node_feature = h[node_index]
        
        scores_model_64 = self.fun(self.t2(node_feature))
        scores_model = self.t3(scores_model_64)
        
        return scores_model, scores_model_64
        
    def to_prob(self, nodes):
        scores_model, scores_model_64 = self.forward(nodes)
        scores_model = torch.sigmoid(scores_model)
        return scores_model
    
    def feat_back(self, nodes):
        scores_model, scores_model_64 = self.forward(nodes)
        return scores_model_64
        
    def loss(self, nodes, labels):
        scores_model, scores_model_64 = self.forward(nodes)
        model_loss = self.xent(scores_model, labels.squeeze())
        
        return model_loss
        
        
            
            
        
        
        
        
        
        
        
        
                  
    
        
    