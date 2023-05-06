# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:19:21 2022

@author: Administrator
"""

import torch
import dgl
import numpy as np
import argparse
from dgl.data import FraudAmazonDataset
from dgl.data import FraudYelpDataset
from sklearn.model_selection import train_test_split
from new_utlis_2 import normalize_row, normalize_col, test_model, node_probability, edge_probability
from new_model_2 import Model
# from new_model_1 import Model
import random
from torch.autograd import Variable
import os
import time, datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


'''
 Yelp                                           Amazon 
 |  - Nodes: 45,954                             |  - Nodes: 11,944
 |  - Edges:                                    |  - Edges:
 |                                              |  
 |      - R-U-R: 98,630                         |      - U-P-U: 351,216
 |      - R-T-R: 1,147,232                      |      - U-S-U: 7,132,958
 |      - R-S-R: 6,805,486                      |      - U-V-U: 2,073,474
 |                                              |  
 |  - Classes:                                  |  - Classes:
 |                                              |                     
 |      - Positive (spam): 6,677                |      - Positive (fraudulent): 821
 |      - Negative (legitimate): 39,277         |      - Negative (benign): 7,818
 |                                              |      - Unlabeled: 3,305
 |                                              |
 |  - Positive-Negative ratio: 1 : 5.9          |  - Positive-Negative ratio: 1 : 10.5
 |  - Node feature size: 32                     |  - Node feature size: 25
'''
parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='amazon', help='The dataset name: [ amazon]')
parser.add_argument('--normalization', type=str, default='row', help='Nomalization for features: [row, col]')
parser.add_argument('--hidden_dim', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout_rate', type=float, default=0)
parser.add_argument('--eps', type=float, default=0.5, help='Fixed scalar or learnable weight.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=70, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default = 256)
parser.add_argument('--test_epochs', type=int, default = 5, help='Epoch interval to run test set.')
parser.add_argument('--test_rate', type=float, default = 0.6, help='Epoch interval to run test set.')
parser.add_argument('--validate_rate', type=float, default = 0.67, help='Epoch interval to run test set.')
parser.add_argument('--loss_balance', type=float, default = 0.4, help='Epoch interval to run test set.')

#b matter
parser.add_argument('--matter', type=str, default='eps', help='The dataset name: [ amazon]')


args = parser.parse_args()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# 1、构建子图；2、特征初始化；3、提取标签label
if args.data == 'yelp':
    dataset = FraudYelpDataset()
    graph = dataset[0]
    
    # feature normalization
    features = graph.ndata['feature']
    if args.normalization == 'row':
        features = normalize_row(features)
    elif args.normalization == 'col':
        features = normalize_col(features)
    
    features = torch.FloatTensor(features).to(device)
    labels = graph.ndata['label']
    number = graph.num_nodes()
    
    g1 = dgl.graph(graph['net_rur'].edges(), num_nodes = number).to(device)
    g2 = dgl.graph(graph['net_rtr'].edges(), num_nodes = number).to(device)
    g3 = dgl.graph(graph['net_rsr'].edges(), num_nodes = number).to(device)
    
elif args.data == 'amazon':
    dataset = FraudAmazonDataset()
    graph = dataset[0]
    
    # feature normalization
    features = graph.ndata['feature']
    if args.normalization == 'row':
        features = normalize_row(features)
    elif args.normalization == 'col':
        features = normalize_col(features)
        
    features = torch.FloatTensor(features).to(device)
    labels = graph.ndata['label']
    number = graph.num_nodes()
    
    g1 = dgl.graph(graph['net_upu'].edges(), num_nodes = number).to(device)
    g2 = dgl.graph(graph['net_usu'].edges(), num_nodes = number).to(device)
    g3 = dgl.graph(graph['net_uvu'].edges(), num_nodes = number).to(device)

# 数据划分
if args.data == 'yelp':
    index = list(range(len(labels)))
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, np.array(labels), stratify = np.array(labels), 
                                                            test_size = args.test_rate, random_state = 2, shuffle = True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify = y_rest, 
                                                            test_size = args.validate_rate, random_state = 2, shuffle = True)   
    
elif args.data == 'amazon':
    index = list(range(3305, len(labels)))
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, np.array(labels)[3305:], stratify = np.array(labels)[3305:],
                                                            test_size = args.test_rate, random_state = 2, shuffle = True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify = y_rest, 
                                                            test_size = args.validate_rate, random_state = 2, shuffle = True) 
    
# 度的归一化
g_list = [g1, g2,g3]
name = 'self_loop'
if name == 'self_loop':
    for g in g_list:
        deg = g.in_degrees().float().to(device)
        deg = deg + torch.ones(len(deg)).to(device)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm
else:
    for g in g_list:
        deg = g.in_degrees().float().clamp(min=1).to(device)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm

node_prob = node_probability(g1, g2, g3, idx_train, y_train, w1=0.3, w2=0.5)
e_index1, e_label1, e_prob1 = edge_probability(g1, idx_train, labels, w1=0.3, w2=0.5)
e_index2, e_label2, e_prob2 = edge_probability(g2, idx_train, labels, w1=0.3, w2=0.5)
e_index3, e_label3, e_prob3 = edge_probability(g3, idx_train, labels, w1=0.3, w2=0.5)

print
n_labels = labels.to(device)
e_label1 = e_label1.to(device)
e_label2 = e_label2.to(device)
e_label3 = e_label3.to(device)

net = Model(g1, g2, g2, features, features.size()[1], args.hidden_dim, 2, args.dropout_rate, args.eps).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

timestamp = time.time()
timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
dir_saver = './model_parms/'+timestamp

path_saver = os.path.join(dir_saver, '{}_{}_{}.pkl'.format(args.matter, args.eps, args.data))
epoch_best = 0
overall_time = 0.
performance_log = []
auc_best = 0
for epoch in range(args.epochs):
    num_batches = int(len(idx_train)/args.batch_size) + 1   
    loss_all = 0.
    start_time = time.time()
    net.train()
    
    for batch in range(num_batches):
        batch_n0 = np.random.choice(a=idx_train, size=args.batch_size, replace=True, p=node_prob).tolist()
        batch_e1 = np.random.choice(a=e_index1, size=args.batch_size, replace=True, p=e_prob1).tolist()
        batch_e2 = np.random.choice(a=e_index2, size=args.batch_size, replace=True, p=e_prob2).tolist()
        batch_e3 = np.random.choice(a=e_index3, size=args.batch_size, replace=True, p=e_prob3).tolist()
        
        label_n0 = n_labels[np.array(batch_n0)]
        label_e1 = e_label1[np.array(batch_e1)]
        label_e2 = e_label2[np.array(batch_e2)]
        label_e3 = e_label3[np.array(batch_e3)]
        
        optimizer.zero_grad()
        
        if torch.cuda.is_available():
            label_n0 = Variable(torch.cuda.LongTensor(label_n0))
            label_e1 = Variable(torch.cuda.LongTensor(label_e1))
            label_e2 = Variable(torch.cuda.LongTensor(label_e2))
            label_e3 = Variable(torch.cuda.LongTensor(label_e3))
        else:
            label_n0 = Variable(torch.LongTensor(label_n0))
            label_e1 = Variable(torch.LongTensor(label_e1))
            label_e2 = Variable(torch.LongTensor(label_e2))
            label_e3 = Variable(torch.LongTensor(label_e3))
        
        #                list       list      list      list   torch.int64
        loss = net.loss(batch_n0, batch_e1, batch_e2, batch_e3, label_n0, label_e1, label_e2, label_e3)

            
        loss.backward()
        optimizer.step()
        loss_all =loss_all + loss.item()
        
    end_time = time.time()
    epoch_time = end_time - start_time
    
    print(f'Epoch: {epoch}, loss: {loss_all / num_batches}')
    overall_time += epoch_time
    
    if (epoch+1) % args.test_epochs == 0:
        net.eval()
        auc_val, precision_val, a_p_val, recall_val, f1_val = test_model(idx_valid, y_valid, net)
        if auc_val > auc_best:
            epoch_best = epoch
            if not os.path.exists(dir_saver):
                os.makedirs(dir_saver)
            print('****************Saving model ...*****************')
            torch.save(net.state_dict(), path_saver)
    
    if (epoch+1) % args.test_epochs == 0:
        print('Test is comming:')
        net.eval()
        auc, precision, a_p, recall, f1 = test_model(idx_test, y_test, net)
        performance_log.append([auc, precision, a_p, recall, f1])

print("The training time:", overall_time)
print("The training time per epoch:",overall_time/args.epochs)

print('new_train_1')
for k in list(vars(args).keys()):
    print('%s: %s'%(k, vars(args)[k]))	
print('performance_log', performance_log)	

print("Restore model from epoch {}".format(epoch_best))
print("Model path: {}".format(path_saver))
net.load_state_dict(torch.load(path_saver))
auc, precision, a_p, recall, f1 = test_model(idx_test, y_test, net)
print('Final test result:')
print(f"GNN auc: {auc:.4f}")
print(f"GNN Recall: {recall:.4f}")
print(f"GNN f1_score: {f1:.4f}")

        
    















    
    
    