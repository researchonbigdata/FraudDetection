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
from new_utlis_1 import normalize_row, normalize_col, test_model, node_probability
from new_model_1 import Model
# from new_model_1 import Model
import random
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
from sklearn import manifold


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

parser.add_argument('--data', type=str, default='yelp', help='The dataset name: [ amazon,yelp]')
parser.add_argument('--normalization', type=str, default='row', help='Nomalization for features: [row, col]')
parser.add_argument('--hidden_dim', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout_rate', type=float, default=0)
parser.add_argument('--eps', type=float, default=0.5, help='Fixed scalar or learnable weight.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=80, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default = 512)
parser.add_argument('--test_epochs', type=int, default = 5, help='Epoch interval to run test set.')
parser.add_argument('--test_rate', type=float, default = 0.6, help='Epoch interval to run test set.')

args = parser.parse_args()

def node_sample_TSNE(test_label, sample_rate_fraud):
    sample_num = int(np.sum(test_label)*sample_rate_fraud)
    index_fraud = np.where(test_label == 1)[0]
    index_normal= np.where(test_label == 0)[0]
    np.random.seed(1)
    sample_index_fraud = np.random.choice(index_fraud, sample_num)
    np.random.seed(1)
    sample_index_normal= np.random.choice(index_normal,sample_num)
    sample_index_all = np.hstack([sample_index_fraud, sample_index_normal])
    np.random.shuffle(sample_index_all) 
    
    return sample_index_all

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
    idx_train, idx_test, y_train, y_test = train_test_split(index, np.array(labels), stratify = np.array(labels), 
                                                            test_size = args.test_rate, random_state = 2, shuffle = True)
    
elif args.data == 'amazon':
    index = list(range(3305, len(labels)))
    idx_train, idx_test, y_train, y_test = train_test_split(index, np.array(labels)[3305:], stratify = np.array(labels)[3305:],
                                                         test_size = args.test_rate, random_state = 2, shuffle = True)
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

node_prob = node_probability(g1, g2, g3, idx_train, y_train, w=0.3)

labels = labels.to(device)
net = Model(g1, g2, g2, features, features.size()[1], args.hidden_dim, 2, args.dropout_rate, args.eps).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

overall_time = 0.
performance_log = []
for epoch in range(args.epochs):
    num_batches = int(len(idx_train)/args.batch_size) + 1   
    loss_all = 0.
    start_time = time.time()
    net.train()
    
    for batch in range(num_batches):
        batch_nodes = np.random.choice(a=idx_train, size=args.batch_size, replace=True, p=node_prob).tolist()
        batch_label = labels[np.array(batch_nodes)]
        optimizer.zero_grad()
        
        if torch.cuda.is_available():
            loss = net.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
        else:
            loss = net.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
            
        loss.backward()
        optimizer.step()
        loss_all =loss_all + loss.item()
        
    end_time = time.time()
    epoch_time = end_time - start_time
    
    print(f'Epoch: {epoch}, loss: {loss_all / num_batches}')
    overall_time += epoch_time
    
    if (epoch+1) % args.test_epochs == 0:
        print('Test is comming:')
        net.eval()
        auc, precision, a_p, recall, f1 = test_model(idx_test, y_test, net)
        performance_log.append([auc, precision, a_p, recall, f1])

        

# TSNE 
net.eval()
node_embedding = net.feat_back(idx_test).data.cpu().numpy()

sample_rate_fraud = 0.2
sample_index_all = node_sample_TSNE(y_test, sample_rate_fraud)

test_label = y_test[sample_index_all]
test_case_feature = node_embedding[sample_index_all]
tsne = manifold.TSNE(n_components=2, init='pca')
feat_tsne = tsne.fit_transform(test_case_feature)

plt.scatter(feat_tsne[:, 0], feat_tsne[:, 1], c=test_label, cmap='brg', alpha=1, s=2)
plt.savefig('TSNE_'+args.data+'_40'+ 'IDGL' + '.pdf', bbox_inches='tight')
plt.show()

print("The training time:", overall_time)
print("The training time per epoch:",overall_time/args.epochs)

print('new_train_1')
for k in list(vars(args).keys()):
    print('%s: %s'%(k, vars(args)[k]))		
print('performance_log', performance_log)	 
    
    
    















    
    
    