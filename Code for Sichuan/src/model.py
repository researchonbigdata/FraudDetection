import argparse
import os

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv, GraphConv
from layers import DualChannel, MultiLstm
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset


class BD_BGL(nn.Module):
    def __init__(self, args, edge_index, label) -> None:
        super(BD_BGL, self).__init__()
        self.module1 = MultiLstm(args)
        self.module2 = DualChannel(args.hidden_size*4+args.id_feature_size, args.hidden_size, 2,
                                   edge_index, label, dropout=args.dropout, eps=args.eps, highlow=args.highlow)

    def forward(self, oneweek, twoweek, threeweek, month, id_feature):
        multilstm_pre, pre2 = self.module1(
            oneweek, twoweek, threeweek, month, id_feature,)
        dual_pre = self.module2(pre2)
        return dual_pre, multilstm_pre
