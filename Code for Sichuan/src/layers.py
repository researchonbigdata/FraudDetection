import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch.nn.parameter import Parameter

class MultiLstm(nn.Module):
    def __init__(self, arg) -> None:
        super().__init__()
        # self.device=torch.device(arg.device)
        self.embedding = Parameter(torch.rand(
            size=(1, arg.hidden_size),), requires_grad=True)
        self.feature_attn = Parameter(torch.rand(
            size=(arg.num_features,),), requires_grad=True)
        self.weekLstm = nn.LSTM(
            input_size=arg.hidden_size, hidden_size=arg.hidden_size, batch_first=True)
        self.twoweekLstm = nn.LSTM(
            input_size=arg.hidden_size, hidden_size=arg.hidden_size, batch_first=True)
        self.threeweekLstm = nn.LSTM(
            input_size=arg.hidden_size, hidden_size=arg.hidden_size, batch_first=True)
        self.monthLstm = nn.LSTM(
            input_size=arg.hidden_size, hidden_size=arg.hidden_size, batch_first=True)
        self.weekmonthattn = Parameter(
            torch.rand(size=(4,)), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)
        self.clf = nn.Linear(arg.hidden_size*4+arg.id_feature_size, 2)
        # self.fagcn=FAGCN()

    def forward(self, input_week, input_twoweek, input_threeweek, input_month, id_feature=None):
        input_week = torch.unsqueeze(input_week, dim=-1)
        input_twoweek = torch.unsqueeze(input_twoweek, dim=-1)
        input_threeweek = torch.unsqueeze(input_threeweek, dim=-1)
        input_month = torch.unsqueeze(input_month, dim=-1)
        input_week = torch.matmul(input_week, self.embedding)
        input_twoweek = input_twoweek@self.embedding
        input_threeweek = input_threeweek@self.embedding
        input_month = input_month@self.embedding
        input_week = torch.permute(input_week, dims=(
            0, 1, 3, 2))@self.softmax(self.feature_attn)
        input_twoweek = torch.permute(input_twoweek, dims=(
            0, 1, 3, 2))@self.softmax(self.feature_attn)
        input_threeweek = torch.permute(input_threeweek, dims=(
            0, 1, 3, 2))@self.softmax(self.feature_attn)
        input_month = torch.permute(input_month, dims=(
            0, 1, 3, 2))@self.softmax(self.feature_attn)
        output, (final_week, _) = self.weekLstm(input_week)
        output2, (final_twoweek, _) = self.weekLstm(input_twoweek)
        output3, (final_threeweek, _) = self.weekLstm(input_threeweek)
        output4, (final_month, _) = self.weekLstm(input_month)
        final_week = final_week[0, :, :]
        final_twoweek = final_twoweek[0, :, :]
        final_threeweek = final_threeweek[0, :, :]
        final_month = final_month[0, :, :]
        # final=torch.stack([final_week,final_twoweek,final_threeweek,final_month],dim=-1)@self.weekmonthattn
        final = torch.concat(
            [final_week, final_twoweek, final_threeweek, final_month, id_feature], dim=-1)
        final2 = self.clf(final)
        final2 = self.softmax(final2)
        return final2, final

        # return final

    def get_fea_attn(self):
        return self.softmax(self.feature_attn)

    def get_timeattn(self):
        return self.softmax(self.weekmonthattn)

class DualChannelLayer(MessagePassing):
    def __init__(self, num_hidden,edge_index, label,dropout,highlow):
        super(DualChannelLayer, self).__init__(aggr='add')
        # self.data = data
        self.edge_index=edge_index
        self.label=label
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * num_hidden, 1)
        self.row, self.col = edge_index
        self.norm_degree = degree(self.row, num_nodes=label.shape[0]).clamp(min=1)
        self.norm_degree = torch.pow(self.norm_degree, -0.5)
        self.highlow=highlow
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def forward(self, h):
        h2 = torch.cat([h[self.row], h[self.col]], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        if self.highlow==1:
            temp=torch.ones_like(g,dtype=g.dtype).to(g.device)
            g=temp
        elif self.highlow==-1:
            temp=torch.ones_like(g,dtype=g.dtype).to(g.device)
            temp=-temp
            g=temp
        norm = g * self.norm_degree[self.row] * self.norm_degree[self.col]
        norm = self.dropout(norm)  
        return self.propagate(self.edge_index, size=(h.size(0), h.size(0)), x=h, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1,1) * x_j

    def update(self, aggr_out):
        return aggr_out




class DualChannel(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes,edge_index,labels, dropout, eps, layer_num=2,highlow=0):
        super(DualChannel, self).__init__()
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(DualChannelLayer( num_hidden, edge_index,labels,dropout,highlow))
        self.t1 = nn.Linear(num_features, num_hidden)
        self.t2 = nn.Linear(num_hidden, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = self.eps * raw + h
        h = self.t2(h)
        return F.log_softmax(h, 1)