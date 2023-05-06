
import numpy as np
import pandas as pd
import time as timee
from contextlib import contextmanager
from sklearn.model_selection import train_test_split
import networkx as nx
import features as features
from features import connector_duplicate2


@contextmanager
def timer(title):
    t0 = timee.time()
    yield
    print(f"{title} - done in {timee.time()-t0}s")


def normalize(mx: np.ndarray):  # 均值方差归一化
    shape = mx.shape
    mx = mx.reshape((-1, shape[-1]))
    for k in range(mx.shape[-1]):
        mx[:, k] = (mx[:, k]-np.mean(mx[:, k]))/np.std(mx[:, k])
    mx = mx.reshape(shape)
    return mx
def main_certainscale(scale):
    data_path = '/home/m21_huangzijun/pythonprojs/sichuan'
    all_user = pd.read_csv(data_path+'/data/0527/train/train_user.csv')   # 用户信息表

    voc = pd.read_csv(data_path+'/data/0527/train/train_voc.csv')   # 通话记录表

    voc['start_datetime'] = pd.to_datetime(voc['start_datetime'])
    voc['hour'] = voc['start_datetime'].dt.hour

    

    ids = all_user['phone_no_m'].tolist() # 所有用户的id
    user_voc = pd.merge(all_user[['phone_no_m', 'label']],    # 合并两表
                        voc, how='inner', on='phone_no_m')
    
    all_user = all_user.set_index('phone_no_m', drop=False)
    # 行为特征

    # 通话时长平均
    user_voc['mean_dur'] = user_voc.groupby(['phone_no_m', pd.Grouper(key='start_datetime', freq=scale)])[
        'call_dur'].transform(np.nanmean)
    # 通话时长方差
    user_voc['var_dur'] = user_voc.groupby(['phone_no_m', pd.Grouper(key='start_datetime', freq=scale)])[
        'call_dur'].transform(np.nanvar)
    # 精力分散度(和时间片无关)
    user_voc['sum_call_times'] = user_voc.groupby(
        ['phone_no_m'])['phone_no_m'].transform('count')
    user_voc['every_one_calltimes'] = user_voc.groupby(['phone_no_m', 'opposite_no_m'])['phone_no_m'].transform(
        'count')
    user_voc['energy_dispersion'] = user_voc['every_one_calltimes'] / \
        user_voc['sum_call_times']
    del user_voc['sum_call_times']
    del user_voc['every_one_calltimes']

    user_vocs = [g for _, g in user_voc.groupby(
        pd.Grouper(key='start_datetime', freq=scale))]
    nets = [nx.Graph() for i in range(len(user_vocs))]
    
    
    # 每个时间片构一个图
    for idx, net in enumerate(nets):
        net.add_nodes_from(ids)
        tmp = user_vocs[idx]
        for row in tmp.to_dict(orient='records'):  # 每次通话
            calltype_id = row['calltype_id']
            source = row['phone_no_m']
            target = row['opposite_no_m']
            if calltype_id == 1:
                net.add_edge(source, target, weight=1)

            elif calltype_id == 2:
                net.add_edge(source, target, weight=-1)
                # net.addEdge(target,source,1)
    
    
    
    tmp_ids = set(ids.copy())
    slice_feature = dict()
    id_feature = dict()
    each_feature_costtime = []
    # 计算动态特征
    for idx, slice in enumerate(user_vocs):
        ps = [g for _, g in slice.groupby('phone_no_m')]
        net = nets[idx]
        net: nx.Graph
        for p in ps:  # each person in this slice
            p_dict = p.to_dict(orient='list')
            id = p['phone_no_m'].iloc[0]
            # 联系人重复率

            if idx == 0:
                repeat_rate = 0
            else:
                pre_week = user_vocs[idx - 1]
                pre_week = pre_week.loc[pre_week['phone_no_m'] == id]
                repeat_rate = connector_duplicate2(
                    pre_week.to_dict(orient='list'), p_dict)

            # 入度出度
            indegree = 0
            outdegree = 0
            for k, v in net[id].items():
                if v['weight'] == 1:
                    outdegree += 1
                elif v['weight'] == -1:
                    indegree += 1

            # 邻居平均度
            neighbor_degree = list()
            for neigh in net.neighbors(id):
                neighbor_degree.append(net.degree(id))
            neighbor_degree = np.mean(neighbor_degree)

            # 聚类系数
            coefficient = nx.clustering(net, id)

            # 回拨率
            recall_rate = features.recall_rate2(p_dict)
            # 通话时间分布
            time_dis = [0 for i in range(24)]
            for time, count in p['hour'].value_counts(normalize=True).to_dict().items():
                time_dis[time] = count
            # 平均通话时长
            mean_dur = p['mean_dur'].iloc[0]
            # 通话时长方差
            var_dur = p['var_dur'].iloc[0]
            if idx == 0:
                slice_feature[id] = list()
            slice_feature[id].append(
                [indegree, outdegree, neighbor_degree, coefficient, recall_rate, repeat_rate, mean_dur, var_dur] + time_dis)

            tmp_ids.discard(id)
        for id in list(tmp_ids):  # persons not in this slice
            if idx == 0:
                slice_feature[id] = list()
            slice_feature[id] .append([0 for i in range(32)])
        tmp_ids = set(ids.copy())
    print(np.array(each_feature_costtime)*len(user_vocs))

    np_feature = []

    for id in ids:
        np_feature.append(slice_feature[id])
        # np_id_feature.append(id_feature[id])

    np_feature = np.array(np_feature).astype(np.float)

    np_feature[:, :, [0, 1, 2, 3, 6, 7]] = normalize(
        np_feature[:, :, [0, 1, 2, 3, 6, 7]])

    np.save(arr=np_feature, file=data_path +
            f'/data_after/timefreq/x_{scale}.npy')
    
    # 身份特征

    id_feature = dict()

    static_voc = user_voc[['phone_no_m', 'energy_dispersion']].drop_duplicates(subset='phone_no_m').set_index(
        keys='phone_no_m', drop=False)
    for id in list(ids):
        ed = 0
        if id not in static_voc['phone_no_m']:
            ed = 0
        else:
            ed = static_voc.loc[id, 'energy_dispersion']
        idcard_cnt = all_user.loc[id, 'idcard_cnt']

        id_feature[id] = [idcard_cnt, ed]
    np_id_feature = []

    labels = []
    for id in ids:
        np_id_feature.append(id_feature[id])
        labels.append(all_user.loc[id, 'label'])
    np_id_feature = np.array(np_id_feature)
    np_id_feature[:, [0]] = normalize(np_id_feature[:, [0]])
    labels = np.array(labels)
    
    np.save(arr=np_id_feature, file=data_path +
            f'/data_after/timefreq/id_x.npy')
    np.save(arr=labels, file=data_path +
            f'/data_after/timefreq/y.npy')
    np.save(arr=ids, file=data_path +
            f'/data_after/timefreq/ids.npy')
    
    
if __name__=='__main__':
    main_certainscale('D')