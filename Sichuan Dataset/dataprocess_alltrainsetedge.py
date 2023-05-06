import numpy as np
import pandas as pd
from collections import defaultdict

ids=np.load('/home/m21_huangzijun/pythonprojs/sichuan/data_after/mymodel5_2_onemonth/train_ids.npy')
ids2=np.load('/home/m21_huangzijun/pythonprojs/sichuan/data_after/mymodel5_2_onemonth/test_ids.npy')
ids=np.concatenate((ids,ids2))
print(len(ids))
souce_node=[]
target_node=[]
neighbos=defaultdict(list)

voc=pd.read_csv('/home/m21_huangzijun/pythonprojs/sichuan/data/0527/test/test_voc.csv')
voc=voc.to_numpy()


for line in voc:
    source=line[0]
    target=line[1]
    neighbos[source].append(target)

k=4
for i in range(len (ids)-1):
    node1=ids[i]
    if node1 not in neighbos:continue
    neigh1=set(neighbos[node1])
    for j in range(i+1,len(ids)):
        
        node2=ids[j]
        if node2 not in neighbos:continue
        
        neigh2=set(neighbos[node2])
        if len(neigh1.intersection(neigh2))>=k:
            souce_node.append(i)
            target_node.append(j)

np.save('/home/m21_huangzijun/pythonprojs/sichuan/data_after/mymodel5_2_onemonth/source_{}p.npy'.format(k),souce_node)
np.save('/home/m21_huangzijun/pythonprojs/sichuan/data_after/mymodel5_2_onemonth/target_{}p.npy'.format(k),target_node)
