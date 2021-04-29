import torch
import os
import numpy

class Namespace(object):
    def __init__(self,adict):
        self.__dict__.update(adict)

class bitcoin():
    def __init__(self):
        self.ecols=Namespace({'FromNodeId':0,
                              'ToNodeId':1,
                              'Weight':2,
                              'TimeStep':3
                              })

    def load_data(self,file):
        with open(file) as file:
            file=file.read().splitlines()
        data=torch.tensor([[float(r) for r in row.split(',')] for row in file])
        data=torch.tensor(data,dtype=torch.long)
        return data

    def make_continous_node_ids(self,edges):
        new_edges=edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]] = new_edges
        return edges


    def get_num_nodes(self,edges):
        all_ids = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        num_nodes = all_ids.max() + 1
        return num_nodes

def aggregate_by_time(time_vector,time_win_aggr):
    time_vector = time_vector - time_vector.min()
    time_vector = time_vector // time_win_aggr
    return time_vector

def cluster_negs_and_pos(ratings):
    pos_indices=ratings>0
    neg_indices=ratings<=0
    ratings[pos_indices]=1
    ratings[neg_indices]=-1
    return ratings

if __name__=='__main__':
    file='./data_net/soc-sign-bitcoinotc.csv'
    dataset=bitcoin()
    data=dataset.load_data(file)
    # print(data)
    edges=dataset.make_continous_node_ids(data)
    print(edges)
    num_nodes=edges[:,[dataset.ecols.FromNodeId,
                    dataset.ecols.ToNodeId]].unique().size(0)
    print('num:',num_nodes)

    timesteps=aggregate_by_time(edges[:,dataset.ecols.TimeStep],1200000)
    print(timesteps)
    max_time=timesteps.max()
    min_time=timesteps.min()
    print('maxtime:',max_time)
    print('mintime',min_time)
    edges[:,dataset.ecols.TimeStep]=timesteps
    edges[:,dataset.ecols.Weight]=cluster_negs_and_pos(edges[:,dataset.ecols.Weight])
    print(edges)
    # make undirected
    edges=torch.cat([edges,edges[:,[dataset.ecols.ToNodeId,
                                    dataset.ecols.FromNodeId,
                                    dataset.ecols.Weight,
                                    dataset.ecols.TimeStep]]])
    print(edges)

    # separate classes
    sp_indices=edges[:,[dataset.ecols.FromNodeId,
                        dataset.ecols.ToNodeId,
                        dataset.ecols.TimeStep]].t()
    sp_values=edges[:,dataset.ecols.Weight]

    print(sp_indices)
    # print(sp_values)
    neg_mask=sp_values==-1
    print('neg',neg_mask)

    neg_sp_indices=sp_indices[:,neg_mask]
    neg_sp_values=sp_values[neg_mask]
    neg_sp_edges=torch.sparse.LongTensor(neg_sp_indices,
                                         neg_sp_values,
                                         torch.Size([num_nodes,
                                                     num_nodes,
                                                     max_time+1])).coalesce() #torch.coalesce()方法返回(indices,values)两个值

    print('indices:',neg_sp_indices) #负向边标记
    print('values',neg_sp_values) #负向边值
    print('edges',neg_sp_edges) #负向边对

    pos_mask=sp_values==1

    pos_sp_indices=sp_indices[:,pos_mask]
    pos_sp_values=sp_values[pos_mask]
    pos_sp_edges=torch.sparse.LongTensor(pos_sp_indices,
                                         pos_sp_values,
                                         torch.Size([num_nodes,
                                                     num_nodes,
                                                     max_time+1])).coalesce() #在三维空间(n,n,t)中构建稀疏张量

    # print('before',pos_sp_edges)
    pos_sp_edges *=1000
    print('after pos:',pos_sp_edges)

    sp_edges=(pos_sp_edges-neg_sp_edges).coalesce()

    print('sp_edges:',sp_edges) #无向图，负向评价的边变成了正值,正向边是1000,负向边是1

    vals=sp_edges._values()

    neg_vals=vals%1000
    pos_vals=vals//1000

    print(neg_vals)
    print(pos_vals)

    vals=pos_vals-neg_vals
    print(vals) #这里又变成了1，-1的值

    new_vals = torch.zeros(vals.size(0), dtype=torch.long)
    new_vals[vals > 0] = 1
    new_vals[vals <= 0] = 0
    print('vals shape',new_vals.shape)
    indices_labels=torch.cat([sp_edges._indices().t(),new_vals.view(-1,1)],dim=1) #按列拼接
    print(indices_labels)

    vals=pos_vals+neg_vals #所有的评论和
    print(vals)

    edges={'idx':indices_labels,'vals':vals}

















