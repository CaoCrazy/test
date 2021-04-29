import torch
import taskers_utils as tu
import utils as u
import numpy as np
import random
class Link_Pred_Tasker():
	'''
	Creates a tasker object which computes the required inputs for training on a link prediction
	task. It receives a dataset object which should have two attributes: nodes_feats and edges, this
	makes the tasker independent of the dataset being used (as long as mentioned attributes have the same
	structure).

	Based on the dataset it implements the get_sample function required by edge_cls_trainer.
	This is a dictionary with:
		- time_step: the time_step of the prediction
		- hist_adj_list: the input adjacency matrices until t, each element of the list 
						 is a sparse tensor with the current edges. For link_pred they're
						 unweighted
		- nodes_feats_list: the input nodes for the GCN models, each element of the list is a tensor
						  two dimmensions: node_idx and node_feats
		- label_adj: a sparse representation of the target edges. A dict with two keys: idx: M by 2 
					 matrix with the indices of the nodes conforming each edge, vals: 1 if the node exists
					 , 0 if it doesn't

	There's a test difference in the behavior, on test (or development), the number of sampled non existing 
	edges should be higher.
	'''
	def __init__(self,args,dataset):
		self.data = dataset
		#max_time for link pred should be one before
		self.max_time = dataset.max_time - 1
		self.args = args
		self.num_classes = 2

		if not (args.use_2_hot_node_feats or args.use_1_hot_node_feats):
			self.feats_per_node = dataset.feats_per_node

		# self.get_node_feats = self.build_get_node_feats(args,idx,dataset)
		self.prepare_node_feats = self.build_prepare_node_feats(args,dataset)
		self.is_static = True
		
		'''TO CREATE THE CSV DATASET TO USE IN DynGEM
		print ('min max time:', self.data.min_time, self.data.max_time)
		file = open('data/autonomous_syst100_adj.csv','w')
		file.write ('source,target,weight,time\n')
		for time in range(self.data.min_time, self.data.max_time):
			adj_t = tu.get_sp_adj(edges = self.data.edges,
					   time = time,
					   weighted = True,
					   time_window = 1)
			#node_feats = self.get_node_feats(adj_t)
			print (time, len(adj_t))
			idx = adj_t['idx']
			vals = adj_t['vals']
			num_nodes = self.data.num_nodes
			sp_tensor = torch.sparse.FloatTensor(idx.t(),vals.type(torch.float),torch.Size([num_nodes,num_nodes]))
			dense_tensor = sp_tensor.to_dense()
			idx = sp_tensor._indices()
			for i in range(idx.size()[1]):
				i0=idx[0,i]
				i1=idx[1,i]
				w = dense_tensor[i0,i1]
				file.write(str(i0.item())+','+str(i1.item())+','+str(w.item())+','+str(time)+'\n')

			#for i, v in zip(idx, vals):
			#	file.write(str(i[0].item())+','+str(i[1].item())+','+str(v.item())+','+str(time)+'\n')

		file.close()
		exit'''

	# def build_get_non_existing(args):
	# 	if args.use_smart_neg_sampling:
	# 	else:
	# 		return tu.get_non_existing_edges
	def getdata(self):
		return self.data

	def build_prepare_node_feats(self,args,dataset):
		if args.use_2_hot_node_feats or args.use_1_hot_node_feats:
			def prepare_node_feats(node_feats):
				return u.sparse_prepare_tensor(node_feats,
											   torch_size= [dataset.num_nodes,
											   				self.feats_per_node])
		else:
			prepare_node_feats = self.data.prepare_node_feats

		return prepare_node_feats


	def build_get_node_feats(self,args,idx,dataset):
		if args.use_2_hot_node_feats:
			max_deg_out, max_deg_in = tu.get_max_degs(args,dataset)
			self.feats_per_node = max_deg_out + max_deg_in
			def get_node_feats(adj):
				return tu.get_2_hot_deg_feats(adj,
											  max_deg_out,
											  max_deg_in,
											  dataset.num_nodes)
		elif args.use_1_hot_node_feats:
			max_deg,_ = tu.get_max_degs(args,idx,dataset)
			self.feats_per_node = max_deg
			def get_node_feats(idx,adj):
				return tu.get_1_hot_deg_feats(adj,
											  max_deg,
											   self.data.num_nodes)
		else:
			def get_node_feats(adj):
				return dataset.nodes_feats

		return get_node_feats


	def get_sample(self,idx,test, **kwargs):
		
		hist_adj_list = []
		hist_ndFeats_list = []
		hist_mask_list = []
		existing_nodes = []
		label_adj = []
		cov = torch.zeros(0)
		t = []
		# # print('wwwwwwwwwwwwwwwwwwwww')
		# return res
		
		# self.get_node_feats = self.build_get_node_feats(args,idx,dataset)
		print('*'*10,idx)
		idx = int(idx)
		times = self.data.sampling[idx][:,2][0:300]
		
		times_ptr = self.data.sampling[idx][:,1]
		if times.size()!=torch.Size([0]):
			index = times.max()
			# times = times[index]
			# times = times.tolist()
			# cov = torch.zeros([len(hist_ndFeats_list),1],dtype = torch.float)
			
			for j,i in enumerate(self.data.sampling[idx][:,1][0:300] ):
				if times[j] == index:
					hist_ndFeats_list.append(torch.zeros([1,self.data.nodes_feats[i].size(0)],dtype=torch.float)[0].tolist())
				else:
					hist_ndFeats_list.append(self.data.nodes_feats[i].tolist())
				
			t = times==times.max()
			label_adj = tu.get_edge_labels(edges = self.data.sampling[idx][0:300], 
										time = torch.tensor(times))
			times = times.tolist()
			cov = torch.zeros([len(hist_ndFeats_list),1],dtype = torch.float)
		
		return {'idx': idx,
				'hist_ndFeats_list': hist_ndFeats_list,
				'times':times,
				'time_ptr':times_ptr,
				'label_sp':label_adj,
				'cov':cov,
				't':t
				}

