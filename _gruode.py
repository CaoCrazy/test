import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import gru_ode_bayes
import gru_ode_bayes.data_utils as data_utils
import time
import tqdm
from sklearn.metrics import roc_auc_score
# from gru_ode_bayes import Logger
import utils as u
import torch
import torch.distributed as dist
import numpy as np
import time
import random
import datautils
#datasets
import bitcoin_dl as bc
#taskers
import link_pred_tasker as lpt
import edge_cls_tasker as ect
import node_cls_tasker as nct
import reddit_dl as rdt
#models


import splitter as sp
import Cross_Entropy as ce
from sklearn.metrics import average_precision_score
import trainer as tr
import utils as u
import logger

def get_MAP(predictions,true_classes, do_softmax=False):
    if do_softmax:
        probs = torch.softmax(predictions,dim=1)[:,1]
    else:
        probs = predictions

    predictions_np = probs.detach().cpu().numpy()
    true_classes_np = true_classes.detach().cpu().numpy()

    return average_precision_score(true_classes_np, predictions_np)

def prepare_sample(args,splitter,sample):
    sample = u.Namespace(sample)
    for i,adj in enumerate(sample.hist_adj_list):
        adj = u.sparse_prepare_tensor(adj,torch_size = [splitter.get_num()])
        sample.hist_adj_list[i] = adj.to(args.device)
        function = splitter.get_prepare()
        nodes = function(sample.hist_ndFeats_list[i])

        sample.hist_ndFeats_list[i] = nodes.to(args.device)
        # node_mask = sample.node_mask_list[i]
        # sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer

    label_sp = ignore_batch_dim(args,sample.label_sp)
    times = sample.times
    if args.task in ["link_pred", "edge_cls"]:
        label_sp['idx'] = label_sp['idx'].to(args.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
    else:
        label_sp['idx'] = label_sp['idx'].to(args.device)

    label_sp['vals'] = label_sp['vals'].type(torch.long).to(args.device)
    sample.label_sp = label_sp
    sample.times = times

    return sample

def ignore_batch_dim(args,adj):
    if args.task in ["link_pred", "edge_cls"]:
        adj['idx'] = adj['idx'][0]
    adj['vals'] = adj['vals'][0]
    return adj


def train_gruode(simulation_name,params_dict,device, train_idx, val_idx, test_idx, spter,args,epoch_max=40):
    # csv_file_path = params_dict["csv_file_path"]
    # csv_file_cov = params_dict["csv_file_cov"]
    # csv_file_tags = params_dict["csv_file_tags"]

    # N = pd.read_csv(csv_file_path)["ID"].nunique()

    if params_dict["lambda"]==0:
        validation = True
        val_options = {"T_val": params_dict["T_val"], "max_val_samples": params_dict["max_val_samples"]}
    else:
        validation = False
        val_options = None

    # if params_dict["lambda"]==0:
    #     logger = Logger(f'./Logs/{simulation_name}')
    # else:
    #     logger = Logger(f'./Logs/{simulation_name}')


    # data_train = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags, cov_file= csv_file_cov, idx=train_idx)
    # data_val   = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags,
    #                                     cov_file= csv_file_cov, idx=val_idx, validation = validation,
    #                                     val_options = val_options)
    # data_test   = data_utils.ODE_Dataset(csv_file=csv_file_path,label_file=csv_file_tags,
    #                                     cov_file= csv_file_cov, idx=test_idx, validation = validation,
    #                                     val_options = val_options)
   
    # dl,dl_val,dl_test = split_data(db)
    # torch.distributed.init_process_group(backend="nccl")
    dl = spter.gettrain()
    dl_val = spter.getdev()
    dl_test = spter.gettest()
    # data = datautils.ODE_Dataset(db,pg = 'train')
    # dl = data.get_train()
    # dl_val = data.get_vaild()
    # dl_test = data.get_test()
    # dl   = DataLoader(dataset=data_train, collate_fn=datautils.collate_fn, shuffle=True, batch_size=len(data_train),num_workers=2)
    # dl_val = DataLoader(dataset=data_val, collate_fn=datautils.collate_fn, shuffle=True, batch_size=len(data_val))
    # dl_test = DataLoader(dataset=data_test, collate_fn=datautils.collate_fn, shuffle=True, batch_size=len(data_test))
   
    # dl = genrate(dl)
    # dl_val = genrate(dl_val)
    # dl_test = genrate(dl_test)
    params_dict["input_size"]=300
    params_dict["cov_size"] = 1
    # print(params_dict["input_size"])
    # params_dict["input_size"]=torch.tensor(train_num)
    # params_dict["cov_size"] = 1.0
    # np.save(f"./../trained_models/{simulation_name}_params.npy",params_dict)

    nnfwobj = gru_ode_bayes.NNFOwithBayesianJumps(input_size = params_dict["input_size"], hidden_size = params_dict["hidden_size"],
                                                    p_hidden = params_dict["p_hidden"], prep_hidden = params_dict["prep_hidden"],
                                                    logvar = params_dict["logvar"], mixing = params_dict["mixing"],
                                                    classification_hidden=params_dict["classification_hidden"],
                                                    cov_size = params_dict["cov_size"], cov_hidden = params_dict["cov_hidden"],
                                                    dropout_rate = params_dict["dropout_rate"],full_gru_ode= params_dict["full_gru_ode"], impute = params_dict["impute"])
    device_ids = [0,1,2,3]
    nnfwobj.to(device)
    # nnfwobj = torch.nn.parallel.DistributedDataParallel(nnfwobj,device_ids=[0,1,2,3])
    
    print(device)
    optimizer = torch.optim.Adam(nnfwobj.parameters(), lr=params_dict["lr"], weight_decay= params_dict["weight_decay"])
    
    # optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids)
    class_criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    print("Start Training")
    val_metric_prev = -1000
    for epoch in range(epoch_max):
        nnfwobj.train()
        total_train_loss = 0
        total_loss = 0
        auc_total_train  = 0
        tot_loglik_loss = 0
        # print("%"*7)
        # print(len(dl))
        # print("%"*7)
        print("epoch :" ,epoch)
        for i,b in enumerate(tqdm.tqdm(dl)):
            # print("&"*7)
            # print(b)

            print("&"*7)
            times    = b["times"]
            # times = times[0:1500]
            # b = prepare_sample(args,spter,b)
            optimizer.zero_grad()
            
            time_ptr = b["time_ptr"]

            # time_ptr= time_ptr[0:1500]
            X = torch.tensor(b['hist_ndFeats_list']).to(device)
            
            print(X.size())
            
            
            
            if(X.size()!=torch.Size([0])):
                cov = b['cov'].to(device)
                # M        = b["M"].to(device)
                # obs_idx  = b["obs_idx"]
                # cov      = b["cov"].to(device)
                
                labels = b['label_sp']['vals'].to(device)
                print('X',X.device,
                        'cov',cov.device,
                        'labels',labels.device)
                # labels.to(device)
                # labels = torch.tensor(labels)
                
                batch_size = labels.size(0)
                
                # print(len(times))
                # print(len(time_ptr))
                # print(X)
                # print(M)
                # print(len(obs_idx))
                # print(len(cov))
                h0 = 0# torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(device)
                hT, loss, class_pred, mse_loss  = nnfwobj(times,time_ptr, X, delta_t=params_dict["delta_t"], T=params_dict["T"], cov = cov,labels=labels)
                print('class_pred',class_pred[0].size())
                print('labels',labels.size())
                # print('*'*10)
                # print('loss' ,loss.size(),'class_pred',class_pred.size(),'labels',labels.size(),'batch_size',batch_size.size())
                # total_loss = (loss + params_dict["lambda"]*class_criterion(class_pred, labels))/batch_size
                class_pred.to('cpu')
                labels.to('cpu')
                # batch_size.to(device)
                if args.task=='edge_cls':
                    total_loss = (class_criterion(class_pred[0].view(1,-1)[0], labels[0].float())/batch_size)
                elif args.task=='link_pred':
                    # print(class_pred)
                    # print(labels)
                    total_loss = (class_criterion(class_pred[0].view(1,-1)[0], labels[0].float())/batch_size)
                
                print(total_loss)
                # total_train_loss += total_loss
                # tot_loglik_loss +=mse_loss
                # try:
                #     auc_total_train += roc_auc_score(labels.detach().cpu(),torch.sigmoid(class_pred).detach().cpu())
                # except ValueError:
                #     if params_dict["verbose"]>=3:
                #         print("Single CLASS ! AUC is erroneous")
                #     pass
                
                total_loss.backward()
                optimizer.step()


        
        # info = { 'training_loss' : total_train_loss.detach().cpu().numpy()/(i+1), 'AUC_training' : auc_total_train/(i+1), "loglik_loss" :tot_loglik_loss.detach().cpu().numpy()}
        # for tag, value in info.items():
        #     logger.scalar_summary(tag, value, epoch)
        # print(f"NegLogLik Loss train : {tot_loglik_loss.detach().cpu().numpy()}")
        
        data_utils.adjust_learning_rate(optimizer,epoch,params_dict["lr"])

        with torch.no_grad():
            nnfwobj.eval()
            total_loss_val = 0
            auc_total_val = 0
            loss_val = 0
            mse_val  = 0
            corr_val = 0
            num_obs = 0
            MAPS = 0
            num  = -1
            for i, b in enumerate(dl_val):
                num = num+1
                times    = b["times"]
                # times = times[0:1500]
                time_ptr = b["time_ptr"]
                # time_ptr = time_ptr[0:1500]
                X        = torch.tensor(b["hist_ndFeats_list"]).to(device)
                # X = torch.tensor(X)
                # X = X[0:1500]
                # X.to(device)
                # M        = b["M"].to(device)
                # obs_idx  = b["obs_idx"]
                
                # cov = cov[0:1500]
            
                

                # if b["hist_ndFeats_list"] is not None:
                    # X_val     = b["hist_ndFeats_list"]
                    # X_val = torch.tensor(X_val)
                    # X_val.to(device)
                    # M_val     = b["M_val"].to(device)
                    # times_val = b["times_val"]
                    # times_idx = b["index_val"]
                if(X.size()!=torch.Size([0])):
                    labels = b['label_sp']['vals'].to(device)
                    # labels.to(device)
                    batch_size = labels.size(0)
                    cov      = b["cov"].to(device)
                    h0 = 0 #torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(device)
                    hT, loss, class_pred, t_vec  = nnfwobj(times, time_ptr, X, delta_t=params_dict["delta_t"], T=params_dict["T"], cov=cov, return_path=False)
                    # total_loss = (loss + params_dict["lambda"]*class_criterion(class_pred, labels))/batch_size
                    # print('class_pred',class_pred)
                    # print('labels',labels)
                    class_pred.to('cpu')
                    labels.to('cpu')
                    # batch_size.to(device)
                    total_loss = (class_criterion(class_pred[0].view(1,-1)[0], labels[0].float())/batch_size)
                    if args.task=='edge_cls':
                        MAP = torch.tensor(get_MAP(class_pred[0].view(1,-1)[0],labels[0].float(), do_softmax=False))
                    elif args.task =='link_pred':
                        t = b['t'][0]
                        # print(t)
                        # print(class_pred[0].view(1,-1)[0].size())
                        # print(labels[0].size())
                        MAP = torch.tensor(get_MAP(class_pred[0].view(1,-1)[0][t],labels[0][t].float(), do_softmax=False))
                #     try:
                #         auc_val=roc_auc_score(labels.cpu(),torch.sigmoid(class_pred).cpu())
                #     except ValueError:
                #         auc_val = 0.5
                #         if params_dict["verbose"]>=3:
                #             print("Only one class : AUC is erroneous")
                #         pass

                #     if params_dict["lambda"]==0:
                #         t_vec = np.around(t_vec,str(params_dict["delta_t"])[::-1].find('.')).astype(np.float32) #Round floating points error in the time vector.
                #         p_val = data_utils.extract_from_path(t_vec,p_vec,times_val,times_idx)
                #         m, v = torch.chunk(p_val,2,dim=1)
                    
                #         last_loss = (data_utils.log_lik_gaussian(X_val,m,v)*M_val).sum()
                #         mse_loss = (torch.pow(X_val-m,2)*M_val).sum()
                #         corr_val_loss = data_utils.compute_corr(X_val, m, M_val)
                    
                #         num_obs = datautils.compute(num_obs,last_loss)
                #         loss_val += last_loss.cpu().numpy()
                    
                #         num_obs += M_val.sum().cpu().numpy()
                #         mse_val += mse_loss.cpu().numpy()
                #         corr_val += corr_val_loss.cpu().numpy()
                #     else:
                #         num_obs=1
                    

                #     total_loss_val += total_loss.cpu().detach().numpy()
                #     auc_total_val += auc_val
            
                # loss_val /= num_obs
                # mse_val /=  num_obs
            
            
                # info = { 'validation_loss' : total_loss_val/(i+1), 'AUC_validation' : auc_total_val/(i+1),
                #             'loglik_loss' : loss_val, 'validation_mse' : mse_val, 'correlation_mean' : np.nanmean(corr_val),
                #         'correlation_max': np.nanmax(corr_val), 'correlation_min': np.nanmin(corr_val)}
                # for tag, value in info.items():
                #     logger.scalar_summary(tag, value, epoch)
                    info = {'Validation Map':MAP}
                    MAPS = MAPS+MAP
                # if params_dict["lambda"]==0:
                #     val_metric = - loss_val
                # else:
                #     val_metric = auc_total_val/(i+1)

                # if val_metric > val_metric_prev:
                #     print(f"New highest validation metric reached ! : {val_metric}")
                #     print("Saving Model")
                #     torch.save(nnfwobj.state_dict(),f"./../trained_models/{simulation_name}_MAX.pt")
                #     val_metric_prev = val_metric
            print(f"Validation MAPS {MAPS/(num+1)}: {MAPS/(num+1)}")
            test_map = test_evaluation(nnfwobj, params_dict, class_criterion, device, dl_test)
            print(f"Test MAP{test_map} : {test_map}")
            #     print(f"Test loglik loss at epoch {epoch} : {test_loglik}")
            #     print(f"Test AUC loss at epoch {epoch} : {test_auc}")
            #     print(f"Test MSE loss at epoch{epoch} : {test_mse}")
            # else:
            #     if epoch % 10:
            #         torch.save(nnfwobj.state_dict(),f"./../trained_models/{simulation_name}.pt")

        # print(f"Total validation loss at epoch {epoch}: {total_loss_val/(i+1)}")
        # print(f"Validation AUC at epoch {epoch}: {auc_total_val/(i+1)}")
        # print(f"Validation loss (loglik) at epoch {epoch}: {loss_val:.5f}. MSE : {mse_val:.5f}. Correlation : {np.nanmean(corr_val):.5f}. Num obs = {num_obs}")
    
    
    print(f"Finished training GRU-ODE for Climate. Saved in ./../trained_models/{simulation_name}")

    # return(info, val_metric_prev, test_loglik, test_auc, test_mse)

def test_evaluation(model, params_dict, class_criterion, device, dl_test):
    with torch.no_grad():
        model.eval()
        total_loss_test = 0
        auc_total_test = 0
        loss_test = 0
        mse_test  = 0
        corr_test = 0
        num_obs = 0
        MAPS = 0
        for i, b in enumerate(dl_test):
            times    = b["times"]
            # times = times[0:1500]
            time_ptr = b["time_ptr"]
            # time_ptr = time_ptr[0:1500]
            X        = torch.tensor(b["hist_ndFeats_list"]).to(device)
            # X = torch.tensor(X)
            # X = X[0:1500]
            # X.to(device)
            # M        = b["M"].to(device)
            # obs_idx  = b["obs_idx"]
            

            # if b["hist_ndFeats_list"] is not None:
                # X_val     = b["hist_ndFeats_list"]
                # X_val = torch.tensor(X_val)
                # X_val.to(device)
                # M_val     = b["M_val"].to(device)
                # times_val = b["times_val"]
                # times_idx = b["index_val"]
            if(X.size()!=torch.Size([0])):
                cov      = b["cov"].to(device)
                # cov = cov[0:1500]
            
                labels = b['label_sp']['vals'].to(device)
                # labels.to(device)
                batch_size = labels.size(0)
                h0 = 0 #torch.zeros(labels.shape[0], params_dict["hidden_size"]).to(device)
                hT, loss, class_pred, t_vec= model(times, time_ptr, X, delta_t=params_dict["delta_t"], T=params_dict["T"], cov=cov, return_path=False)
                # total_loss = (loss + params_dict["lambda"]*class_criterion(class_pred, labels))/batch_size
                # print('class_pred',class_pred)
                # print('labels',labels)
                class_pred.to('cpu')
                labels.to('cpu')
                # batch_size.to(device)
                total_loss = (class_criterion(class_pred[0].view(1,-1)[0], labels[0].float())/batch_size)
                if args.task=='edge_cls':
                    MAP = torch.tensor(get_MAP(class_pred[0].view(1,-1)[0],labels[0].float(), do_softmax=False))
                elif args.task =='link_pred':
                    t = b['t'][0]
                    MAP = torch.tensor(get_MAP(class_pred[0].view(1,-1)[0][t],labels[0][t].float(), do_softmax=False))
                MAP = torch.tensor(get_MAP(class_pred[0].view(1,-1)[0],labels[0].float(), do_softmax=False))
                MAPS = MAPS+MAP
            #     try:
            #         auc_test=roc_auc_score(labels.cpu(),torch.sigmoid(class_pred).cpu())
            #     except ValueError:
            #         if params_dict["verbose"]>=3:
            #             print("Only one class. AUC is wrong")
            #         auc_test = 0
            #         pass

            #     if params_dict["lambda"]==0:
            #         t_vec = np.around(t_vec,str(params_dict["delta_t"])[::-1].find('.')).astype(np.float32) #Round floating points error in the time vector.
            #         p_val = data_utils.extract_from_path(t_vec,p_vec,times_val,times_idx)
            #         m, v = torch.chunk(p_val,2,dim=1)
                
            #         last_loss = (data_utils.log_lik_gaussian(X_val,m,v)*M_val).sum()
            #         mse_loss = (torch.pow(X_val-m,2)*M_val).sum()
            #         corr_test_loss = data_utils.compute_corr(X_val, m, M_val)

            #         loss_test += last_loss.cpu().numpy()
            #         # num_obs += M_val.sum().cpu().numpy()
            #         mse_test += mse_loss.cpu().numpy()
            #         corr_test += corr_test_loss.cpu().numpy()
            #     else:
            #         num_obs=1

            #     total_loss_test += total_loss.cpu().detach().numpy()
            #     auc_total_test += auc_test

        # loss_test /= num_obs
        # mse_test /=  num_obs
        # auc_total_test /= (i+1)

        return MAPS/(i+1)


def random_param_value(param, param_min, param_max, type='int'):
    if str(param) is None or str(param).lower()=='none':
        if type=='int':
            return random.randrange(param_min, param_max+1)
        elif type=='logscale':
            interval=np.logspace(np.log10(param_min), np.log10(param_max), num=100)
            return np.random.choice(interval,1)[0]
        else:
            return random.uniform(param_min, param_max)
    else:
        return param

def build_random_hyper_params(args):
    if args.model == 'all':
        model_types = ['gcn', 'egcn_o', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
        args.model=model_types[args.rank]
    elif args.model == 'all_nogcn':
        model_types = ['egcn_o', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
        args.model=model_types[args.rank]
    elif args.model == 'all_noegcn3':
        model_types = ['gcn', 'egcn_h', 'gruA', 'gruB','egcn','lstmA', 'lstmB']
        args.model=model_types[args.rank]
    elif args.model == 'all_nogruA':
        model_types = ['gcn', 'egcn_o', 'egcn_h', 'gruB','egcn','lstmA', 'lstmB']
        args.model=model_types[args.rank]
        args.model=model_types[args.rank]
    elif args.model == 'saveembs':
        model_types = ['gcn', 'gcn', 'skipgcn', 'skipgcn']
        args.model=model_types[args.rank]

    args.learning_rate =random_param_value(args.learning_rate, args.learning_rate_min, args.learning_rate_max, type='logscale')
    # args.adj_mat_time_window = random_param_value(args.adj_mat_time_window, args.adj_mat_time_window_min, args.adj_mat_time_window_max, type='int')

    if args.model == 'gcn':
        args.num_hist_steps = 0
    else:
        args.num_hist_steps = random_param_value(args.num_hist_steps, args.num_hist_steps_min, args.num_hist_steps_max, type='int')

    args.gcn_parameters['feats_per_node'] =random_param_value(args.gcn_parameters['feats_per_node'], args.gcn_parameters['feats_per_node_min'], args.gcn_parameters['feats_per_node_max'], type='int')
    args.gcn_parameters['layer_1_feats'] =random_param_value(args.gcn_parameters['layer_1_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
    if args.gcn_parameters['layer_2_feats_same_as_l1'] or args.gcn_parameters['layer_2_feats_same_as_l1'].lower()=='true':
        args.gcn_parameters['layer_2_feats'] = args.gcn_parameters['layer_1_feats']
        args.gcn_parameters['layer_3_feats'] = args.gcn_parameters['layer_1_feats']
        args.gcn_parameters['layer_4_feats'] = args.gcn_parameters['layer_1_feats']
    else:
        args.gcn_parameters['layer_2_feats'] =random_param_value(args.gcn_parameters['layer_2_feats'], args.gcn_parameters['layer_1_feats_min'], args.gcn_parameters['layer_1_feats_max'], type='int')
        args.gcn_parameters['layer_3_feats'] = random_param_value(args.gcn_parameters['layer_3_feats'],
                                                                    args.gcn_parameters['layer_1_feats_min'],
                                                                    args.gcn_parameters['layer_1_feats_max'], type='int')
        args.gcn_parameters['layer_4_feats'] = random_param_value(args.gcn_parameters['layer_4_feats'],
                                                                    args.gcn_parameters['layer_1_feats_min'],
                                                                    args.gcn_parameters['layer_1_feats_max'], type='int')
    args.gcn_parameters['lstm_l1_feats'] =random_param_value(args.gcn_parameters['lstm_l1_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
    if args.gcn_parameters['lstm_l2_feats_same_as_l1'] or args.gcn_parameters['lstm_l2_feats_same_as_l1'].lower()=='true':
        args.gcn_parameters['lstm_l2_feats'] = args.gcn_parameters['lstm_l1_feats']
        args.gcn_parameters['lstm_l3_feats'] = args.gcn_parameters['lstm_l1_feats']
        args.gcn_parameters['lstm_l4_feats'] = args.gcn_parameters['lstm_l1_feats']
    else:
        args.gcn_parameters['lstm_l2_feats'] =random_param_value(args.gcn_parameters['lstm_l2_feats'], args.gcn_parameters['lstm_l1_feats_min'], args.gcn_parameters['lstm_l1_feats_max'], type='int')
        args.gcn_parameters['lstm_l3_feats'] = random_param_value(args.gcn_parameters['lstm_l3_feats'],
                                                                    args.gcn_parameters['lstm_l1_feats_min'],
                                                                    args.gcn_parameters['lstm_l1_feats_max'], type='int')
        args.gcn_parameters['lstm_l4_feats'] = random_param_value(args.gcn_parameters['lstm_l4_feats'],
                                                                    args.gcn_parameters['lstm_l1_feats_min'],
                                                                    args.gcn_parameters['lstm_l1_feats_max'], type='int')
    args.gcn_parameters['cls_feats']=random_param_value(args.gcn_parameters['cls_feats'], args.gcn_parameters['cls_feats_min'], args.gcn_parameters['cls_feats_max'], type='int')
    return args

def build_dataset(args):
    if args.data == 'bitcoinotc' or args.data == 'bitcoinalpha':
        if args.data == 'bitcoinotc':
            args.bitcoin_args = args.bitcoinotc_args
        elif args.data == 'bitcoinalpha':
            args.bitcoin_args = args.bitcoinalpha_args
        return bc.bitcoin_dataset(args)
    elif args.data == 'aml_sim':
        return aml.Aml_Dataset(args)
    elif args.data == 'elliptic':
        return ell.Elliptic_Dataset(args)
    elif args.data == 'elliptic_temporal':
        return ell_temp.Elliptic_Temporal_Dataset(args)
    elif args.data == 'uc_irv_mess':
        return ucim.Uc_Irvine_Message_Dataset(args)
    elif args.data == 'dbg':
        return dbg.dbg_dataset(args)
    elif args.data == 'colored_graph':
        return cg.Colored_Graph(args)
    elif args.data == 'autonomous_syst':
        return aus.Autonomous_Systems_Dataset(args)
    elif args.data == 'reddit':
        return rdt.Reddit_Dataset(args)
    elif args.data.startswith('sbm'):
        if args.data == 'sbm20':
            args.sbm_args = args.sbm20_args
        elif args.data == 'sbm50':
            args.sbm_args = args.sbm50_args
        return sbm.sbm_dataset(args)
    else:
        raise NotImplementedError('only arxiv has been implemented')

def build_tasker(args,dataset):
    if args.task == 'link_pred':
        return lpt.Link_Pred_Tasker(args,dataset)
    elif args.task == 'edge_cls':
        return ect.Edge_Cls_Tasker(args,dataset)
    elif args.task == 'node_cls':
        return nct.Node_Cls_Tasker(args,dataset)
    elif args.task == 'static_node_cls':
        return nct.Static_Node_Cls_Tasker(args,dataset)

    else:
        raise NotImplementedError('still need to implement the other tasks')

def build_gcn(args,tasker):
    gcn_args = u.Namespace(args.gcn_parameters)
    gcn_args.feats_per_node = tasker.feats_per_node
    if args.model == 'gcn':
        return mls.Sp_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
    elif args.model == 'skipgcn':
        return mls.Sp_Skip_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
    elif args.model == 'skipfeatsgcn':
        return mls.Sp_Skip_NodeFeats_GCN(gcn_args,activation = torch.nn.RReLU()).to(args.device)
    else:
        assert args.num_hist_steps > 0, 'more than one step is necessary to train LSTM'
        if args.model == 'lstmA':
            return mls.Sp_GCN_LSTM_A(gcn_args,activation = torch.nn.RReLU()).to(args.device)
        elif args.model == 'gruA':
            return mls.Sp_GCN_GRU_A(gcn_args,activation = torch.nn.RReLU()).to(args.device)
        elif args.model == 'lstmB':
            return mls.Sp_GCN_LSTM_B(gcn_args,activation = torch.nn.RReLU()).to(args.device)
        elif args.model == 'gruB':
            return mls.Sp_GCN_GRU_B(gcn_args,activation = torch.nn.RReLU()).to(args.device)
        elif args.model == 'egcn':
            return egcn.EGCN(gcn_args, activation = torch.nn.RReLU()).to(args.device)
        elif args.model == 'egcn_h':
            return egcn_h.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
        elif args.model == 'skipfeatsegcn_h':
            return egcn_h.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device, skipfeats=True)
        elif args.model == 'egcn_o':
            return egcn_o.EGCN(gcn_args, activation = torch.nn.RReLU(), device = args.device)
        else:
            raise NotImplementedError('need to finish modifying the models')

def build_classifier(args,tasker):
    if 'node_cls' == args.task or 'static_node_cls' == args.task:
        mult = 1
    else:
        mult = 2
    if 'gru' in args.model or 'lstm' in args.model:
        in_feats = args.gcn_parameters['lstm_l2_feats'] * mult
    elif args.model == 'skipfeatsgcn' or args.model == 'skipfeatsegcn_h':
        in_feats = (args.gcn_parameters['layer_2_feats'] + args.gcn_parameters['feats_per_node']) * mult
    else:
        in_feats = args.gcn_parameters['layer_2_feats'] * mult

    return mls.Classifier(args,in_features = in_feats, out_features = tasker.num_classes).to(args.device)


if __name__ =="__main__":

    simulation_name="small_climate"
    # torch.cuda.set_device(0)
    device = torch.device("cuda")


    # train_idx = np.load("../../gru_ode_bayes/datasets/Climate/folds/small_chunk_fold_idx_0/train_idx.npy",allow_pickle=True)
    # val_idx = np.load("../../gru_ode_bayes/datasets/Climate/folds/small_chunk_fold_idx_0/val_idx.npy",allow_pickle=True)
    # test_idx = np.load("../../gru_ode_bayes/datasets/Climate/folds/small_chunk_fold_idx_0/test_idx.npy",allow_pickle=True)
    train_idx = 0
    val_idx = 0
    test_idx = 0
    #Model parameters.
    params_dict=dict()

    params_dict["csv_file_path"] = "../../gru_ode_bayes/datasets/Climate/small_chunked_sporadic.csv" 
    params_dict["csv_file_tags"] = None
    params_dict["csv_file_cov"]  = None

    params_dict["hidden_size"] = 75
    params_dict["p_hidden"] = 20
    params_dict["prep_hidden"] = 5#log2 10
    params_dict["logvar"] = True
    params_dict["mixing"] = 1e-4 #Weighting between KL loss and MSE loss.
    params_dict["delta_t"]=0.1
    params_dict["T"]=200
    params_dict["lambda"] = 0 #Weighting between classification and MSE loss.

    params_dict["classification_hidden"] = 2
    params_dict["cov_hidden"] = 50
    params_dict["weight_decay"] = 0.0001
    params_dict["dropout_rate"] = 0.2
    params_dict["lr"]=0.001
    params_dict["full_gru_ode"] = True
    params_dict["no_cov"] = True
    params_dict["impute"] = False
    params_dict["verbose"] = 0 #from 0 to 3 (highest)

    params_dict["T_val"] = 150
    params_dict["max_val_samples"] = 3
    

    ##############参数解析，args中存所有yaml信息################
    parser = u.create_parser()
    args = u.parse_args(parser)
    ##############添加/更改配置文件信息
    global rank, wsize, use_cuda
    args.use_cuda = (torch.cuda.is_available() and args.use_cuda)
    args.device='cuda'
    if args.use_cuda:
        args.device='cuda'
    print ("use CUDA:", args.use_cuda, "- device:", args.device)
    try:
        dist.init_process_group(backend='mpi') #, world_size=4
        rank = dist.get_rank()#0-world_size的进程组等级
        wsize = dist.get_world_size()#当前进程组的进程数
        print("this")
        print('Hello from process {} (out of {})'.format(dist.get_rank(), dist.get_world_size()))
        if args.use_cuda:
            torch.cuda.set_device(rank )  # are we sure of the rank+1????
            print('using the device {}'.format(torch.cuda.current_device()))
    except:
        rank = 0
        wsize = 1
        print(('MPI backend not preset. Set process rank to {} (out of {})'.format(rank,
                                                                                    wsize)))

    if args.seed is None and args.seed!='None':
        seed = 123+rank#int(time.time())+rank
    else:
        seed=args.seed#+rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args.seed=seed
    args.rank=rank
    args.wsize=wsize
    print(device)
    # Assign the requested random hyper parameters
    args = build_random_hyper_params(args)

    #build the dataset
    #bitcoin_dataset类
    dataset = build_dataset(args)
    #build the tasker
    tasker = build_tasker(args,dataset)
    splitter = sp.splitter(args,tasker)
    # splitter = sp.splitter(args,tasker)
    # gcn = build_gcn(args, tasker)
    print("fuck")
    torch.backends.cudnn.enabled = True

    torch.backends.cudnn.benchmark = True
    info, val_metric_prev, test_loglik, test_auc, test_mse = train_gruode(simulation_name = simulation_name,
                        params_dict = params_dict,
                        device = device,
                        train_idx = train_idx,
                        val_idx = val_idx,
                        test_idx = test_idx,
                        epoch_max=100,
                        spter = splitter,
                        args = args
                       )

