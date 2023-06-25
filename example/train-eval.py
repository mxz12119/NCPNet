import random
from time import time
import numpy as np
import torch
import os
import torch_geometric
import sys
import os.path as osp

import traceback
import copy
import argparse
import yaml
from NCPNet.brain_data import HemiBrain,LinkPred_Loader,Celegans19,LinkPred_PairNeigh_Loader

import torch_geometric.transforms as T
from NCPNet.approaches import Net
from NCPNet.utils import load_config,edge_index2Graph
from NCPNet.task import Base_Task
from torch_geometric.loader import RandomNodeSampler,NeighborLoader
from torch_geometric.transforms import RandomLinkSplit
from NCPNet.trainer import LinkPred_trainer
import matplotlib
matplotlib.use('agg')
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic =True
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default='configs/fly_linkpred.yaml')
    parser.add_argument('--search',type=str,default='plain',help='config search method')
    args = parser.parse_args()
    return args
def main():
    args=parse_args()
    task_conf=Base_Task(args.c,search_method=args.search)
    print('Task num:%d'%len(task_conf))
    for k,task in enumerate(task_conf):
        print('*'*200)
        print('Begin %d Task'%k)
        if not torch.cuda.is_available() and 'cuda' not in task['device']:
            task['device']='cpu'
        print(task)
        setup_seed(task['seed'])
        path = osp.join('data', task['Experiment'])
        task_mode=task['Model']
        transform = T.Compose([
            T.ToDevice(task['device']),
            RandomLinkSplit(num_val=task['val'], num_test=task['test'],
                            add_negative_train_samples=False),
        ])
        
        if task['Experiment']=='C.Elegans19':
            dataset = Celegans19(path, transform=transform,name=task['var'])
        elif task['Experiment']=='HemiBrain':
            dataset=HemiBrain(path,transform=transform)
        task['num_node']=dataset.data.num_nodes
        task['type_dim']=dataset.typedim
        train_data, val_data, test_data = dataset[0]
        nxg=edge_index2Graph(train_data.edge_index.cpu())
        nxg.to_undirected()
        train_loader=LinkPred_PairNeigh_Loader(task,train_data,nxg)
        test_loader=LinkPred_PairNeigh_Loader(task,test_data,nxg)
        model=Net(task)
        model=model.to(device=task['device'])
        train=LinkPred_trainer(task,model)
        train.excecute_epoch(train_loader,test_loader)    
        if 'save_dataset' in task and task['save_dataset']:
            torch.save(train_data,osp.join(train.logdir,'train_data.dt'))
            torch.save(test_data,osp.join(train.logdir,'test_data.dt'))   
        del model,train_loader,test_loader
        torch.cuda.empty_cache()
        del dataset,train_data
        print('%d Task is ended'%k)
        print('*' * 200)
    try:
        task_conf.linkpred_task_report(task_conf.logdir)
    except Exception as e:
        traceback.print_exc()
if __name__=='__main__':
    main()

