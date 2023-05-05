import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Linear
from captum.attr import IntegratedGradients

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.nn import GCNConv, SAGEConv,to_captum_model,TransformerConv
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import to_dense_adj

import copy
import argparse
from collections import Counter
import random
from sklearn.metrics import roc_auc_score


def perturb_node_feature(data, noise_ratio = 0.1, noise_mean=0, noise_std =0.01, noise_prob=0.01, noise = "None"):
    
    '''
    data_name : the name of the dataset
    noise_ratio : how much noise node features added to the data
    noise_mean : the mean of the gaussian noise
    noise_std : the standard deviation of the gaussian noise
    noise_prob : the probability of the Bernoulli noise
    noise : add Gaussian noise, Bernoulli noise or none
    
    output: perturbed data and the number of features
    '''
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if noise == 'G':
        #Gaussian noise
        noise = torch.normal(mean=noise_mean, std = noise_std, size=(data.x.shape[0], int(data.x.shape[1] * noise_ratio)))
        noise = noise.to(device)
        data.x = torch.cat((data.x, noise),1)
        original_feature = copy.deepcopy(data.x)
        num_features = data.x.shape[1]
    elif noise == 'B':
        #Bernoulli noise
        noise = torch.bernoulli(torch.ones(data.x.shape[0], int(data.x.shape[1]*noise_ratio))*noise_prob)
        noise = noise.to(device)
        data.x = torch.cat((data.x, noise),1)
        original_feature = copy.deepcopy(data.x)
        num_features = data.x.shape[1]
    else:
        original_feature = copy.deepcopy(data.x)
        num_features = data.x.shape[1]
        
    return data, num_features, original_feature
        
    
        
        
def perturb_edge(data, edge_noise_ratio = 0.1, noise_prob=0.5, edge_noise = "None"):
    
    '''
    data_name : the name of the dataset
    noise_ratio : how much noise edges added to the data
    noise_prob : the probability of the Bernoulli noise
    noise : add Bernoulli noise or none
    '''
        
    #Bernoulli noise
    if edge_noise == "B":
        adj = to_dense_adj(data.edge_index)
        NonEdge = torch.vstack((torch.where(adj == 0)[1],torch.where(adj == 0)[2]))
        num_noise = int(edge_noise_ratio * data.edge_index.shape[1])
        idx = random.sample(range(data.edge_index.shape[1]), num_noise)
        new_edges = torch.hstack((data.edge_index, NonEdge[:,idx]))
        
        original_edges = copy.deepcopy(new_edges)
        data.edge_index = new_edges
    else:
        original_edges = copy.deepcopy(data.edge_index)
        
        
    return data, original_edges
        
    
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Linear
from captum.attr import IntegratedGradients

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.nn import GCNConv, SAGEConv, TransformerConv, GINConv, GATConv
from torch_geometric.utils import k_hop_subgraph


class GNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads = 1, dropRate = 0.6, GNN_name="GCN"):
        super().__init__()
        
        if GNN_name == "GCN":
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
        elif GNN_name == "SAGE":
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, out_dim)
        elif GNN_name == "GIN":
            self.conv1 = GINConv(in_dim, hidden_dim)
            self.conv2 = GINConv(hidden_dim, out_dim)
        elif GNN_name == "GAT":
            self.conv1 = GATConv(in_dim, 8, heads=8, dropRate=0.6)
            self.conv2 = GATConv(8 * 8, out_dim, heads=num_heads, concat=False,
                             dropRate=0.6)

    def forward(self, x, edge_index, edge_weight = None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    

def train(data, model, optimizer, edge_weight=None):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data.x, data.edge_index, edge_weight)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()
    

@torch.no_grad()
def test(data, model, edge_weight=None):
    model.eval()
    logits, accs = model(data.x, data.edge_index, edge_weight), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def train_citation_data(data_name, train_mode, lr, num_epochs, pre_epochs, 
                        noise_ratio = 0.1, noise_mean=0, noise_std =0.01, noise_prob=0.5,
                        inject_noise =0.05, noise = "None", patienceT = 30, hidden_dim = 128, GNN_name="GCN",
                        node_cutoff = 0, edge_cutoff = 0, IG_type = 0, edge_drop = 1, edge_noise_ratio = 0.01,
                        edge_noise = "None"):
    """
    data_name : the name of the dataset
    train_mode : random_drop, IG_drop, no_drop
    lr : learning rate
    
    noise_ratio : how much noise node features added to the data
    noise_mean : the mean of the gaussian noise
    noise_std : the standard deviation of the gaussian noise
    noise_prob : the probability of the Bernoulli noise
    noise : add Gaussian noise, Bernoulli noise or none
    inject_noiseR : the scale of the injected noise
    patienceT : early stop critieria
    IG_type : three settings, nodes(0), edges(1), nodes and edges(3)
    edge_drop : interpretable attention (0), drop_edge(1)
    
    output: perturbed data and the number of features
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if data_name in ['cora', "CiteSeer", 'pubmed']:
        #dataset = Planetoid('.', data_name, transform=T.NormalizeFeatures())
        dataset = Planetoid('.', data_name, transform=T.NormalizeFeatures())
        data = dataset[0].to(device)
        
        
        #Normalize node features (standardize)
#         node_features = data.x
#         node_features_mean = node_features.mean(0)
#         node_features_std = node_features.std(0)
#         node_features = (node_features - node_features_mean) / node_features_std
#         data.x = node_features
    
    data, num_features, original_feature = perturb_node_feature(data, noise_ratio = noise_ratio, 
                                                                  noise_mean=noise_mean, noise_std =noise_std, 
                                                                  noise_prob=noise_prob, noise = noise)
    
    data, original_edges = perturb_edge(data, edge_noise_ratio = edge_noise_ratio, 
                                        noise_prob=noise_prob, edge_noise = edge_noise)
    
    
    train_nodes_idx = np.where(data.cpu().train_mask)[0]
    val_nodes_idx = np.where(data.cpu().val_mask)[0]
    test_nodes_idx = np.where(data.cpu().test_mask)[0]
    data = data.to(device)
    
    model = GNN(in_dim = num_features, hidden_dim = hidden_dim, out_dim = dataset.num_classes, 
                num_heads = 1, dropRate = 0.6, GNN_name=GNN_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    best_val_acc = test_acc = 0
    
    
    for epoch in range(1, num_epochs):
        
        if train_mode == 'random_drop':
            train(data, model, optimizer)
            feature_mask = torch.bernoulli(torch.ones(data.x.shape)*0.8).to(device)
            new_feature = feature_mask * original_feature
            data.x = new_feature
        if train_mode == "random_drop_E":
            train(data, model, optimizer)
            edge_mask = torch.bernoulli(torch.ones(data.edge_index.shape[1])*0.9).to(device)
            new_edge = edge_mask.int() * original_edges
            data.edge_index = new_edge
            
        elif train_mode == 'original':
            train(data, model, optimizer)
        elif train_mode == 'IG_drop':
            # initialize the edges to the original edges
            
            #data.x = original_feature
            #print(data.edge_index.shape)
            
            # pretrain for some epochs
            if epoch <= pre_epochs:
                train(data, model, optimizer)
                if epoch == pre_epochs:
                    edge_weight = torch.ones(data.edge_index.shape[0]).to(device)
                    
                    y_pred = model(data.x, data.edge_index).max(1)[1]
                    y_pred[train_nodes_idx] = data.y[train_nodes_idx]
                    
                    best_val_acc = test_acc = 0
                    with torch.no_grad():
                        for param in model.parameters():
                            param.add_(torch.randn(param.size()).to(device) * inject_noise)
            else:
                # IG drop on nodes (IG_type = 0)
                if IG_type == 0:
        
                    train(data, model, optimizer)
                    captum_model = to_captum_model(model, mask_type='node')

                    ig = IntegratedGradients(captum_model)
                
                    if epoch > 1000:
                        
                        ig_attr_node = ig.attribute(data.x.unsqueeze(0), target=y_pred,
                            additional_forward_args=(data.edge_index),
                            internal_batch_size=1)
                        y_pred = model(data.x, data.edge_index).max(1)[1]
                        y_pred[train_nodes_idx] = data.y[train_nodes_idx]
                        # normalize the gradient
                        #ig_attr_node = ig_attr_node.squeeze(0).abs().sum(dim=1)
                        #ig_attr_node /= ig_attr_node.max()

                    else:
                        
                        ig_attr_node = ig.attribute(data.x.unsqueeze(0), target=y_pred,
                            additional_forward_args=(data.edge_index),
                            internal_batch_size=1)
                        ig_attr_node = ig_attr_node.abs()
                        y_pred = model(data.x, data.edge_index).max(1)[1]
                        y_pred[train_nodes_idx] = data.y[train_nodes_idx]
                        
      
                    new_feature = (ig_attr_node.squeeze() > node_cutoff) * original_feature
                    data.x = new_feature.squeeze(0)
            
                # IG drop on edges
                elif IG_type == 1:
                    train(data, model, optimizer, edge_weight)

                    data.edge_index = original_edges

                    y_pred = model(data.x, data.edge_index).max(1)[1]
                    y_pred[train_nodes_idx] = data.y[train_nodes_idx]
                    captum_model = to_captum_model(model, mask_type='edge')
                    edge_mask = torch.ones(data.num_edges, requires_grad=True, device=device)
                    #y_pred = model(data.x, data.edge_index).max(1)[1]
                    #y_pred[train_nodes_idx] = data.y[train_nodes_idx]
                    
                    ig = IntegratedGradients(captum_model)
                    ig_attr_edge = ig.attribute(edge_mask.unsqueeze(0), target=y_pred,
                            additional_forward_args=(data.x, data.edge_index),
                            internal_batch_size=1)
                    ig_attr_edge = ig_attr_edge.squeeze(0).abs()
                    ig_attr_edge /= ig_attr_edge.max()
                    
                    if edge_drop == 0:
                        edge_weight = ig_attr_edge
                        
                        #print(data.edge_index.shape)
                        
                    elif edge_drop == 1:
                        new_edges = original_edges[:,(ig_attr_edge.squeeze() > edge_cutoff)]
                        data.edge_index = new_edges
                        
                        #print(data.edge_index.shape)

                    
                elif IG_type == 2:
                    train(data, model, optimizer)
                    data.edge_index = original_edges
                    
                    captum_model = to_captum_model(model, mask_type='node_and_edge')
                    edge_mask = torch.ones(data.num_edges, requires_grad=True, device=device)
                    ig = IntegratedGradients(captum_model)
                    ig_attr_node, ig_attr_edge = ig.attribute((data.x.unsqueeze(0), edge_mask.unsqueeze(0)), 
                                                              target=y_pred,additional_forward_args=(data.edge_index), 
                                                              internal_batch_size=1)

                    # Scale attributions to [0, 1]:
#                     ig_attr_node = ig_attr_node.squeeze(0).abs().sum(dim=1)
#                     ig_attr_node /= ig_attr_node.max()
                    ig_attr_edge = ig_attr_edge.squeeze(0).abs()
                    ig_attr_edge /= ig_attr_edge.max()
                    
                    y_pred = model(data.x, data.edge_index).max(1)[1]
                    y_pred[train_nodes_idx] = data.y[train_nodes_idx]
                    
                    new_feature = (ig_attr_node.squeeze() > node_cutoff) * original_feature
                    data.x = new_feature.squeeze(0)
                    new_edges = original_edges[:,(ig_attr_edge.squeeze() > edge_cutoff)]
                    data.edge_index = new_edges
                    
   
        train_acc, val_acc, tmp_test_acc = test(data, model)
        if val_acc > best_val_acc:
            patience = 0
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        else:
            patience += 1
            if patience > patienceT and epoch > pre_epochs:
                break
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
          f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
    return(test_acc)


parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('-f')
parser.add_argument('--data_name', type=str, default="MR")
parser.add_argument('--train_mode', type=str, default="IG_drop") # random_drop_E, random_drop, original, IG_drop
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--pre_epochs', type=int, default=50)
parser.add_argument('--noise_ratio', type=float, default=0.2)
parser.add_argument('--noise_mean', type=float, default=0)
parser.add_argument('--noise_std', type=float, default=0.01)
parser.add_argument('--noise_prob', type=float, default=0.01)
parser.add_argument('--inject_noise', type=float, default=0)
parser.add_argument('--noise', type=str, default="None")
parser.add_argument('--patienceT', type=int, default=30)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--GNN_name', type=str, default="GCN")
parser.add_argument('--node_cutoff', type=float, default=0.001)
parser.add_argument('--edge_cutoff', type=float, default=0.0001)
parser.add_argument('--IG_type', type=int, default=2) # IGF(0), IGE(1), IGEF(2)
parser.add_argument('--edge_drop', type=int, default=0) # drop edge(1), interpretable attention(0)
parser.add_argument('--edge_noise_ratio', type=float, default=0.1)
parser.add_argument('--edge_noise', type=str, default="None")

args = parser.parse_args()

print(args)

accs = []
for i in range(5):
    test_acc = train_citation_data(data_name = args.data_name, train_mode = args.train_mode, lr = args.lr, 
                    num_epochs = args.num_epochs, pre_epochs = args.pre_epochs, 
                        noise_ratio = args.noise_ratio, noise_mean = args.noise_mean, 
                    noise_std = args.noise_std, noise_prob = args.noise_prob,
                        inject_noise = args.inject_noise, noise = args.noise, patienceT = args.patienceT,
                   hidden_dim = args.hidden_dim, GNN_name = args.GNN_name, node_cutoff = args.node_cutoff, 
                    edge_cutoff = args.edge_cutoff, IG_type = args.IG_type, edge_drop = args.edge_drop, 
                   edge_noise_ratio = args.edge_noise_ratio, edge_noise = args.edge_noise)
    accs.append(test_acc)
print(accs)
print("mean:", np.mean(accs))
print("var:", np.std(accs)/(5-1))
