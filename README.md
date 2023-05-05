
dataset can be found at https://github.com/yao8839836/text_gcn/tree/master/data

The code can be run with command line : python IG.py, the argument are listed below:

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
