import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

def compute_distance(ts, p=2):
    if ts.size(0) == 2:
        return F.pairwise_distance(ts[0], ts[1], p)
    if ts.size(0) == 4:
        g12 = F.pairwise_distance(ts[0], ts[1], p)
        g13 = F.pairwise_distance(ts[2], ts[3], p)
        return g12, g13
    raise ValueError('Failed to compute distance!')

def compute_loss(ts, margin=1.0):
    """Compute triplet loss."""
    loss = F.triplet_margin_loss(ts[0], ts[1], ts[3], margin)

    return loss

def vector_to_binary(vector):
    """full-precision vector to binary, 1, -1"""
    pos = torch.gt(vector, 0).int()
    neg = torch.le(vector, 0).int()

    return pos - neg

def hsim(x, y, beta=5):
    return torch.mean(torch.tanh(beta * x) * torch.tanh(beta * y), dim=-1)

def binary_distance(x, y):
    return torch.sum(torch.abs(x - y), dim=-1)

def compute_binary_loss1(vs, margin=2.0):
    """Compute triplet loss."""
    x, y, z, k = vs[0], vs[1], vs[2], vs[3]
    d_pos = binary_distance(torch.tanh(5 * x), torch.tanh(5 * y)) / 2
    d_neg = binary_distance(torch.tanh(5 * k), torch.tanh(5 * z)) / 2

    l1 = (hsim(x, y) ** 2 + hsim(z, k) ** 2) / 4
    l2 = torch.abs(margin + d_pos - d_neg) * 1
    l3 = (2 * (hsim(x, x) - 1)**2 + (hsim(y, y) - 1)**2 + (hsim(k, k) - 1)**2) / 16 * 1
    l4 = torch.mean(2 * x ** 2 + y ** 2 + k **2, dim=1) / 16


    loss = l1 + l2 + l3 + l4
    return loss.mean()

def compute_binary_loss(vs, margin=2.0):
    """Compute triplet loss."""
    x, y, z, k = vs[0], vs[1], vs[2], vs[3]
    # d_pos = hsim(x, y, 1)
    # d_neg = hsim(k, z, 1) 
    
    d_pos = binary_distance(torch.tanh(5 * x), torch.tanh(5 * y)) / 2
    d_neg = binary_distance(torch.tanh(5 * k), torch.tanh(5 * z)) / 2


    # l1 = torch.abs(margin + d_pos - d_neg) * 1
    l1 = torch.log(1 + torch.exp(d_pos - d_neg + 2))

    l2 = (hsim(x, y, 1) ** 2 + hsim(z, k, 1) ** 2) / 4

    l3 = (2 * (hsim(x, x) - 1)**2 + (hsim(y, y) - 1)**2 + (hsim(k, k) - 1)**2) / 16 * 1
    l4 = torch.mean(2 * x ** 2 + y ** 2 + k **2, dim=1) / 16


    loss = l1 + l2 + l3 + l4
    return loss.mean()


def compute_similarity(ts, p=2):
    if ts.size(0) == 2:
        return -compute_distance(ts, p)
    if ts.size(0) == 4:
        g12, g13 = compute_distance(ts, p)
        return [torch.tensor([1.]) if g12[i] < g13[i] else torch.tensor([0.]) for i in range(ts.size(1))]

    raise ValueError('Failed to compute similarity!')

def compute_binary_similarity(ts):
    pos = torch.gt(ts, 0).int()
    if ts.size(0) == 2:
        return -1 * torch.sum(torch.abs(pos[0] - pos[1]), dim=-1)

    elif ts.size(0) == 4:
        # pos = torch.gt(ts, 0).int()
        # neg = torch.le(ts, 0).int()
        g12 = torch.sum(torch.abs(pos[0] - pos[1]), dim=-1)
        g13 = torch.sum(torch.abs(pos[2] - pos[3]), dim=-1)

        return [torch.tensor([1.]) if g12[i] < g13[i] else torch.tensor([0.]) for i in range(ts.size(1))]

def auc_score(label, predict):
    predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)
    label = (label + 1) / 2
    auc = metrics.roc_auc_score(label.astype(int), predict)

    return auc

def precision_recall_curve(label, predict):
    predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)
    label = (label + 1) / 2
    precision, recall, thresholds = metrics.precision_recall_curve(label, predict)

    return precision, recall, thresholds
    
def node_deg(edge_index, num_nodes):
    edge_weight = torch.ones((edge_index.size(1), ), dtype=torch.float32, device=device)
    deg = torch.zeros((num_nodes, ), dtype=torch.float32, device=device)
    row, col = edge_index
    for i, index in enumerate(row):
        deg[index] = deg[index] + edge_weight[index]

    return deg

def orthogonal_regularization(matrix):
    loss_orth = torch.tensor(0., dtype=torch.float32, device=device)
    weight_squared = torch.mm(matrix, matrix.transpose(0, 1))
    ones = torch.ones(matrix.size(0), matrix.size(0), dtype=torch.float32, device=device)
    diag = torch.eye(matrix.size(0), dtype=torch.float32, device=device)

    loss_orth += ((weight_squared * (ones - diag)) ** 2).sum()

    return loss_orth

def write_log(vectors, file):
    with open(file, 'a+') as f:
        for i in range(4):
            vec=' '.join(str(num)[:5] for num in vectors[i][:-1])
            f.write(vec)
            f.write('\n')
        f.write('\n')


def global_save_model(result, file, n_node, p_edge, path, model):
    data = np.loadtxt(file)

    if n_node==20 and p_edge==0.2 and result[0] > data[0] and result[1] > data[1]:
        torch.save({'model_state_dict': model.state_dict()}, path)
        data[0], data[1]= result[0], result[1]

    if n_node==20 and p_edge==0.5 and result[0] > data[2] and result[1] > data[3]:
        torch.save({'model_state_dict': model.state_dict()}, path)
        data[2], data[3]= result[0], result[1]

    if n_node==50 and p_edge==0.2 and result[0] > data[4] and result[1] > data[5]:
        torch.save({'model_state_dict': model.state_dict()}, path) 
        data[4], data[5]= result[0], result[1] 

    if n_node==50 and p_edge==0.5 and result[0] > data[6] and result[1] > data[7]:
        torch.save({'model_state_dict': model.state_dict()}, path)
        data[6], data[7]= result[0], result[1]

    np.savetxt(file, data, fmt='%.4e')

def plot_graphs(lists, name):
    nx_graphs = lists
    n_pair = len(nx_graphs) // 4
    plt.figure(figsize=(8, n_pair * 4))

    g1 = nx_graphs[0]
    g2 = nx_graphs[1]
    g3 = nx_graphs[2]
    ax = plt.subplot(1, 3, 1)

    pos = nx.drawing.spring_layout(g1)
    nx.draw_networkx(g1, pos=pos, ax=ax)
    ax.set_title('Graph 0')
    ax.axis('off')
    ax = plt.subplot(1, 3, 2)
    nx.draw_networkx(g2, pos=pos, ax=ax)
    ax.set_title('Graph 1')
    ax.axis('off')
    ax = plt.subplot(1, 3, 3)
    nx.draw_networkx(g3, pos=pos, ax=ax)
    ax.set_title('Graph 2')
    ax.axis('off')

    plt.savefig(name)
    plt.show()

if __name__ == "__main__":
    file = "./logs/log_tmp.txt"
    global_save_model(np.array([1, 1]), "./logs/log_tmp.txt", 50, 0.2)

    