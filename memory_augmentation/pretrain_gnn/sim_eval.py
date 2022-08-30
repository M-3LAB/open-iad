import numpy as np
import os
import torch
import time
from colorama import init, Fore

from gen_datasets import GraphGen, GraphData
from model import GNN
from tools import compute_distance, compute_loss, auc_score
from tools import compute_similarity, compute_binary_similarity
from tools import write_log
from tools import plot_graphs


def fill_data_to_device(data_batch):
    node = np.array(data_batch.node_features)
    edge = np.array(data_batch.edge_features)
    edge_index = np.array([data_batch.from_idx, data_batch.to_idx])
    graph_idx = np.array(data_batch.graph_idx)

    node = torch.from_numpy(node).float().to(device)
    edge = torch.from_numpy(edge).float().to(device)
    edge_index = torch.from_numpy(edge_index).long().to(device)
    graph_idx = torch.from_numpy(graph_idx).long().to(device)

    return node, edge, edge_index, graph_idx

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init(autoreset=True)

    is_print = False
    
    n_node = 20
    p_edge = 0.2
    batch_size = 100
    encode_dim = 64
    hidden_dim = 64
    n_prop_layer = 5
    g_repr_dim = 128
    n_test = 1000

    isTestBinary = 0

    model_path = "/ssd-sata1/wjb/code/open-ad/memory_augmentation/pretrain_gnn/model_20_0.2_128_0.8787_0.9160.pth"
    idx = 'eval'
    chack_vector = './logs/log_' + idx + '.txt'

    data_load = GraphGen(n_node, p_edge, 1, 2)
    data_train_iter = data_load.triplets(batch_size=batch_size)

    data_test_trip_iter = data_load.triplets_fixed(batch_size=batch_size, datasest_size=n_test)
    data_test_pair_iter = data_load.pairs_fixed(batch_size=batch_size, datasest_size=n_test)

    test_trip = [trip for trip in data_test_trip_iter]
    test_pair = [pair for pair in data_test_pair_iter]

    model = GNN(encode_dim, hidden_dim, n_prop_layer, g_repr_dim).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Load model, {}'.format(model_path))


    model.eval()

    # triplet acc
    scores = []
    log = True
    for trip in test_trip:
        data = fill_data_to_device(trip)

        with torch.no_grad():
            vectors = model(data)

            if is_print and log:
                write_log(vectors.view(4 * batch_size, -1).cpu().numpy(), chack_vector)
                log = False

        splits = vectors.view(-1, 4, g_repr_dim).permute(1, 0, 2)
        score = compute_similarity(splits)
        tmp = [x.cpu().detach().numpy() for x in score]
        scores.append(tmp)
        
    triplet_acc = np.mean(scores)

    # pair auc
    scores, labels = [], []
    for pair in test_pair:
        data_pair, label = pair
        data = fill_data_to_device(data_pair)

        with torch.no_grad():
            vectors = model(data)
            
        splits = vectors.view(-1, 2, g_repr_dim).permute(1, 0, 2)
        score = compute_similarity(splits).cpu().numpy()
            
        for s, l in zip(score, label):
            scores.append(s)
            labels.append(l)
        
    pair_auc = auc_score(np.squeeze(np.array(labels)), np.array(scores))

    print(Fore.GREEN + 'N: {}, P: {}, B: {}, pair_auc: {:.4f}, triplet_acc: {:.4f}'.format(
        n_node, p_edge, batch_size, pair_auc, triplet_acc))

