import numpy as np
import os
import torch
import time
import argparse
import yaml
import glob
import random

from gen_datasets import GraphGen, get_graph_triplet_loader
from model import GNN
from tools import compute_distance, compute_loss, compute_similarity, auc_score, precision_recall_curve
from colorama import init, Fore


def fill_data_to_device(data_batch, device):
    node = np.array(data_batch.node_features)
    edge = np.array(data_batch.edge_features)
    edge_index = np.array([data_batch.from_idx, data_batch.to_idx])
    graph_idx = np.array(data_batch.graph_idx)

    node = torch.from_numpy(node).float().to(device)
    edge = torch.from_numpy(edge).float().to(device)
    edge_index = torch.from_numpy(edge_index).long().to(device)
    graph_idx = torch.from_numpy(graph_idx).long().to(device)

    return node, edge, edge_index, graph_idx

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_node", "-n", type=int, default=20)
    parser.add_argument("--p_edge", "-p", type=float, default=0.2)
    parser.add_argument("--batch_size", "-b", type=int, default=40)
    parser.add_argument("--encode_dim", "-ed", type=int, default=64)
    parser.add_argument("--hidden_dim", "-hd", type=int, default=64)
    parser.add_argument("--n_prop_layer", "-g", type=int, default=5)
    parser.add_argument("--g_repr_dim", "-d", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr_step", type=int, default=4000)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--n_iteration", type=int, default=100000)
    parser.add_argument("--graph_coffin", type=float, default=0.001) # 20-0.2:0.014, 50-0.2:0.001
    parser.add_argument("--dmap_coffin", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--gpu", type=str, default="1")
    parser.add_argument("--save-model", type=bool, default=True)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default='./work_dir')

    return parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def save_arg(para_dict, file_path):
    with open('{}/config.yaml'.format(file_path), 'w') as f:
        yaml.dump(para_dict, f)

def record_path(args):
    # mkdir ./work_dir
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    
    localtime = time.asctime(time.localtime(time.time()))
    file_path = '{}/{}'.format(args.work_dir, localtime)
    os.makedirs(file_path)

    return file_path   

def hgnn(): 
    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(seed=1234)
    init(autoreset=True)

    file_path = record_path(args)
    save_arg(args, file_path)

    test_result = '{}/log_test_{}_{}.txt'.format(file_path, args.n_node, args.p_edge)
    init_step = 0

    best_pair_auc = 0
    best_triplet_acc = 0

    begin = time.time()
    data_load = GraphGen(args.n_node, args.p_edge, 1, 2, permute=True)
    data_train_iter = iter(get_graph_triplet_loader(args.n_node, args.p_edge, 1, 2, batch_size=args.batch_size, num_workers=8))

    data_test_trip_iter = data_load.triplets_fixed(batch_size=args.batch_size, datasest_size=args.n_test)
    data_test_pair_iter = data_load.pairs_fixed(batch_size=args.batch_size, datasest_size=args.n_test)

    test_trip = [trip for trip in data_test_trip_iter]
    test_pair = [pair for pair in data_test_pair_iter]

    print('Load datasets: {:.2f} s'.format(time.time() - begin))

    model = GNN(args.encode_dim, args.hidden_dim, args.n_prop_layer, args.g_repr_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    mse = torch.nn.MSELoss()
    print('Load model: {:.2f} s'.format(time.time() - begin))


    if args.load_model != None:
        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Load model, done')

    run = time.time()
    for e in range(init_step, args.n_iteration):
        model.train()
        optimizer.zero_grad()

        data_batch = next(data_train_iter)
        data = fill_data_to_device(data_batch, device)

        vectors = model(data)

        graph = torch.norm(vectors, dim=1).mean() / args.n_node
        splits = vectors.view(args.batch_size, 4, args.g_repr_dim).permute(1, 0, 2)
        loss_tri = compute_loss(splits, margin=args.margin)    

        loss = loss_tri + graph * args.graph_coffin

        loss.backward()
        optimizer.step()
        scheduler.step()

        g12, g13 = compute_distance(splits)
        pos = g12.mean().item()
        neg = g13.mean().item()
        lr = scheduler.get_last_lr()

        if (e + 1) % 50 == 0:
            print("lr：{:.0e}  loss: {:2.4f}  graph: {: >7.4f}  pos: {: 2.4f}  neg: {: 2.4f}  diff: {: 2.4f}  time: {:.2f}".format(
                lr[0], loss, graph, pos, neg, pos - neg, time.time() - run))
            run = time.time()

        # testing
        if (e + 1) % 200 == 0:
            model.eval()
            start = time.time()
            
            # triplet acc
            scores = []
            log = True
            for trip in test_trip:
                data = fill_data_to_device(trip, device)

                with torch.no_grad():
                    vectors = model(data)

                splits = vectors.view(-1, 4, args.g_repr_dim).permute(1, 0, 2)
                score = compute_similarity(splits)
                tmp = [x.cpu().detach().numpy() for x in score]
                scores.append(tmp)

            triplet_acc = np.mean(scores)

            # pair auc
            scores, labels = [], []
            for pair in test_pair:
                data_pair, label = pair
                data = fill_data_to_device(data_pair, device)

                with torch.no_grad():
                    vectors = model(data)
                    
                splits = vectors.view(-1, 2, args.g_repr_dim).permute(1, 0, 2)
                score = compute_similarity(splits).cpu().numpy()

                for s, l in zip(score, label):
                    scores.append(s)
                    labels.append(l)
                
            pair_auc = auc_score(np.squeeze(np.array(labels)), np.array(scores))

            infor = '---> step: {:<5d} lr：{:.0e} N: {} P: {} B: {} M: {} time: {:.4f} pair_auc: {:.4f} triplet_acc: {:.4f}'.format(
                e + 1, lr[0], args.n_node, args.p_edge, args.batch_size, args.n_test, time.time() - start, pair_auc, triplet_acc)
            print(Fore.GREEN + infor)
            
            with open(test_result, 'a+') as f:
                f.write(infor + '\n')

            run = time.time()

            if pair_auc > best_pair_auc and triplet_acc > best_triplet_acc:
                for file in glob.glob('{}/*.pth'.format(file_path)):
                    os.remove(file) 
                best_pair_auc = pair_auc
                best_triplet_acc = triplet_acc

                model_save_path = '{}/model_{}_{}_{}_{:.4f}_{:.4f}.pth'.format(file_path, args.n_node, args.p_edge, args.g_repr_dim, best_pair_auc, best_triplet_acc)
                torch.save({'model_state_dict': model.state_dict()}, model_save_path)
                                                                                                                                                                                                                                                            


if __name__ == "__main__":
    hgnn()