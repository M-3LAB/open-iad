import numpy as np
import networkx as nx
import random
import copy
import contextlib
import collections
import time
from torch.utils.data import Dataset, DataLoader

GraphData = collections.namedtuple('GraphData', 
['from_idx', 'to_idx', 'node_features', 'edge_features', 'graph_idx', 'n_graphs', 'node_weights'])

@contextlib.contextmanager
def reset_random_state(seed):
    """This function creates a context that uses the given seed."""
    np_rnd_state = np.random.get_state()
    rnd_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed + 1)
    try:
        yield
    finally:
        random.setstate(rnd_state)
        np.random.set_state(np_rnd_state)

class GraphGen():
    def __init__(self, n_node, p_edge, n_change_pos=1, n_change_neg=2, k_list=20, permute=True):
        super(GraphGen, self).__init__()
        self.n_node = n_node
        self.p_edge = p_edge
        self.n_change_pos = n_change_pos
        self.n_change_neg = n_change_neg
        self.permute = permute
        self._idx = np.array([[[i, j] for i in range(j + 1, n_node)] for j in range(n_node)])
        self.idx = [item for sublist in self._idx for item in sublist]
        self.test_triplet_datasets = []
        self.test_pair_datasets = []

        # for visualization
        self._k_list = k_list

    def _permute_graph_nodes(self, g):
        """Permute node ordering of a graph, returns a new graph."""
        n = g.number_of_nodes()
        new_g = nx.Graph()
        new_g.add_nodes_from(range(n))
        perm = np.random.permutation(n)
        edges = g.edges()
        new_edges = []
        for x, y in edges:
            new_edges.append((perm[x], perm[y]))
        new_g.add_edges_from(new_edges)
        return new_g

    def _gen_graph(self):
        """Generate one graph."""
        n_nodes = np.random.randint(self.n_node, self.n_node + 1)
        p_edge = np.random.uniform(self.p_edge, self.p_edge)
        # do a little bit of filtering
        n_trials = 100
        for _ in range(n_trials):
            g = nx.erdos_renyi_graph(n_nodes, p_edge)
            if nx.is_connected(g):
                return g

        raise ValueError('Failed to generate a connected graph.')
    
    def _substitute_random_edges(self, g, n):
        """Substitutes n edges from graph g with another n randomly picked edges."""
        g = copy.deepcopy(g)
        n_nodes = g.number_of_nodes()
        edges = list(g.edges())
        # sample n edges without replacement
        e_remove = [edges[i] for i in np.random.choice(np.arange(len(edges)), n, replace=False)]
        edge_set = set(edges)
        e_add = set()
        while len(e_add) < n:
            e = np.random.choice(n_nodes, 2, replace=False)
            # make sure e does not exist and is not already chosen to be added
            if ((e[0], e[1]) not in edge_set and (e[1], e[0]) not in edge_set and
                    (e[0], e[1]) not in e_add and (e[1], e[0]) not in e_add):
                e_add.add((e[0], e[1]))

        for i, j in e_remove:
            g.remove_edge(i, j)
        for i, j in e_add:
            g.add_edge(i, j)
        return g

    def _create_graph(self, g, e_del, e_add):
        g = copy.deepcopy(g)
        n_nodes = g.number_of_nodes()
        edges = list(g.edges())
        edge_set = set(edges)
        e_t = list(edge_set - e_del)
        e_del_t = [e_t[i] for i in np.random.choice(np.arange(len(e_t)), 1, replace=False)]
        e_del.add(e_del_t[0])

        e_add_t = set()
        while len(e_add_t) < 1:
            e = np.random.choice(n_nodes, 2, replace=False)
            if ((e[0], e[1]) not in edge_set and (e[1], e[0]) not in edge_set and
                    (e[0], e[1]) not in e_add and (e[1], e[0]) not in e_add):
                e_add_t.add((e[0], e[1]))
                e_add.add((e[0], e[1]))

        for i, j in e_del:
            g.remove_edge(i, j)
        for i, j in e_add:
            g.add_edge(i, j)

        return g, e_del, e_add

    def _create_graph_list_strict_diff(self, g, n, is_replaced=False):
        g_init = copy.deepcopy(g)
        edges = list(g.edges())
        e_del = set()
        e_add = set()

        if n > len(edges):
            raise ValueError('Graph edit distance is larger than the edge number n of graphs')

        g_list = []
        for i in range(n):
            _gn, _e_del, _e_add = self._create_graph(g, e_del, e_add)
            # n_ged = nx.algorithms.similarity.graph_edit_distance(g_init, _gn)

            if is_replaced:
                if i == 0:
                    _g0 = g
                else:
                    # _g0 = self._substitute_random_edges(g_init, 1)
                    _g0 = self._permute_graph_nodes(g_init)  # isomorphism
            else:
                _g0 = g_init

            g_list.append(_g0)
            g_list.append(_gn)
            e_del = _e_del
            e_add = _e_add

        return g_list

    def _gen_triplet(self):
        """Generate one triplet of graphs."""
        g = self._gen_graph()
        if self.permute:
            permuted_g = self._permute_graph_nodes(g)
        else:
            permuted_g = g
        pos_g = self._substitute_random_edges(g, self.n_change_pos)
        neg_g = self._substitute_random_edges(g, self.n_change_neg)

        return permuted_g, pos_g, g, neg_g

    def _gen_pair(self, positive):
        """Generate one pair of graphs."""
        g = self._gen_graph()
        if self.permute:
            permuted_g = self._permute_graph_nodes(g)
        else:
            permuted_g = g
        n_change = self.n_change_pos if positive else self.n_change_neg
        g_ = self._substitute_random_edges(g, n_change)

        return permuted_g, g_

    def _get_list(self, is_replaced):
        g = self._gen_graph()
        g_list = self._create_graph_list_strict_diff(g, self._k_list, is_replaced)
        return g_list

    def _get_list_v1(self):
        g = self._gen_graph()
        g0 = g
        g_list = []
        for _ in range(self._k_list):
            g_tmp = self._substitute_random_edges(g, 1)
            g_list.append(g0)
            g_list.append(g_tmp)
            g = g_tmp

        return g_list

    def triplets(self, batch_size=20):
        """Yields batches of triplet data."""
        while True:
            triplets = []
            for _ in range(batch_size):
                g1, g2, g3, g4 = self._gen_triplet()
                triplets.append([g1, g2, g3, g4])
            yield pack_graph(triplets)

    def pairs(self, batch_size=20):
        """Yields batches of pair data."""
        while True:
            pairs, labels = [], []
            positive = True
            for _ in range(batch_size):
                g1, g2 = self._gen_pair(positive)
                pairs.append([g1, g2])
                labels.append(1 if positive else -1)
                positive = not positive
            packed_graphs = pack_graph(pairs)
            labels = np.array(labels, dtype=np.int32)
        yield packed_graphs, labels

    def triplets_fixed(self, batch_size=20, datasest_size=1000):
        """Yields batches of triplet data."""
        triplets = []
        with reset_random_state(seed=1234):
            for _ in range(datasest_size):
                g1, g2, g3, g4 = self._gen_triplet()
                triplets.append([g1, g2, g3, g4])
        ptr = 0
        self.test_triplet_datasets = triplets
        while ptr + batch_size <= len(triplets):
            batch_graphs = triplets[ptr:ptr + batch_size]
            yield pack_graph(batch_graphs)
            ptr += batch_size

    def pairs_fixed(self, batch_size=20, datasest_size=1000):
        """Yields batches of pair data."""
        pairs, labels = [], []
        positive = True
        with reset_random_state(seed=1234):
            for _ in range(datasest_size):
                g1, g2 = self._gen_pair(positive)
                pairs.append([g1, g2])
                labels.append(1 if positive else -1)
                positive = not positive
        labels = np.array(labels, dtype=np.int32)
        self.test_pair_datasets = [pairs, labels]
        ptr = 0
        while ptr + batch_size <= len(pairs):
            batch_graphs = pairs[ptr:ptr + batch_size]
            packed_batch = pack_graph(batch_graphs)
            yield packed_batch, labels[ptr:ptr + batch_size]
            ptr += batch_size

    def lists(self, batch_size, is_replaced=False):
        """Yield lists."""
        lists = []
        with reset_random_state(seed=1234):
            for _ in range(batch_size):
                g_list = self._get_list(is_replaced)
                batch_graphs = pack_graph([g_list])
                lists.append(batch_graphs)
        return lists

    def lists_v1(self, batch_size):
        """Yield lists."""
        lists = []
        with reset_random_state(seed=1234):
            for _ in range(batch_size):
                g_list = self._get_list_v1()
                batch_graphs = pack_graph([g_list])
                lists.append(batch_graphs)
        return lists

    def get_datasets_pairs(self):
        return self.test_pair_datasets

    def get_datasets_triplets(self):
        return self.test_triplet_datasets

def pack_graph(graphs):
    """Pack a batch of graphs into a single `GraphData` instance."""
    from_idx = []
    to_idx = []
    graph_idx = []
    node_weights = []
    n_total_nodes = 0
    n_total_edges = 0

    graphs = [val for sublist in graphs for val in sublist]
    for i, g in enumerate(graphs):
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
        edges = np.array(g.edges(), dtype=np.int32)
        # shift the node indices for the edges
        from_idx.append(edges[:, 0] + n_total_nodes)
        to_idx.append(edges[:, 1] + n_total_nodes)
        graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)
        
        # cluster_coeff = [g.degree[i] for i in range(g.number_of_nodes())]
        # cluster_coeff /= np.sum(cluster_coeff)
        # node_weights.append(cluster_coeff)

        n_total_nodes += n_nodes
        n_total_edges += n_edges

    node_features = np.ones((n_total_nodes, 1), dtype=np.float32)
    edge_features = np.ones((n_total_edges, 1), dtype=np.float32)
    from_idx = np.concatenate(from_idx, axis=0)
    to_idx = np.concatenate(to_idx, axis=0)
    graph_idx = np.concatenate(graph_idx, axis=0)

    packages = GraphData(from_idx=from_idx, to_idx=to_idx, node_features=node_features, 
    edge_features=edge_features, graph_idx=graph_idx, n_graphs=len(graphs), node_weights=node_weights)

    return packages

class FastTriData(Dataset):
    def __init__(self, n_node, p_egde, n_change_pos=1, n_change_neg=2):
        self.data_generator = GraphGen(n_node, p_egde, n_change_pos, n_change_neg)

    def __getitem__(self, ind):
        g1, g2, g3, g4 = self.data_generator._gen_triplet()
        return [g1, g2, g3, g4]

    def __len__(self):
        return int(1e8)


def graph_collate_fn(data):
    triplets = []
    g1, g2, g3, g4 = zip(*data)
    for i in range(len(g1)):
        triplets.append([g1[i], g2[i], g3[i], g4[i]])
    
    return pack_graph(triplets)


def get_graph_triplet_loader(n_node, p_egde, n_change_pos, n_change_neg, batch_size, num_workers=8):
    traindata = FastTriData(n_node, p_egde, n_change_pos, n_change_neg)
    dataloader = DataLoader(traindata, batch_size=batch_size, num_workers=num_workers, collate_fn=graph_collate_fn)
    return dataloader