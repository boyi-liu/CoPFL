import math

import torch

from torch.utils.data import Subset, DataLoader
from alg.clusterbase import ClusterServer, ClusterClient
from utils.time_utils import time_record


def add_args(parser):
    parser.add_argument('--proxy_prop', type=float, default=0.1)
    parser.add_argument('--cluster_num', type=int, default=4)
    return parser.parse_args()

class Client(ClusterClient):
    def __init__(self, id, args):
        super().__init__(id, args)

        self.cluster_tensors = []
        self.new_cluster_id = 0

        dataset_size = len(self.dataset_train)

        sample_proportion = args.proxy_prop
        indices = torch.randperm(dataset_size).tolist()
        subset_indices = indices[:math.floor(sample_proportion * dataset_size)]
        self.proxy_dataset = Subset(self.dataset_train, subset_indices)
        self.proxy_loader = DataLoader(
            dataset=self.proxy_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=None,
        )

    @time_record
    def run(self):
        self.cluster()
        self.train()

    def cluster(self):
        # === retrieve training result ===
        loss_list = []

        for c_tensor in self.cluster_tensors:
            self.tensor2model(c_tensor)

            total_loss = 0.0
            for data in self.proxy_loader:
                X, y = self.preprocess(data)
                preds = self.model(X)
                loss = self.loss_func(preds, y)
                total_loss += loss.item()
            loss_list.append(total_loss / len(self.proxy_loader))

        # === refine cluster relationship ===
        res = loss_list.index(min(loss_list))
        self.tensor2model(self.cluster_tensors[res])
        self.new_cluster_id = res


class Server(ClusterServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)

        for client in self.clients:
            client_id = client.id
            client.cluster_id = client_id // (self.client_num // self.cluster_num + 1)

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.update_cluster_info()
        self.aggregate()

    def downlink(self):
        assert (len(self.sampled_clients) > 0)
        cluster_tensors = [cluster.model for cluster in self.cluster_list]
        for client in self.sampled_clients:
            client.cluster_tensors = cluster_tensors

    def update_cluster_info(self):
        for client in self.sampled_clients:
            client.cluster_id = client.new_cluster_id