import torch

from alg.base import BaseClient, BaseServer


class ClusterClient(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.cluster_id = 0

    def clone_model(self, target):
        target_cluster = target.cluster_list[self.cluster_id]
        p_tensor = target_cluster.model_tensor
        self.tensor2model(p_tensor)


class ClusterServer(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)

        self.cluster_num = args.cluster_num
        assert self.cluster_num > 0
        self.cluster_list = [Cluster(idx, self.model2tensor()) for idx in range(self.cluster_num)]
        self.cluster_list[0].clients = [idx for idx in range(self.client_num)]

    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        def nan_to_zero(tensor):
            return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        for cluster in self.cluster_list:
            cluster.received_params = [nan_to_zero(client.model2tensor())
                                       for client in self.sampled_clients
                                       if client.cluster_id == cluster.id]

    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        for cluster in self.cluster_list:
            total_samples = sum(len(client.dataset_train) for client in self.sampled_clients if client.cluster_id == cluster.id)
            weights = [len(client.dataset_train) / total_samples for client in self.sampled_clients if client.cluster_id == cluster.id]

            cluster.received_params = [params * weight for weight, params in zip(weights, cluster.received_params)]
            avg_tensor = sum(cluster.received_params)
            cluster.model_tensor = avg_tensor


class Cluster:
    def __init__(self, id, model_tensor):
        self.id = id
        self.clients = []
        self.model = model_tensor
        self.received_params = []