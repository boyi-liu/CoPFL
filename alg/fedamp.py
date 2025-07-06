import torch
from utils.time_utils import time_record
from alg.base import BaseClient, BaseServer

def add_args(parser):
    parser.add_argument('--lam', type=float, default=1, help="Lambda in local training")
    parser.add_argument('--sigma', type=float, default=10, help="Sigma in server aggregation")
    parser.add_argument('--xiii', type=float, default=0.7, help="xi_{ii}")
    return parser.parse_args()


class Client(BaseClient):
    @time_record
    def __init__(self, id, args):
        super().__init__(id, args)
        self.lam = args.lam

    def run(self):
        self.train()

    def clone_model(self, target):
        # p_tensor = target.client_models[self.id]
        # self.tensor2model(p_tensor)
        pass

    def train(self):
        w_last = self.server.client_models[self.id]

        total_loss = 0.0
        for _ in range(self.epoch):
            for data in self.loader_train:
                X, y = self.preprocess(data)
                preds = self.model(X)

                loss = self.loss_func(preds, y)

                w_cur = torch.cat([param.view(-1) for param in self.model.parameters()], dim=0)
                loss += 0.5 * self.lam * torch.norm(w_cur - w_last, p=2)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                total_loss += loss.item()

        self.metric['loss'] = total_loss / len(self.loader_train)


class Server(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.sims = torch.zeros(self.client_num, self.client_num)
        self.sigma = args.sigma
        self.xiii = args.xiii
        self.client_models = [c.model2tensor() for c in self.clients]

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.update_sims()
        self.aggregate()

    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        def nan_to_zero(tensor):
            return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        for c in self.sampled_clients:
            self.client_models[c.id] = nan_to_zero(c.model2tensor())

    def update_sims(self):
        for c_i in self.sampled_clients:
            for c_j in self.clients:
                idx_i = c_i.id
                idx_j = c_j.id
                self.sims[idx_i, idx_j] = self.sims[idx_j, idx_i] = torch.nn.functional.cosine_similarity(self.client_models[idx_i],
                                                                                                          self.client_models[idx_j],
                                                                                                          dim=0)

    def aggregate(self):
        res = []
        for idx, c in enumerate(self.sampled_clients):
            w_aggr = self.sims[c.id]

            w_aggr = torch.exp(w_aggr * self.sigma)
            w_aggr /= (torch.sum(w_aggr)-w_aggr[idx])
            w_aggr *= (1-self.xiii)
            w_aggr[idx] = self.xiii
            res.append(sum([w * tensor for w, tensor in zip(w_aggr, self.client_models)]))

        for c, aggr_tensor in zip(self.sampled_clients, res):
            self.client_models[c.id] = aggr_tensor