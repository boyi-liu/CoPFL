import numpy as np
import torch
import cvxpy as cp

from alg.base import BaseClient, BaseServer

def add_args(parser):
    parser.add_argument('--alpha', type=float, default=0.8, help="Alpha in weight optimization")
    parser.add_argument('--lam', type=float, default=0.01, help="Lambda in local training")
    return parser.parse_args()

class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.lam = args.lam

    def run(self):
        self.train()

    def clone_model(self, target):
        p_tensor = target.client_models[self.id]
        self.tensor2model(p_tensor)

    def train(self):
        w_last = self.model2tensor()

        total_loss = 0.0
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)
                preds = self.model(X)

                loss = self.loss_func(preds, y)

                w_cur = torch.cat([param.view(-1) for param in self.model.parameters()], dim=0)
                loss += self.lam * torch.dot(w_cur, w_last) / torch.linalg.norm(w_cur)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item()

        self.metric['loss'] = total_loss / len(self.loader_train)


class Server(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.sims = np.zeros([self.client_num, self.client_num])
        self.graph_w = torch.zeros(self.client_num, self.client_num)
        self.alpha = args.alpha
        self.client_models = [c.model2tensor() for c in self.clients]

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.update_sims()
        self.update_w()
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
                self.sims[idx_i, idx_j] = self.sims[idx_j, idx_i] = -torch.nn.functional.cosine_similarity(
                    self.client_models[idx_i],
                    self.client_models[idx_j],
                    dim=0)

    # https://github.com/MediaBrain-SJTU/pFedGraph/blob/main/pfedgraph_cosine/utils.py
    def update_w(self):
        total_samples = sum(len(client.dataset_train) for client in self.clients)
        w_all = [len(c.dataset_train) / total_samples for c in self.clients]

        for idx, c in enumerate(self.sampled_clients):
            sims = self.sims[c.id]
            n = len(self.clients)

            p = np.array(w_all)
            P = self.alpha * np.identity(n)
            P = cp.atoms.affine.wraps.psd_wrap(P)
            G = - np.identity(n)
            h = np.zeros(n)
            A = np.ones((1, n))
            b = np.ones(1)
            d =  sims
            q = d - 2 * self.alpha * p
            x = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                              [G @ x <= h,
                               A @ x == b])
            prob.solve()
            self.graph_w[idx] = torch.Tensor(x.value)
            self.graph_w[:, idx] = torch.Tensor(x.value)

    def aggregate(self):
        for idx, client in enumerate(self.sampled_clients):
            w_aggr = self.graph_w[client.id]
            aggr_tensor = sum([w * tensor for w, tensor in zip(w_aggr, self.client_models)])

            self.client_models[client.id] = aggr_tensor