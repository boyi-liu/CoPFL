import numpy as np
import torch
import cvxpy as cp

from alg.base import BaseClient, BaseServer
from sklearn.decomposition import PCA

from utils.time_utils import time_record

k_principal = 3
pca = PCA(n_components=k_principal)

def add_args(parser):
    parser.add_argument('--alpha', type=float, default=1.5, help="Alpha in weight optimization")
    parser.add_argument('--beta', type=float, default=1.4, help="Beta in weight optimization")
    parser.add_argument('--lam', type=float, default=0.01, help="Lambda in local training")
    return parser.parse_args()

class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.lam = args.lam
        self.basis = None

    @time_record
    def run(self):
        self.train()

    def clone_model(self, target):
        p_tensor = target.client_models[self.id]
        self.tensor2model(p_tensor)

    def train(self):
        w_last = self.model2tensor()

        reps = []

        # === train ===
        batch_loss = []
        for epoch in range(self.epoch):
            for idx, data in enumerate(self.loader_train):
                X, y = self.preprocess(data)
                preds, rep = self.model(X, return_feat=True)

                reps.append(rep)

                loss = self.loss_func(preds, y)

                w_cur = torch.cat([param.view(-1) for param in self.model.parameters()], dim=0)
                loss -= self.lam * torch.nn.functional.cosine_similarity(w_cur, w_last, dim=0)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                batch_loss.append(loss.item())

        pca.fit_transform(torch.cat(reps).detach().cpu().numpy())
        self.basis = pca.components_

        # === record loss ===
        self.metric['loss'].append(sum(batch_loss) / len(batch_loss))


class Server(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.sims = np.zeros([self.client_num, self.client_num])
        self.comps = np.zeros([self.client_num, self.client_num])
        self.graph_w = torch.zeros(self.client_num, self.client_num)
        self.alpha = args.alpha
        self.beta = args.beta
        self.client_models = [c.model2tensor() for c in self.clients]
        self.client_basis = [None for _ in self.clients]

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.update_sims()
        self.update_comps()
        self.update_w()
        self.aggregate()

    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        def nan_to_zero(tensor):
            return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        for c in self.sampled_clients:
            self.client_models[c.id] = nan_to_zero(c.model2tensor())
            self.client_basis[c.id] = c.basis

    def update_sims(self):
        for c_i in self.sampled_clients:
            for c_j in self.clients:
                idx_i = c_i.id
                idx_j = c_j.id
                self.sims[idx_i, idx_j] = self.sims[idx_j, idx_i] = torch.nn.functional.cosine_similarity(
                    self.client_models[idx_i],
                    self.client_models[idx_j],
                    dim=0)

    def update_comps(self):
        k = self.sampled_clients[0].basis.shape[0]

        for c_i in self.sampled_clients:
            for c_j in self.clients:
                idx_i = c_i.id
                idx_j = c_j.id
                if self.client_basis[idx_j] is not None:
                    phi = compute_principal_angles(self.client_basis[idx_i], self.client_basis[idx_j])
                    self.comps[idx_i, idx_j] = self.comps[idx_j, idx_i] = np.cos((1 / k) * np.sum(phi))

    # https://github.com/MediaBrain-SJTU/pFedGraph/blob/main/pfedgraph_cosine/utils.py
    def update_w(self):
        total_samples = sum(len(client.dataset_train) for client in self.clients)
        w_all = [len(c.dataset_train) / total_samples for c in self.clients]

        for idx, c in enumerate(self.sampled_clients):
            sims = self.sims[c.id]
            comps = self.comps[c.id]
            n = len(self.clients)

            p = np.array(w_all)
            P = self.alpha * np.identity(n)
            P = cp.atoms.affine.wraps.psd_wrap(P)
            G = - np.identity(n)
            h = np.zeros(n)
            A = np.ones((1, n))
            b = np.ones(1)
            s = sims
            c = comps
            q = -self.alpha * c - self.beta * s - 2 * p
            x = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                              [G @ x <= h,
                               A @ x == b])
            prob.solve()
            self.graph_w[idx] = torch.Tensor(x.value)
            # self.graph_w[:, idx] = torch.Tensor(x.value)


    def aggregate(self):
        res = [sum([w * tensor for w, tensor in zip(self.graph_w[c.id], self.client_models)])
               for c in self.sampled_clients]

        for client, aggr_tensor in zip(self.clients, res):
            self.client_models[client.id] = aggr_tensor


# https://github.com/yankd22/FedSaC/blob/main/FedSaC/utils.py
def compute_principal_angles(A, B):
    assert A.shape[0] == B.shape[0], "A and B must have the same number of vectors"

    k = A.shape[0]
    norm_A = np.linalg.norm(A, axis=1)[:, np.newaxis]
    norm_B = np.linalg.norm(B, axis=1)
    dot_product = np.dot(A, B.T)
    cosine_matrix = dot_product / (norm_A * norm_B)
    cos_phi_values = []

    for _ in range(k):
        i, j = np.unravel_index(np.argmax(cosine_matrix, axis=None), cosine_matrix.shape)
        cos_phi_values.append(cosine_matrix[i, j])
        cosine_matrix[i, :] = -np.inf
        cosine_matrix[:, j] = -np.inf
    phi = np.arccos(np.clip(cos_phi_values, -1, 1))

    return phi