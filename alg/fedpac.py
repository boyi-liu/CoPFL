import copy
import torch
import torch.nn as nn
import numpy as np
import cvxpy as cvx

from collections import defaultdict
from base import BaseClient, BaseServer

p_keys = ['fc']

def add_args(parser):
    parser.add_argument('--p_epoch', type=int, default=1, help="Personalized epoch")
    parser.add_argument('--lam', type=float, default=1, help="Lambda")
    return parser.parse_args()

class Client(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.p_epoch = args.p_epoch
        self.class_num = args.class_num
        self.lam = args.lam

        self.protos = None
        self.global_protos = None
        self.mu = None
        self.V = None

    def run(self):
        self.statistics_extraction()
        self.p_train()
        self.train()
        self.collect_protos()

    def p_train(self):
        # NOTE: freeze base, update head
        for name, param in self.model.named_parameters():
            param.requires_grad = True if name.split('.')[0] in p_keys else False

        for epoch in range(self.p_epoch):
            for data in self.loader_train:
                X, y = self.preprocess(data)
                preds = self.model(X)

                loss = self.loss_func(preds, y)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def train(self):
        # NOTE: freeze head, update base
        for name, param in self.model.named_parameters():
            param.requires_grad = False if name.split('.')[0] in p_keys else True

        for epoch in range(self.epoch):
            for data in self.loader_train:
                X, y = self.preprocess(data)
                preds, rep = self.model(X, return_feat=True)

                loss = self.loss_func(preds, y)

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += nn.MSELoss(proto_new, rep) * self.lam

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()


    # https://github.com/JianXu95/FedPAC/blob/main/methods/fedpac.py#L126
    def statistics_extraction(self):
        for data in self.loader_train:
            x, y = self.preprocess(data)
            with torch.no_grad():
                _, rep = self.model(x, return_feat=True)
                rep = rep.detach()
            break
        d = rep.shape[1]
        feature_dict = {}
        with torch.no_grad():
            for data in self.loader_train:
                x, y = self.preprocess(data)

                _, features = self.model(x, return_feat=True)
                features = features.detach()
                feat_batch = features.clone().detach()
                for i in range(len(y)):
                    yi = y[i].item()
                    if yi in feature_dict.keys():
                        feature_dict[yi].append(feat_batch[i, :])
                    else:
                        feature_dict[yi] = [feat_batch[i, :]]
        for k in feature_dict.keys():
            feature_dict[k] = torch.stack(feature_dict[k])

        py = torch.zeros(self.class_num)
        for x, y in self.loader_train:
            for yy in y:
                py[yy.item()] += 1
        py = py / torch.sum(py)
        py2 = py.mul(py)
        v = 0
        h_ref = torch.zeros((self.class_num, d), device=self.device)
        for k in range(self.class_num):
            if k in feature_dict.keys():
                feat_k = feature_dict[k]
                num_k = feat_k.shape[0]
                feat_k_mu = feat_k.mean(dim=0)
                h_ref[k] = py[k] * feat_k_mu
                v += (py[k] * torch.trace((torch.mm(torch.t(feat_k), feat_k) / num_k))).item()
                v -= (py2[k] * (torch.mul(feat_k_mu, feat_k_mu))).sum().item()
        v = v / len(self.dataset_train)

        self.V = v
        self.mu = h_ref

    # https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/clients/clientpac.py#L131
    def collect_protos(self):
        self.model.eval()
        protos = defaultdict(list)
        with torch.no_grad():
            for i, data in enumerate(self.loader_train):
                x, y = self.preprocess(data)
                _, rep = self.model(x, return_feat=True)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)

class Server(BaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.received_protos = []
        self.received_heads = []
        self.global_protos = None
        self.mus = []
        self.Vs = []

    def run(self):
        self.sample()
        self.downlink()
        self.client_update()
        self.uplink()
        self.aggregate()

    def downlink(self):
        super().downlink()
        if self.global_protos is None: return
        for client in self.sampled_clients:
            client.global_protos = copy.deepcopy(self.global_protos)

    def uplink(self):
        super().uplink()
        self.received_protos = [client.protos for client in self.sampled_clients]
        self.mus = [client.mu for client in self.sampled_clients]
        self.Vs = [client.V for client in self.sampled_clients]
        self.received_heads = [client.model2tensor(params=client.p_params) for client in self.sampled_clients]

    def aggregate(self):
        super().aggregate()
        proto_aggregation(self.received_protos)

        head_weights = solve_quadratic(len(self.sampled_clients), self.Vs, self.mus)
        for idx, client in enumerate(self.sampled_clients):
            print(f"Client {client.id} has weights: {head_weights[idx]}")
            weighted_head_tensors = [h * w for h, w in zip(self.received_heads, head_weights[idx])]
            client.tensor2model(sum(weighted_head_tensors), params=client.p_params)


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label

# https://github.com/JianXu95/FedPAC/blob/main/tools.py#L94
def solve_quadratic(num_users, Vars, Hs):
    device = Hs[0][0].device
    num_cls = Hs[0].shape[0]  # number of classes
    d = Hs[0].shape[1]  # dimension of feature representation
    avg_weight = []
    for i in range(num_users):
        # ---------------------------------------------------------------------------
        # variance ter
        v = torch.tensor(Vars, device=device)
        # ---------------------------------------------------------------------------
        # bias term
        h_ref = Hs[i]
        dist = torch.zeros((num_users, num_users), device=device)
        for j1, j2 in pairwise(tuple(range(num_users))):
            h_j1 = Hs[j1]
            h_j2 = Hs[j2]
            h = torch.zeros((d, d), device=device)
            for k in range(num_cls):
                h += torch.mm((h_ref[k] - h_j1[k]).reshape(d, 1), (h_ref[k] - h_j2[k]).reshape(1, d))
            dj12 = torch.trace(h)
            dist[j1][j2] = dj12
            dist[j2][j1] = dj12

        # QP solver
        p_matrix = torch.diag(v) + dist
        p_matrix = p_matrix.cpu().numpy()  # coefficient for QP problem
        evals, evecs = torch.linalg.eig(torch.tensor(p_matrix))

        # for numerical stablity
        p_matrix_new = 0
        for ii in range(num_users):
            if evals[ii].real >= 0.01:
                p_matrix_new += evals[ii].real * torch.mm(evecs[:, ii].reshape(num_users, 1),
                                                          evecs[:, ii].reshape(1, num_users))
        p_matrix = p_matrix_new.numpy() if not np.all(np.linalg.eigvals(p_matrix) >= 0.0) else p_matrix

        # solve QP
        alpha = 0
        eps = 1e-3
        if np.all(np.linalg.eigvals(p_matrix) >= 0):
            alphav = cvx.Variable(num_users)
            obj = cvx.Minimize(cvx.quad_form(alphav, p_matrix))
            prob = cvx.Problem(obj, [cvx.sum(alphav) == 1.0, alphav >= 0])
            prob.solve()
            alpha = alphav.value
            alpha = [(i) * (i > eps) for i in alpha]  # zero-out small weights (<eps)
        else:
            alpha = None  # if no solution for the optimization problem, use local classifier only

        avg_weight.append(alpha)

    return avg_weight

# https://github.com/JianXu95/FedPAC/blob/main/tools.py#L10
def pairwise(data):
    n = len(data)
    for i in range(n):
        for j in range(i, n):
            yield (data[i], data[j])