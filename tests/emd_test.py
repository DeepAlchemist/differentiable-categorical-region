import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# from qpth.qp import QPFunction

class DeepEMD(nn.Module):

    def __init__(self, args, mode='meta'):
        super().__init__()

        self.mode = mode
        self.args = args

        self.encoder = resnet50(pretrained=True)

        if self.mode == 'pre_train':
            self.fc = nn.Linear(640, self.args.num_class)

    def forward(self, input):
        if self.mode == 'meta':
            support, query = input
            return self.emd_forward_1shot(support, query)

        elif self.mode == 'pre_train':
            return self.pre_train_forward(input)

        elif self.mode == 'encoder':
            if self.args.deepemd == 'fcn':
                dense = True
            else:
                dense = False
            return self.encode(input, dense)
        else:
            raise ValueError('Unknown mode')

    def pre_train_forward(self, input):
        return self.fc(self.encode(input, dense=False).squeeze(-1).squeeze(-1))

    def get_weight_vector(self, q, anchor):
        """
        Args:
            q (4D Tensor): (M c h1 w1)
            anchor (4D Tensor): (N c h2 w2)

        Returns:
            combination (3D Tensor): (M N h1*w1)
        """
        M = q.shape[0]
        N = anchor.shape[0]

        anchor = F.adaptive_avg_pool2d(anchor, [1, 1])
        anchor = anchor.repeat(1, 1, q.shape[2], q.shape[3])

        q = q.unsqueeze(1)
        anchor = anchor.unsqueeze(0)

        q = q.repeat(1, N, 1, 1, 1)
        anchor = anchor.repeat(M, 1, 1, 1, 1)

        combination = (q * anchor).sum(2)
        combination = combination.view(M, N, -1)
        combination = F.relu(combination) + 1e-3
        return combination

    def emd_forward_1shot(self, proto, query):
        proto = proto.squeeze(0)

        weight_1 = self.get_weight_vector(query, proto)  # (N way h2*w2)
        weight_2 = self.get_weight_vector(proto, query)  # (way N h1*w1)

        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)

        similarity_map = self.get_similarity_map(proto, query)  # (N way h2*w2 h1*w1)
        if self.args.solver == 'opencv' or (not self.training):
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='opencv')
        else:
            logits = self.get_emd_distance(similarity_map, weight_1, weight_2, solver='qpth')
        return logits

    def get_sfc(self, support):
        support = support.squeeze(0)
        # init the proto
        SFC = support.view(self.args.shot, -1, 640, support.shape[-2], support.shape[-1]).mean(dim=0).clone().detach()
        SFC = nn.Parameter(SFC.detach(), requires_grad=True)

        optimizer = torch.optim.SGD([SFC], lr=self.args.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0)

        # crate label for finetune
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        label_shot = label_shot.type(torch.cuda.LongTensor)

        with torch.enable_grad():
            for k in range(0, self.args.sfc_update_step):
                rand_id = torch.randperm(self.args.way * self.args.shot).cuda()
                for j in range(0, self.args.way * self.args.shot, self.args.sfc_bs):
                    selected_id = rand_id[j: min(j + self.args.sfc_bs, self.args.way * self.args.shot)]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.emd_forward_1shot(SFC, batch_shot.detach())
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC

    def get_emd_distance(self, similarity_map, weight_1, weight_2, solver='opencv', temperature=12.5):
        r"""
        Args:
            similarity_map: (N way h2*w2 h1*w1)
            weight_1: (N way h2*w2)
            weight_2: (way N h1*w1)
            solver:

        Returns:
            logits: (N way)
        """
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node = weight_1.shape[-1]

        if solver == 'opencv':  # use openCV solver
            for i in range(num_query):
                for j in range(num_proto):
                    _, flow = opencv_emd(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])
                    # flow is of shape (h2*w2 h1*w1)
                    similarity_map[i, j, :, :] = (similarity_map[i, j, :, :]) * torch.from_numpy(flow).cuda()

            temperature = (temperature / num_node)
            logits = similarity_map.sum(-1).sum(-1) * temperature
            return logits
        elif solver == 'qpth':
            weight_2 = weight_2.permute(1, 0, 2)
            similarity_map = similarity_map.view(num_query * num_proto, similarity_map.shape[-2],
                                                 similarity_map.shape[-1])
            weight_1 = weight_1.view(num_query * num_proto, weight_1.shape[-1])
            weight_2 = weight_2.reshape(num_query * num_proto, weight_2.shape[-1])

            _, flows = qpth_emd(1 - similarity_map, weight_1, weight_2, form=self.args.form, l2_strength=self.args.l2_strength)

            logits = (flows * similarity_map).view(num_query, num_proto, flows.shape[-2], flows.shape[-1])
            temperature = (self.args.temperature / num_node)
            logits = logits.sum(-1).sum(-1) * temperature
        else:
            raise ValueError('Unknown Solver')

        return logits

    def normalize_feature(self, x):
        if self.args.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
            return x

    def get_similarity_map(self, proto, query, metric='cosine'):
        r"""
        Args:
            proto (4D Tensor): (way c h1 w1)
            query (4D Tensor): (n c h2 w2)
            metric:

        Returns:
            similarity_map (4D Tensor): (n way h2*w2 h1*w1)
        """
        way = proto.shape[0]
        num_query = query.shape[0]
        query = query.view(query.shape[0], query.shape[1], -1)
        proto = proto.view(proto.shape[0], proto.shape[1], -1)

        proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
        query = query.unsqueeze(1).repeat([1, way, 1, 1])
        proto = proto.permute(0, 1, 3, 2)
        query = query.permute(0, 1, 3, 2)
        feature_size = proto.shape[-2]

        similarity_map = None
        if metric == 'cosine':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = F.cosine_similarity(proto, query, dim=-1)
        if metric == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map
        return similarity_map

    def encode(self, x, dense=True):

        if x.shape.__len__() == 5:  # batch of image patches
            num_data, num_patch = x.shape[:2]
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            x = self.encoder(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.reshape(num_data, num_patch, x.shape[1], x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1, 3, 4)
            x = x.squeeze(-1)
            return x

        else:
            x = self.encoder(x)
            if dense == False:
                x = F.adaptive_avg_pool2d(x, 1)
                return x
            if self.args.feature_pyramid is not None:
                x = self.build_feature_pyramid(x)
        return x

    def build_feature_pyramid(self, feature):
        feature_list = []
        for size in self.args.feature_pyramid:
            feature_list.append(F.adaptive_avg_pool2d(feature, size).view(feature.shape[0], feature.shape[1], 1, -1))
        feature_list.append(feature.view(feature.shape[0], feature.shape[1], 1, -1))
        out = torch.cat(feature_list, dim=-1)
        return out

def emd_inference_opencv_test(distance_matrix, weight1, weight2):
    distance_list = []
    flow_list = []

    for i in range(distance_matrix.shape[0]):
        cost, flow = opencv_emd(distance_matrix[i], weight1[i], weight2[i])
        distance_list.append(cost)
        flow_list.append(torch.from_numpy(flow))

    emd_distance = torch.Tensor(distance_list).cuda().double()
    flow = torch.stack(flow_list, dim=0).cuda().double()

    return emd_distance, flow

def qpth_emd(dist_mat, q_w, p_w, form='QP', l2_strength=0.0001):
    r""" To use the QP solver QPTH to derive EMD (LP problem),
    one can transform the LP problem to QP,
    or omit the QP term by multiplying it with a small value,i.e. l2_strength.

    Args:
        dist_mat (3D Tensor): (N*way h2*w2 h1*w1)
        q_w (2D Tensor): (N*way h2*w2)
        p_w (2D Tensor): (N*way h1*w1)
        form:
        l2_strength:

    Returns:
        emd_score: (N*way )
        flow: (N*way h2*w2 h1*w1)
    """
    q_w = (q_w * q_w.shape[-1]) / q_w.sum(1).unsqueeze(1)
    p_w = (p_w * p_w.shape[-1]) / p_w.sum(1).unsqueeze(1)

    nbatch = dist_mat.shape[0]
    nelement_distmatrix = dist_mat.shape[1] * dist_mat.shape[2]
    nelement_weight1 = q_w.shape[1]
    nelement_weight2 = p_w.shape[1]

    Q_1 = dist_mat.view(-1, 1, nelement_distmatrix).double()

    if form == 'QP':
        # version: QTQ
        Q = torch.bmm(Q_1.transpose(2, 1), Q_1).double().cuda() + 1e-4 * torch.eye(
            nelement_distmatrix).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)  # 0.00001 *
        p = torch.zeros(nbatch, nelement_distmatrix).double().cuda()
    elif form == 'L2':
        # version: regularizer
        Q = (l2_strength * torch.eye(nelement_distmatrix).double()).cuda().unsqueeze(0).repeat(nbatch, 1, 1)
        p = dist_mat.view(nbatch, nelement_distmatrix).double()
    else:
        raise ValueError('Unkown form')

    h_1 = torch.zeros(nbatch, nelement_distmatrix).double().cuda()
    h_2 = torch.cat([q_w, p_w], 1).double()
    h = torch.cat((h_1, h_2), 1)

    G_1 = -torch.eye(nelement_distmatrix).double().cuda().unsqueeze(0).repeat(nbatch, 1, 1)
    G_2 = torch.zeros([nbatch, nelement_weight1 + nelement_weight2, nelement_distmatrix]).double().cuda()
    # sum_j(xij) = si
    for i in range(nelement_weight1):
        G_2[:, i, nelement_weight2 * i:nelement_weight2 * (i + 1)] = 1
    # sum_i(xij) = dj
    for j in range(nelement_weight2):
        G_2[:, nelement_weight1 + j, j::nelement_weight2] = 1
    # xij>=0, sum_j(xij) <= si,sum_i(xij) <= dj, sum_ij(x_ij) = min(sum(si), sum(dj))
    G = torch.cat((G_1, G_2), 1)
    A = torch.ones(nbatch, 1, nelement_distmatrix).double().cuda()
    b = torch.min(torch.sum(q_w, 1), torch.sum(p_w, 1)).unsqueeze(1).double()
    flow = QPFunction(verbose=-1)(Q, p, G, h, A, b)

    emd_score = torch.sum((1 - Q_1).squeeze() * flow, 1)
    return emd_score, flow.view(-1, nelement_weight1, nelement_weight2)

def opencv_emd(cost_matrix, q_w, p_w):
    """
    Args:
        cost_matrix (2D Tensor): (h2*w2 h1*w1)
        q_w (1D Tensor): (h2*w2)
        p_w (1D Tensor): (h1*w1)

    Returns:
        cost: (h2*w2 h1*w1) ?
        flow: (h2*w2 h1*w1)
    """
    cost_matrix = cost_matrix.detach().cpu().numpy()

    q_w = F.relu(q_w) + 1e-5
    p_w = F.relu(p_w) + 1e-5

    q_w = (q_w * (q_w.shape[0] / q_w.sum().item())).view(-1, 1).detach().cpu().numpy()
    p_w = (p_w * (p_w.shape[0] / p_w.sum().item())).view(-1, 1).detach().cpu().numpy()

    cost, _, flow = cv2.EMD(q_w, p_w, cv2.DIST_USER, cost_matrix)
    return cost, flow

def earth_mover_dist(query, proto, solver):
    q_w = get_query_weight(query, proto)  # (N way h2*w2)
    p_w = get_query_weight(proto, query)  # (way N h1*w1)

    query = center_normalize(query, norm='center')
    proto = center_normalize(proto, norm='center')

    similarity_map = get_similarity_map(proto, query)  # (N way h2*w2 h1*w1)
    if solver == 'opencv':  # inference
        dist = get_emd(similarity_map, q_w, p_w, solver='opencv')
    else:  # training
        dist = get_emd(similarity_map, q_w, p_w, solver='qpth')
    return dist

def get_query_weight(q, anchor):
    """
    Args:
        q (4D Tensor): (M c h1 w1)
        anchor (4D Tensor): (N c h2 w2)

    Returns:
        combination (3D Tensor): (M N h1*w1)
    """
    M = q.shape[0]
    N = anchor.shape[0]

    anchor = F.adaptive_avg_pool2d(anchor, [1, 1])
    anchor = anchor.repeat(1, 1, q.shape[2], q.shape[3])

    q = q.unsqueeze(1)
    anchor = anchor.unsqueeze(0)

    q = q.repeat(1, N, 1, 1, 1)
    anchor = anchor.repeat(M, 1, 1, 1, 1)

    combination = (q * anchor).sum(2)
    combination = combination.view(M, N, -1)
    combination = F.relu(combination) + 1e-3
    return combination

def get_similarity_map(proto, query, metric='cosine'):
    r"""
    Args:
        proto (4D Tensor): (way c h1 w1)
        query (4D Tensor): (n c h2 w2)
        metric:

    Returns:
        similarity_map (4D Tensor): (n way h2*w2 h1*w1)
    """
    way = proto.shape[0]
    num_query = query.shape[0]
    query = query.view(query.shape[0], query.shape[1], -1)
    proto = proto.view(proto.shape[0], proto.shape[1], -1)

    proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
    query = query.unsqueeze(1).repeat([1, way, 1, 1])
    proto = proto.permute(0, 1, 3, 2)
    query = query.permute(0, 1, 3, 2)
    feature_size = proto.shape[-2]

    similarity_map = None
    if metric == 'cosine':
        proto = proto.unsqueeze(-3)
        query = query.unsqueeze(-2)
        query = query.repeat(1, 1, 1, feature_size, 1)
        similarity_map = F.cosine_similarity(proto, query, dim=-1)
    if metric == 'l2':
        proto = proto.unsqueeze(-3)
        query = query.unsqueeze(-2)
        query = query.repeat(1, 1, 1, feature_size, 1)
        similarity_map = (proto - query).pow(2).sum(-1)
        similarity_map = 1 - similarity_map
    return similarity_map

def get_emd(sim_map,
            query_weight,
            proto_weight,
            solver='opencv',
            temperature=12.5,
            form='L2',
            l2_strength=1e-6):
    r"""
    Args:
        sim_map: (N way h2*w2 h1*w1)
        query_weight: (N way h2*w2)
        proto_weight: (way N h1*w1)
        solver: in [opencv qpth]
        form: in [QP L2]

    Returns:
        dist: (N way)
    """
    num_query = sim_map.shape[0]
    num_proto = sim_map.shape[1]
    num_node = query_weight.shape[-1]

    if solver == 'opencv':  # use openCV solver
        for i in range(num_query):
            for j in range(num_proto):
                _, flow = opencv_emd(1 - sim_map[i, j, :, :], query_weight[i, j, :], proto_weight[j, i, :])
                # flow is of shape (h2*w2 h1*w1)
                sim_map[i, j, :, :] = (sim_map[i, j, :, :]) * torch.from_numpy(flow).cuda()

        temperature = (temperature / num_node)
        dist = sim_map.sum(-1).sum(-1) * temperature
        return dist

    elif solver == 'qpth':
        proto_weight = proto_weight.permute(1, 0, 2)
        sim_map = sim_map.view(num_query * num_proto, sim_map.shape[-2], sim_map.shape[-1])
        query_weight = query_weight.view(num_query * num_proto, query_weight.shape[-1])
        proto_weight = proto_weight.reshape(num_query * num_proto, proto_weight.shape[-1])

        _, flows = qpth_emd(1 - sim_map, query_weight, proto_weight, form=form, l2_strength=l2_strength)

        dist = (flows * sim_map).view(num_query, num_proto, flows.shape[-2], flows.shape[-1])
        temperature = (temperature / num_node)
        dist = dist.sum(-1).sum(-1) * temperature
        return dist

    else:
        raise ValueError('Unknown Solver')

def center_normalize(x, norm='center'):
    if norm == 'center':
        x = x - x.mean(1, keepdim=True)
        return x
    else:
        return x

if __name__ == '__main__':
    query = torch.rand(5, 10, 3, 2).cuda()
    proto = torch.rand(1, 10, 3, 3).cuda()
    dist = earth_mover_dist(query, proto, solver='opencv')
    print(dist.shape)
