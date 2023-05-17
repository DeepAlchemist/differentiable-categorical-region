from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return F.softmax(y * temperature, dim=-1)

def _gumbel_softmax(logits, temperature=5):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    logits = logits.log() * 20000
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y, device=logits.device).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

def gumbel_softmax(logits, tau=1, hard=False):
    return

if __name__ == '__main__':
    logits = torch.tensor([[0.1, 0.4, 0.3, 0.2]], requires_grad=True).cuda().log() * 20000
    tau = 0.8
    res = gumbel_softmax(logits, tau)

    print(res.requires_grad)
