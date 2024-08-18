import torch


def pairwise_euclidean_distance(a, b):
    return torch.norm(a.unsqueeze(1) - b.unsqueeze(0), dim=-1)
