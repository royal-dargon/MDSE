"""
written by Jingzhe Li 2023.04.04
用于计算文本、图像之间相似度的文件
"""
import torch


def l1norm(X, dim=-1, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X



def similarity(source_emb, global_emb):
    """
    :param source_emb:表示的是local特征，大小为(batch_size, length, dim)
    :param global_emb:表示的是global特征， 大小为(batch_size, dim)
    :return:返回一个计算相似度得分的向量，大小为(batch_size, length)
    """
    length = source_emb.size(1)
    global_emb = global_emb.unsqueeze(dim=1)
    global_emb_expand = global_emb.repeat(1, length, 1)
    w12 = torch.sum(source_emb * global_emb_expand, dim=-1)
    w1 = torch.norm(source_emb, 2, dim=-1)
    w2 = torch.norm(global_emb_expand, 2, dim=-1)
    r = w12 / (w1 * w2).clamp(min=1e8)
    return torch.unsqueeze(1 + r, dim=-1), torch.unsqueeze(1 - r, dim=-1)




# test部分，测试该模块的有效性
# a = torch.randn(1, 64, 768)
# b = torch.randn(1, 768)
# res = similarity(a, b)
