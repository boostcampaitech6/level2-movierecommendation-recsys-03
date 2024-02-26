
import torch
import numpy as np
from tqdm import tqdm

def get_ndcg(pred_list, true_list):
    idcg = sum((1 / np.log2(rank + 2) for rank in range(1, len(pred_list))))
    dcg = 0
    for rank, pred in enumerate(pred_list):
        if pred in true_list:
            dcg += 1 / np.log2(rank + 2)
    ndcg = dcg / idcg
    return ndcg

# hit == recall == precision
def get_hit(pred_list, true_list):
    hit_list = set(true_list) & set(pred_list)
    hit = len(hit_list) / len(true_list)
    return hit


def evaluate(pred, X, user):
    mat = torch.from_numpy(X.to_numpy)

    NDCG = 0.0 # NDCG@10
    HIT = 0.0 # HIT@10

    recon_mat1 = pred.cpu()
    recon_mat1[mat == 1] = -np.inf
    rec_list1 = recon_mat1.argsort(dim = 1)

    for user, rec1 in tqdm(enumerate(rec_list1)):
        uv = user[user]
        # ranking
        up = rec1[-10:].cpu().numpy().tolist()[::-1]

        NDCG += get_ndcg(pred_list = up, true_list = uv)
        HIT += get_hit(pred_list = up, true_list = uv)

    NDCG /= len(user)
    HIT /= len(user)

    return NDCG, HIT