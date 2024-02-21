r"""
EASE
################################################
Reference:
    Harald Steck. "Embarrassingly Shallow Autoencoders for Sparse Data" in WWW 2019.
"""

import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd

from recbole.utils import InputType, ModelType
from recbole.model.abstract_recommender import GeneralRecommender


class EASE(GeneralRecommender):
    r"""EASE is a linear model for collaborative filtering, which combines the
    strengths of auto-encoders and neighborhood-based approaches.

    """
    input_type = InputType.POINTWISE
    type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # load parameters info
        reg_weight = config["reg_weight"]

        # need at least one param
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

        #change user feature
        #X = dataset.inter_matrix(form="csr", value_field='similarity').astype(np.float32)
        X = dataset.inter_matrix(form="csr").astype(np.float32) 
        

        # gram matrix
        G = X.T @ X

        # add reg to diagonal
        G += reg_weight * sp.identity(G.shape[0]).astype(np.float32)

        # convert to dense because inverse will be dense
        G = G.todense()

        # invert. this takes most of the time
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))  
        # zero out diag
        np.fill_diagonal(B, 0.0)
      

        ####rescale
        #get popularity
        pop = np.array(list(dataset.item_counter.values()))

        #W = 1/pop^a  #a=0.5 (hyper param)
        #B*diag(W)
        W = 1/pop*0.001
        diag_W = sp.spdiags(W, [0], len(W), len(W)).toarray()

        # B와 사이즈 맞추기  # X: (6808, 6808)
        first_row = np.zeros((1, diag_W.shape[1]))
        diag_W = np.concatenate((first_row, diag_W), axis=0)
        last_column = np.zeros((diag_W.shape[0], 1))
        diag_W = np.concatenate((diag_W, last_column), axis=1)

        B = B * diag_W  #remove diagonal

        # instead of computing and storing the entire score matrix,
        # just store B and compute the scores on demand
        # more memory efficient for a larger number of users
        # but if there's a large number of items not much one can do:
        # still have to compute B all at once
        # S = X @ B
        # self.score_matrix = torch.from_numpy(S).to(self.device)

        # torch doesn't support sparse tensor slicing,
        # so will do everything with np/scipy
        self.item_similarity = B
        self.interaction_matrix = X
        self.other_parameter_name = ["interaction_matrix", "item_similarity"]
        self.device = config.device

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()
        item = interaction[self.ITEM_ID].cpu().numpy()

        return torch.from_numpy(
            (self.interaction_matrix[user, :].multiply(self.item_similarity[:, item].T))
            .sum(axis=1)
            .getA1()
        ).to(self.device)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID].cpu().numpy()

        r = self.interaction_matrix[user, :] @ self.item_similarity
        return torch.from_numpy(r.flatten())