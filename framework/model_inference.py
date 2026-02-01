"""
Minimal LightGCN Model Definition for Inference Only
This file contains only the model class definition without training code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import Tuple

class LightGCN(nn.Module):
    """A self-contained LightGCN implementation.

    Features:
    - Builds normalized adjacency from a scipy COO interaction matrix (user-item bipartite)
    - Layer-wise propagation (no non-linearities / no feature transform)
    - Mean aggregation over (L+1) layers (including the 0-th embedding)
    """
    def __init__(self, args):
        super().__init__()
        self.num_users = args['user_num']
        self.num_items = args['item_num']
        self.embedding_dim = args.get('embedding_dim', 64)
        self.num_layers = args.get('num_layers', 3)
        self.interaction_matrix = args.get('interaction_matrix', None)
        self.device = torch.device(args.get('device', 'cpu'))
        
        # storage variables for rank evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # Embeddings
        self.embed_user = nn.Embedding(self.num_users, self.embedding_dim)
        self.embed_item = nn.Embedding(self.num_items, self.embedding_dim)
        self.apply(self._init_weights)

        # Only build adjacency matrix if provided (not needed for inference from checkpoint)
        if self.interaction_matrix is not None:
            if not sp.issparse(self.interaction_matrix):
                raise TypeError("interaction_matrix must be a scipy sparse matrix")
            self.register_buffer('norm_adj_matrix', self._build_norm_adj(self.interaction_matrix).coalesce())
        else:
            # For inference only, we don't need the adjacency matrix
            # since embeddings are already learned
            self.norm_adj_matrix = None

    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight)

    def _build_norm_adj(self, inter_M: sp.coo_matrix) -> torch.sparse.FloatTensor:
        """Build symmetric normalized adjacency A_hat for user-item bipartite graph."""
        inter_M = inter_M.tocoo()
        A = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        # user->item (offset items by num_users)
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.num_users), [1]*inter_M.nnz))
        # item->user
        data_dict.update(dict(zip(zip(inter_M.col + self.num_users, inter_M.row), [1]*inter_M.nnz)))
        A._update(data_dict)

        sum_arr = (A > 0).sum(axis=1)
        deg = np.array(sum_arr.flatten())[0] + 1e-7
        deg_inv_sqrt = np.power(deg, -0.5)
        D = sp.diags(deg_inv_sqrt)
        L = D * A * D  # symmetric norm
        L = sp.coo_matrix(L)
        indices = torch.LongTensor(np.vstack([L.row, L.col]))
        values = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(indices, values, torch.Size(L.shape))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation"""
        if self.norm_adj_matrix is None:
            # For inference only, just return the embeddings
            return self.embed_user.weight, self.embed_item.weight
            
        all_embeddings = torch.cat([self.embed_user.weight, self.embed_item.weight], dim=0)
        embeddings_list = [all_embeddings]
        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        # Mean over layers
        final = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)
        user_final, item_final = torch.split(final, [self.num_users, self.num_items])
        return user_final, item_final
