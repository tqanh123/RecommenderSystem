import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils

import pandas as pd
import numpy as np
import scipy.sparse as sp
import random 
from tqdm import tqdm
from collections import Counter
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# ============================================= function split data =============================================
class TestSplitter(object):
    def __init__(self, args):
        self.test_size = args['test_size']
        self.uid = 'user_id'
        self.tid = 'item_id'

    def split(self, df):
        train_index, test_index = split_test(df, self.test_size, self.uid, self.tid)

        return train_index, test_index


class ValidationSplitter(object):
    def __init__(self, args):
        # self.fold_num = args.fold_num
        self.val_size = args['val_size']
        self.uid = 'user_id'
        self.tid = 'item_id'

    def split(self, df):
        train_val_index_zip = split_validation(df, self.val_size, self.uid, self.tid)

        return train_val_index_zip


def split_test(df, test_size=0.1, uid='user', tid='timestamp'):

    test_ids = df.groupby(uid).apply(
        lambda x: x.sample(frac=test_size).index
    ).explode().values
    # test_ids = np.array([int(x) for x in test_ids if not pd.isna(x)])
    test_ids = np.array(list(test_ids))
    train_ids = np.setdiff1d(df.index.values, test_ids)

    return train_ids, test_ids


def split_validation(train_set, val_size=.1, uid='user', tid='timestamp'):

    train_set = train_set.reset_index(drop=True)

    # train_set_list, val_set_list = [], []
    # for _ in range(fold_num):
    val_ids = train_set.groupby(uid).apply(
        lambda x: x.sample(frac=val_size).index
    ).explode().values
    # val_ids = np.array([int(x) for x in val_ids if not pd.isna(x)])
    val_ids = np.array(list(val_ids))
    train_ids = np.setdiff1d(train_set.index.values, val_ids)

    # train_set     _list.append(train_ids)
    # val_set_list.append(val_ids)

    return train_ids, val_ids 

# ============================================= function metrics =============================================

class Metric(object):
    def __init__(self, config) -> None:
        self.metrics = config['metrics']
        self.item_num = config['item_num']
        self.item_pop = config['item_pop'] if 'coverage' in self.metrics else None
        self.i_categories = config['i_categories'] if 'diversity' in self.metrics else None

    def run(self, test_ur, pred_ur, test_u):
        res = []
        for mc in self.metrics:
            if mc == 'ndcg':
                kpi = NDCG(test_ur, pred_ur, test_u)
            elif mc == 'recall':
                kpi = Recall(test_ur, pred_ur, test_u)
            elif mc == 'precision':
                kpi = Precision(test_ur, pred_ur, test_u)
            else:
                raise ValueError(f'Invalid metric name {mc}')

            res.append(kpi)

        return res

def Precision(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        pre = np.in1d(pred, list(gt)).sum() / len(pred)

        res.append(pre)

    return np.mean(res)


def Recall(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        rec = np.in1d(pred, list(gt)).sum() / len(gt)

        res.append(rec)

    return np.mean(res)

def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 1)+1),
        # np.divide(scores, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)+1),
        dtype=np.float32)

def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)
    idcg = getDCG(relevance)

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg

def NDCG(test_ur, pred_ur, test_u):
    res = []
    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]
        nd = getNDCG(pred, gt)
        res.append(nd)
    return np.mean(res)


def AUC(test_ur, pred_ur, test_u):
    res = []

    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]

        r = np.in1d(pred, list(gt))
        pos_num = r.sum()
        neg_num = len(pred) - pos_num

        # Handle edge cases: if no positives or no negatives, AUC is undefined
        if pos_num == 0 or neg_num == 0:
            continue  # Skip this user instead of adding NaN

        pos_rank_num = 0
        for j in range(len(r) - 1):
            if r[j]:
                pos_rank_num += np.sum(~r[j + 1:])

        auc = pos_rank_num / (pos_num * neg_num)
        res.append(auc)

    # Return 0.0 if no valid users, otherwise return mean
    return np.mean(res) if len(res) > 0 else 0.0


def F1(test_ur, pred_ur, test_u):
    res = []

    for idx in range(len(test_u)):
        u = test_u[idx]
        gt = test_ur[u]
        pred = pred_ur[idx]

        r = np.in1d(pred, list(gt))
        pre = r.sum() / len(pred)
        rec = r.sum() / len(gt)

        # Avoid division by zero when computing F1
        if pre + rec > 0:
            f1 = 2 * pre * rec / (pre + rec)
        else:
            f1 = 0.0
        res.append(f1)

    return np.mean(res)

# ============================================= function get data =============================================

def get_ur(df):
    print("Method of getting user-rating pairs")
    ur = df.groupby('user_id').item_id.apply(list).to_dict()
    # print(ur)
    return ur

class BasicDataset(data_utils.Dataset):
    def __init__(self, samples):
  
        super(BasicDataset, self).__init__()
        self.data = samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1], self.data[index][2]

class Cf_valDataset(data_utils.Dataset):
    def __init__(self, data):
        super(Cf_valDataset, self).__init__()
        self.user = data
        # self.data = data

    def __len__(self):
        return len(self.user)

    def __getitem__(self, index):
        user = self.user[index]
        return torch.tensor(user)#, torch.tensor(self.data[user])

def get_train_loader(dataset, args):
    dataloader = data_utils.DataLoader(dataset, batch_size=args['train_batch_size'], shuffle=True, pin_memory=True)
    return dataloader

def get_val_loader(dataset, args):
    dataloader = data_utils.DataLoader(dataset, batch_size=args['val_batch_size'], shuffle=False, pin_memory=True)
    return dataloader

def get_test_loader(dataset, args):
    dataloader = data_utils.DataLoader(dataset, batch_size=args['test_batch_size'], shuffle=False, pin_memory=True)
    return dataloader

def get_inter_matrix(df, args):
    '''
    get the whole sparse interaction matrix
    '''
    print("get the whole sparse interaction matrix")
    user_num, item_num = args['user_num'], args['item_num']

    src, tar = df['user_id'].values, df['item_id'].values
    data = df['click'].values

    mat = sp.coo_matrix((data, (src, tar)), shape=(user_num, item_num))

    return mat

def build_relation_matrices_from_df(df: pd.DataFrame, relations: List[str], user_num: int, item_num: int) -> Dict[str, sp.coo_matrix]:
    """
    df: should have columns ['user_id', 'item_id', <relations...>] with binary flags 0/1
    relations: e.g. ['click','like','share','follow','exposed']
    Returns dict: relation -> scipy.sparse.coo_matrix (user_num x item_num)
    """
    rel_mats = {}
    for r in relations:
        sub = df[df[r] == 1]
        if sub.shape[0] == 0:
            rel_mats[r] = sp.coo_matrix((user_num, item_num))
            continue
        rows = sub['user_id'].values.astype(np.int32)
        cols = sub['item_id'].values.astype(np.int32)
        data = np.ones(len(rows), dtype=np.float32)
        mat = sp.coo_matrix((data, (rows, cols)), shape=(user_num, item_num))
        rel_mats[r] = mat
    return rel_mats

# ============================================= function neg sampler =============================================

class PopularNegativeSampler:
    def __init__(self, train_data, val_data, test_data, user_count, item_count, sample_size, seed):
        self.train = train_data
        self.val = val_data
        self.test = test_data  
        self.user_count = user_count
        self.item_count = item_count
        self.sample_size = sample_size 
        self.seed = seed
           
    def generate_negative_samples(self):
        # Tính popularity distribution của items
        popular_items_freq = self.items_by_popularity()
        print("popular_items_freq: ", popular_items_freq)
        negative_samples = {}
        
        print('Popular Sampling negative items...')
        for user in range(1, self.user_count + 1):
            seen = set(self.train[user])
            print("set: ", seen)
            
            # Loại bỏ items đã tương tác khỏi popularity distribution
            temp = popular_items_freq.copy()
            for item in seen:
                temp.pop(item, None)
            
            # Weighted sampling dựa trên popularity
            samples = random.choices(
                list(temp.keys()), 
                weights=list(temp.values()), 
                k=self.sample_size
            )
            negative_samples[user] = samples
            
        return negative_samples
    
    def items_by_popularity(self):
        # Đếm frequency của mỗi item
        popularity = Counter()
        print("user count: ", self.user_count)
        for user in range(1, self.user_count + 1):
            popularity.update(self.train[user])
            print("after train: ", popularity)
            popularity.update(self.val[user]) 
            print("after val: ", popularity)
            popularity.update(self.test[user])
            print("after test: ", popularity)

        # Convert to frequency distribution
        popular_items = dict(popularity.most_common(self.item_count))
        word_counts = np.array([count for count in popular_items.values()], dtype=np.float32)
        
        # Áp dụng công thức smoothing freq^(3/4)
        word_freqs = word_counts / np.sum(word_counts)
        word_freqs = word_freqs ** (3.0 / 4.0)  # Smoothing
        word_freqs = word_freqs / np.sum(word_freqs)  # Re-normalize
        
        # Cập nhật lại probability cho mỗi item
        i = 0
        for key in popular_items.keys(): 
            popular_items[key] = word_freqs[i]
            i += 1
            
        return popular_items

class BasicNegativeSampler:
    def __init__(self, df, args):
        """
        negative sampling class for <u, pos_i, neg_i> or <u, pos_i, r>
        Parameters
        ----------
        df : pd.DataFrame, the raw <u, i, r> dataframe
        user_num: int, the number of users
        item_num: int, the number of items
        num_ng : int, No. of nagative sampling per sample, default is 4
        sample_method : str, sampling method, default is 'uniform',
                        'uniform' discrete uniform sampling
                        'high-pop' sample items with high popularity as priority
                        'low-pop' sample items with low popularity as prority
        sample_ratio : float, scope [0, 1], it determines the ratio that the other sample method except 'uniform' occupied, default is 0
        """
        self.uid_name = 'user_id'
        self.iid_name = 'item_id'
        self.item_num = args['item_num']
        self.user_num = args['user_num']
        self.ur = args['train_ur']
        self.num_ng = args['num_ng']
        self.inter_name = 'click'
        self.sample_method = args['sample_method']
        self.sample_ratio = args['sample_ratio']

        assert self.sample_method in ['uniform', 'low-pop',
                                      'high-pop'], f'Invalid sampling method: {self.sample_method}'
        assert 0 <= self.sample_ratio <= 1, 'Invalid sample ratio value'

        self.df = df
        self.pop_prob = None

        if self.sample_method in ['high-pop', 'low-pop']:
            pop = df.groupby(self.iid_name).size()
            # rescale to [0, 1]
            pop /= pop.sum()
            pop = pop ** (3. / 4.)
            if self.sample_method == 'high-pop':
                norm_pop = np.zeros(self.item_num)
                norm_pop[pop.index] = pop.values
            if self.sample_method == 'low-pop':
                norm_pop = np.ones(self.item_num)
                norm_pop[pop.index] = (1 - pop.values)
            self.pop_prob = norm_pop / norm_pop.sum()

    def sampling(self):
        if self.num_ng == 0:
            raise NotImplementedError('loss function (BPR) need num_ng > 0')

        js = np.zeros((self.user_num, self.num_ng), dtype=np.int32)
        if self.sample_method in ['low-pop', 'high-pop']:
            other_num = int(self.sample_ratio * self.num_ng)
            uniform_num = self.num_ng - other_num

            for u in tqdm(range(self.user_num)):
                past_inter = list(self.ur[u])

                uni_negs = []
                for _ in range(uniform_num):
                    item = np.random.choice(self.item_num)
                    while item in past_inter or item in uni_negs:
                        item = np.random.choice(self.item_num)
                    uni_negs.append(item)
                uni_negs = np.array(uni_negs)
                # uni_negs = np.random.choice(
                #     np.setdiff1d(np.arange(self.item_num), past_inter),
                #     size=uniform_num
                # )
                other_negs = []
                for _ in range(other_num):
                    item = np.random.choice(self.item_num, p=self.pop_prob)
                    while item in past_inter or item in uni_negs or item in other_negs:
                        item = np.random.choice(self.item_num, p=self.pop_prob)
                    other_negs.append(item)
                other_negs = np.array(other_negs)
                # other_negs = np.random.choice(
                #     np.arange(self.item_num),
                #     size=other_num,
                #     p=self.pop_prob
                # )
                js[u] = np.concatenate((uni_negs, other_negs), axis=None)

        else:
            # all negative samples are sampled by uniform distribution
            for u in tqdm(range(self.user_num)):
                past_inter = list(self.ur[u])
                neg = []
                for _ in range(self.num_ng):
                    item = np.random.choice(self.item_num)
                    while item in past_inter or item in neg:
                        item = np.random.choice(self.item_num)
                    neg.append(item)
                js[u] = np.array(neg)
                # js[u] = np.random.choice(
                #     np.setdiff1d(np.arange(self.item_num), past_inter),
                #     size=self.num_ng
                # )


        self.df['neg_set'] = self.df[self.uid_name].agg(lambda u: js[u])        
        self.df = self.df[[self.uid_name, self.iid_name, 'neg_set']].explode('neg_set')
        return self.df.values.astype(np.int32)

# ============================================= Model =============================================


class LightGCN(nn.Module):
    """A self-contained LightGCN implementation.

    Features:
    - Builds normalized adjacency from a scipy COO interaction matrix (user-item bipartite)
    - Layer-wise propagation (no non-linearities / no feature transform)
    - Mean aggregation over (L+1) layers (including the 0-th embedding)
    - Supports BPR, hinge (HL), TOP1 (TL), and point-wise (BCEWithLogits, MSE) losses via configure_loss
    - Ranking utilities: rank(), full_rank()
    """
    def __init__(self, args):
        super().__init__()
        self.num_users = args['user_num']
        self.num_items = args['item_num']
        self.embedding_dim = args.get('embedding_dim', 64)
        self.num_layers = args.get('num_layers', 3)
        self.interaction_matrix = args.get('interaction_matrix', None)
        self.device = torch.device(args.get('device', 'cpu'))
        self.reg_1 = args.get('reg_1', 0.0)
        self.reg_2 = args.get('reg_2', 0.0)
        self.lr = args.get('lr', 0.001)
        self.topk = args.get('k', 20)
        self.val_ur = args.get('val_ur', None)
        self.val_u = args.get('val_u', None)
        self.early_stop = args.get('early_stop', True)
        self.save_path = args.get('save_path', './')
                #  device: str = 'cuda',
                #  reg_1: float = 0.0,
                #  reg_2: float = 0.0,
                #  lr: float = 0.001):
        # self.num_users = num_users
        # self.num_items = num_items
        # self.embedding_dim = embedding_dim
        # self.num_layers = num_layers
        # self.device = torch.device(device)
        # self.reg_1 = reg_1
        # self.reg_2 = reg_2
        # self.lr = lr

        # storage variables for rank evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # Embeddings
        self.embed_user = nn.Embedding(self.num_users, self.embedding_dim)
        self.embed_item = nn.Embedding(self.num_items, self.embedding_dim)
        self.apply(self._init_weights)

        if self.interaction_matrix is None:
            raise ValueError("interaction_matrix (scipy sparse) is required")
        if not sp.issparse(self.interaction_matrix):
            raise TypeError("interaction_matrix must be a scipy sparse matrix")

        self.register_buffer('norm_adj_matrix', self._build_norm_adj(self.interaction_matrix).coalesce())

    #  Initialization 
    def _init_weights(self, m):
        if isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight)

    #  Adjacency 
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

    #  Forward Propagation 
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        all_embeddings = torch.cat([self.embed_user.weight, self.embed_item.weight], dim=0)
        embeddings_list = [all_embeddings]
        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        # Mean over layers
        final = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)
        user_final, item_final = torch.split(final, [self.num_users, self.num_items])
        return user_final, item_final

    def _bpr_loss(self, pos_scores, neg_scores):
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    #  Training Step 
    # def calc_loss(self, batch):
    #     """batch: (user, pos_item, neg_item) or (user, item, label) depending on loss."""
    #     user, pos_item, neg_item = batch
    #     user = user.to(self.device).long()
    #     pos_item = pos_item.to(self.device).long()
    #     neg_item = neg_item.to(self.device).long()
    #     user_emb, item_emb = self.forward()
    #     u = user_emb[user]
    #     p = item_emb[pos_item]
    #     n = item_emb[neg_item]
    #     pos_scores = (u * p).sum(dim=-1)
    #     neg_scores = (u * n).sum(dim=-1)
    #     loss = self._bpr_loss(pos_scores, neg_scores)
    #     # Regularization (L2)
    #     if self.reg_2 > 0:
    #         loss = loss + self.reg_2 * (u.norm(dim=1).mean() + p.norm(dim=1).mean() + n.norm(dim=1).mean())
    #     return loss
    def calc_loss(self, batch):
        # ensure model is on correct device
        self.to(self.device)
        # clear stored embeddings before computing
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        # prepare batch indices
        user = batch[0].to(self.device).long()
        if user.dim() == 0:
            user = user.unsqueeze(0)
        pos_item = batch[1].to(self.device).long()
        if pos_item.dim() == 0:
            pos_item = pos_item.unsqueeze(0)

        # compute embeddings
        embed_user, embed_item = self.forward()
        embed_user = embed_user.to(self.device)
        embed_item = embed_item.to(self.device)

        # positive predictions
        u_emb = embed_user[user]
        p_emb = embed_item[pos_item]
        pos_pred = (u_emb * p_emb).sum(dim=1)

        # ego embeddings for regularization
        u_ego = self.embed_user(user)
        p_ego = self.embed_item(pos_item)

        # compute loss
        
        neg = batch[2].to(self.device).long()
        neg_emb = embed_item[neg]
        neg_pred = (u_emb * neg_emb).sum(dim=1)
        neg_ego = self.embed_item(neg)
        loss = self._bpr_loss(pos_pred, neg_pred)
        loss += self.reg_1 * (u_ego.norm(p=1) + p_ego.norm(p=1) + neg_ego.norm(p=1))
        loss += self.reg_2 * (u_ego.norm() + p_ego.norm() + neg_ego.norm())

        return loss

    #  Ranking 
    def rank(self, test_loader):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        rec_ids = torch.tensor([], device=self.device)
        self.eval()
        with torch.no_grad():
            for us in test_loader:
                us = us.to(self.device)
                rank_list = self.full_rank(us)

                rec_ids = torch.cat((rec_ids, rank_list), 0)

        return rec_ids.cpu().numpy().astype(np.int)
    
    def full_rank(self, u):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        # ensure CPU indices for CPU embeddings
        if u.device != self.restore_user_e.device:
            u_idx = u.to(self.restore_user_e.device)
        else:
            u_idx = u
        user_emb = self.restore_user_e[u_idx]  # (batch_size, dim)
        items_emb = self.restore_item_e  # (num_items, dim)
        # compute scores and top-k
        scores = torch.matmul(user_emb, items_emb.transpose(1, 0))
        rank = torch.argsort(scores, descending=True)[:, :self.topk]
        # move to evaluation device
        return rank.to(self.device)
    
    def predict(self, u, i):
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()

        u_embedding = self.restore_user_e[u]
        i_embedding = self.restore_item_e[i]
        pred = torch.matmul(u_embedding, i_embedding.t())

        return pred.cpu().item()

    def fit(self, train_loader, val_loader, epochs: int = 10):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        self.to(self.device)

        train_losses = []
        last_loss = 0.0
        val_metrics = {'ndcg': [], 'recall': [], 'precision': [], 'F1': [], 'AUC': []}
        best_weights = None
        best_ndcg = -np.inf

        for epoch in range(1, epochs+1):
            # print(f"Embedding pre update Epoch {epoch}:")
            # print_layer_embeddings(self)
            self.train()
            current_loss = 0.0
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for batch in pbar:
                opt.zero_grad()
                loss = self.calc_loss(batch)
                if torch.isnan(loss):
                    raise ValueError("NaN loss encountered")
                loss.backward()
                opt.step()
                current_loss += loss.item()

            # print(f"Embedding sau update Epoch {epoch}:")
            # print_layer_embeddings(self)
                

            epoch_loss = current_loss / len(train_loader)
            pbar.set_postfix(loss=epoch_loss)
            train_losses.append(epoch_loss)

            preds = self.rank(val_loader)
            # print("preds: ", preds[10])
            # print("val_u: ", self.val_u[10])
            # print("val_ur: ", self.val_ur[10])
            ndcg = NDCG(self.val_ur, preds, self.val_u)
            recall = Recall(self.val_ur, preds, self.val_u)
            precision = Precision(self.val_ur, preds, self.val_u)
            f1_score = F1(self.val_ur, preds, self.val_u)
            auc_score = AUC(self.val_ur, preds, self.val_u)
            val_metrics['ndcg'].append(ndcg)
            val_metrics['recall'].append(recall)
            val_metrics['precision'].append(precision)
            val_metrics['F1'].append(f1_score)
            val_metrics['AUC'].append(auc_score)

            
            print(f"Training - Loss {epoch_loss:.4f} | Validation - NDCG@{self.topk}: {ndcg:.4f}, Recall@{self.topk}: {recall:.4f}, Precision@{self.topk}: {precision:.4f}, F1@{self.topk}: {f1_score:.4f}, AUC@{self.topk}: {auc_score:.4f}")
            
            # Save best model
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                torch.save(self.state_dict(), Path(self.save_path) / 'best_model.pth')
            
            # Early stopping
            if self.early_stop and abs(current_loss - last_loss) < 1e-5:
                print('Satisfy early stop mechanism')
                break
            last_loss = current_loss
        return train_losses, val_metrics

    def recommend(self, user_idx, top_k=10):
        u_emb = self.user_embedding(user_idx)
        # Dot product with all items
        scores = torch.matmul(self.item_embedding.weight, u_emb)
        return torch.topk(scores, top_k)

# ============================================= main =============================================

# CF Task Pipeline
# 1. Define CF parameters
cf_args = {
    'task_name': 'cf',
    'seed': 0,
    'model_name': 'lightgcn',
    'dataset_path': '.\data\QB-video.csv',
    'train_batch_size': 4096,
    'val_batch_size': 32,
    'test_batch_size': 32,
    'epochs': 50,
    'lr': 0.005,
    'block_num': 2,
    'sample_method': 'high-pop',
    'sample_ratio': 0.3,
    'num_ng': 4,
    'item_min': 10,
    'device': 'cuda',
    'test_size': 0.1,
    'val_size': 0.1111,
    'k': 20,
    'save_path': './checkpoint/',
    'plot_path': './plots/',
    'num_layers': 3,
    'reg_1': 0.0,
    'reg_2': 0.0,
    'embedding_dim': 128,
    'early_stop': True,
    'metrics': ['ndcg', 'recall', 'precision'],
}

# 2. Load and preprocess data
read_df = pd.read_csv(cf_args['dataset_path'], usecols=['user_id', 'item_id', 'click', 'like', 'share']) 
read_df = read_df[read_df.click.isin([1]) | read_df.like.isin([1]) | read_df.share.isin([1])]
user_counts = read_df.groupby('user_id').size()
user_subset = np.in1d(read_df.user_id, user_counts[user_counts >= cf_args['item_min']].index)
df = read_df[user_subset].reset_index(drop=True)
del read_df
assert (df.groupby('user_id').size() < cf_args['item_min']).sum() == 0
cf_args['user_num'] = len(set(df['user_id']))
cf_args['item_num'] = len(set(df['item_id']))
# reset_ob = reset_df()
df['user_id'] = pd.Categorical(df['user_id']).codes
df['item_id'] = pd.Categorical(df['item_id']).codes



# 3. Split train/val/test
test_splitter = TestSplitter(cf_args)
train_idx, test_idx = test_splitter.split(df)
print("df.index.dtype:", df.index.dtype, "len:", len(df))
print("example df.index[:10]:", df.index[:10])
print("train_idx sample:", train_idx[:10])
print("test_idx sample:", test_idx[:10] if test_idx.size else '[] (empty)')

# Ensure non-empty test by global fallback when needed
if test_idx.size == 0:
    rng = np.random.default_rng()
    gk = max(1, int(round(cf_args['test_size'] * len(df))))
    gk = min(gk, max(1, len(df) - 1))
    test_idx = rng.choice(df.index.values.astype(np.int64, copy=False), size=gk, replace=False)
    train_idx = np.setdiff1d(df.index.values.astype(np.int64, copy=False), test_idx)
    print(f"test_idx was empty; used global fallback of size {gk}")

if test_idx.size > 0:
    print("max test_idx:", np.max(test_idx), "min:", np.min(test_idx))

print("split train and test")    
train_set, test = df.iloc[train_idx, :].copy(), df.iloc[test_idx, :].copy()
v_splitter = ValidationSplitter(cf_args)
train_idx2, val_idx = v_splitter.split(train_set)
print("split train and val")
train, validation = train_set.iloc[train_idx2, :].copy(), train_set.iloc[val_idx, :].copy()


# 4. Build user-rating dicts
cf_args['train_ur'] = get_ur(train)
cf_args['val_ur']   = get_ur(validation)
cf_args['test_ur']  = get_ur(test)
cf_args['val_u']    = sorted(cf_args['val_ur'].keys())
cf_args['test_u']   = sorted(cf_args['test_ur'].keys())
cf_args['interaction_matrix'] = get_inter_matrix(train, cf_args)
# cf_args['rel_matrix'] = build_relation_matrices_from_df(train, ['click', 'like', 'share', 'follow', 'watching_times'], cf_args['user_num'], cf_args['item_num'])

# 5. Create dataloaders
sampler = BasicNegativeSampler(train, cf_args)
samples = sampler.sampling()
train_ds = BasicDataset(samples)

train_loader = get_train_loader(train_ds, cf_args)
val_loader   = get_val_loader(Cf_valDataset(cf_args['val_u']), cf_args)
test_loader  = get_test_loader(Cf_valDataset(cf_args['test_u']), cf_args)

print("train loader: ", next(iter(train_loader)))
print("val loader: ", next(iter(val_loader)))
print("test loader: ", next(iter(test_loader)))

# 6. Train model
model = LightGCN(cf_args)
train_losses, val_metrics = model.fit(train_loader, val_loader, cf_args['epochs'])

# Load best model for testing
best_weight = torch.load(Path(cf_args['save_path']) / 'best_model.pth')
model.load_state_dict(best_weight)
model = model.to(cf_args['device'])
# 7. Evaluate on test set
preds = model.rank(test_loader)
ndcg_test = NDCG(cf_args['test_ur'], preds, cf_args['test_u'])
recall_test = Recall(cf_args['test_ur'], preds, cf_args['test_u'])
precision_test = Precision(cf_args['test_ur'], preds, cf_args['test_u'])
F1_test = F1(cf_args['test_ur'], preds, cf_args['test_u'])
AUC_test = AUC(cf_args['test_ur'], preds, cf_args['test_u'])
print(f"Test NDCG@{cf_args['k']}: {ndcg_test:.4f}")
print(f"Test Recall@{cf_args['k']}: {recall_test:.4f}")
print(f"Test Precision@{cf_args['k']}: {precision_test:.4f}")
print(f"Test F1@{cf_args['k']}: {F1_test:.4f}")
print(f"Test AUC@{cf_args['k']}: {AUC_test:.4f}")

# Collect test results for summary
test_results = {
    'NDCG': ndcg_test,
    'Recall': recall_test,
    'Precision': precision_test,
    'F1': F1_test,
    'AUC': AUC_test
}

# 8. Plot training history
import matplotlib.pyplot as plt
import os
import seaborn as sns

try:
    os.makedirs(cf_args['save_path'], exist_ok=True)
    
    #  Set style cho đồ thị
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Tạo figure với subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 12))
    fig.suptitle(f'Training History - LightGCN', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_metrics['ndcg'], 'g-', label='Val NDCG')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].set_title('CF Training History')

    axes[1].plot(epochs, val_metrics['ndcg'], 'g-', label='Val NDCG')
    axes[1].plot(epochs, val_metrics['recall'], 'r-', label='Val Recall')
    axes[1].plot(epochs, val_metrics['precision'], 'm-', label='Val Precision')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].set_title('CF Validation Metrics')
    
    axes[2].plot(epochs, val_metrics['F1'], 'c-', label='Val F1')
    axes[2].plot(epochs, val_metrics['AUC'], 'y-', label='Val AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].set_title('CF Validation Metrics (F1 & AUC)')
    plt.tight_layout()
            
    # Save plots
    filename = f"LightGCN.png"
    plt.savefig(os.path.join('./plots/', filename), dpi=300, bbox_inches='tight')
    print(f"📊 Đồ thị training history đã được lưu tại: ./plots/{filename}")

    plt.close('all')

    # Save summary
    with open(f'./plots/LightGCN_training_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"=== TRAINING SUMMARY - LIGHTGCN ===\n\n")
        f.write("📋 CONFIGURATION:\n")
        f.write(f"Model: LightGCN\n")
        f.write(f"Dataset: {cf_args['dataset_path']}\n")
        f.write(f"Learning Rate: {cf_args['lr']}\n")
        f.write(f"Batch Size: {cf_args['train_batch_size']}\n")
        f.write(f"Embedding Factors: {cf_args['factor_num']}\n")
        f.write(f"Number of Layers: {cf_args['block_num']}\n")
        f.write(f"Epochs: {cf_args['epochs']}\n")
        f.write("📈 TRAINING PROGRESS:\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.6f}\n")
        f.write(f"Best Training Loss: {min(train_losses):.6f}\n")
        f.write(f"Loss Improvement: {train_losses[0] - train_losses[-1]:.6f}\n\n")
        if val_metrics:
            f.write("✅ VALIDATION RESULTS:\n")
            for metric, values in val_metrics.items():
                if values:
                    f.write(f"Best {metric.upper()}: {max(values):.6f}\n")
                    f.write(f"Final {metric.upper()}: {values[-1]:.6f}\n")
            f.write("\n")
        f.write("🎯 TEST RESULTS:\n")
        for metric, value in test_results.items():
            f.write(f"{metric}: {value:.6f}\n")
    
    print("📝 Training summary đã được lưu")

except Exception as e:
    print("Plotting failed:", e)



