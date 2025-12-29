import torch
import torch.utils.data as data_utils
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import List, Dict


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
