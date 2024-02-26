import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from train_args import parse_args


############# 중요 #############
# data_path는 사용자의 디렉토리에 맞게 설정해야 합니다.

def preprocess(args):
    df = pd.read_csv(args.data_dir + "train_ratings.csv")

    item_ids = df['item'].unique() # 6807
    user_ids = df['user'].unique() # 31360
    num_item, num_user = len(item_ids), len(user_ids)
    num_batch = num_user // args.batch_size

    # user, item indexing
    item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids) # item re-indexing (1~num_item), num_item+1: mask idx
    user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) # user re-indexing (0~num_user-1)

    # dataframe indexing
    df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
    df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
    df.sort_values(['user_idx', 'time'], inplace=True)
    del df['item'], df['user']

    # train set, valid set 생성
    users = defaultdict(list) # defaultdict은 dictionary의 key가 없을때 default 값을 value로 반환
    user_train = {}
    user_valid = {}
    for u, i, t in zip(df['user_idx'], df['item_idx'], df['time']):
        users[u].append(i)

    for user in users:
        user_train[user] = users[user][:-1]
        user_valid[user] = [users[user][-1]]

    print(f'num users: {num_user}, num items: {num_item}')
    
    return num_user, num_item, df, user_train, user_valid