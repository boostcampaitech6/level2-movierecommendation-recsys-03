import os
import torch
import numpy as np

import pandas as pd
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk

from args import parse_args

def seq_sort(dataset,topk_iid_list):
    topk_iid_list_1000 = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
    data_path = '/data/data/train'
    interactions = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
    grouped_items = interactions.groupby('user')['item'].apply(list)
    result_tensor = torch.zeros((topk_iid_list_1000.shape[0], 10), dtype=torch.int32, device='cuda:0')
    for user_idx in range(topk_iid_list_1000.shape[0]):
        user_topk_iid = topk_iid_list_1000[user_idx,:]
        user_grouped_items = grouped_items.iloc[user_idx]
        filtered_items = user_topk_iid[~np.isin(user_topk_iid, user_grouped_items)]
        top_10_items = filtered_items[:10].astype(int)
        result_tensor[user_idx] = torch.from_numpy(top_10_items)
    return result_tensor


def main(model_path):
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file = model_path
    )
    
    external_user_ids = dataset.id2token(dataset.uid_field, list(range(dataset.user_num)))[1:]
    uid_series = dataset.token2id(dataset.uid_field, external_user_ids)
    
    if args.sequence: #Recbole의 Sequence model에서는 추가로 과거시청이력 제거해줘야한다.
        topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=3000, device=config["device"])
        external_item_list = seq_sort(dataset,topk_iid_list).flatten() 
    else: #Recbole의 General model은 library에서 과거시청이력을 제거해준다.
        topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config["device"])
        external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu()).flatten()        
    output = pd.read_csv('/data/data/eval/sample_submission.csv')
    output['item'] = [int(item) for item in external_item_list]
    if not os.path.exists('output/'):
        os.makedirs('output/')
    output.to_csv('output/submission.csv')
    
if __name__ == '__main__':
    args = parse_args()
    main(args.model_path)