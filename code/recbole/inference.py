import os

import pandas as pd
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk

from args import parse_args

def main(model_path):
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file = model_path
    )
    
    external_user_ids = dataset.id2token(dataset.uid_field, list(range(dataset.user_num)))[1:]
    uid_series = dataset.token2id(dataset.uid_field, external_user_ids)
    
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