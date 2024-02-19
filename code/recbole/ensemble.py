import os
from args import parse_args

import pandas as pd
import torch
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_scores

def main(args):
    
    models = list(map(lambda x: x.strip(),args.models.split(',')))
    if not models:
        raise RuntimeError("there's no input models")
    else:
        for model_path in models:
            if not os.path.exists(model_path):
                raise RuntimeError(f"{model_path} not exists")
    
    if args.voting == 'soft':
        weights = list(map(lambda x: float(x.strip()),args.weights.split(',')))
        assert len(models) == len(weights), "num of models and num of weights need to be same"
    
    
        for i, model_path in enumerate(models):
            config, model, dataset, _, _, test_data = load_data_and_model(model_path)
            
            external_user_ids = dataset.id2token(dataset.uid_field, list(range(dataset.user_num)))[1:]
            uid_series = dataset.token2id(dataset.uid_field, external_user_ids)
            
            if i==0:
                scores = weights[i]*full_sort_scores(uid_series, model, test_data, config['device']).to(config['device'])
            else:
                scores += weights[i]*full_sort_scores(uid_series, model, test_data, config['device']).to(config['device'])
    
    elif args.voting == 'hard':
        
        for i, model_path in enumerate(models):
            config, model, dataset, _, _, test_data = load_data_and_model(model_path)
            
            external_user_ids = dataset.id2token(dataset.uid_field, list(range(dataset.user_num)))[1:]
            uid_series = dataset.token2id(dataset.uid_field, external_user_ids)
            
            if i == 0:
                scores = full_sort_scores(uid_series, model, test_data, config['device']).to(config['device'])
            else:
                scores += full_sort_scores(uid_series, model, test_data, config['device']).to(config['device'])
            
    topk_score, topk_iid_list = torch.topk(scores,10)
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu()).flatten()
            
    output = pd.read_csv('/data/data/eval/sample_submission.csv')
    output['item'] = [int(item) for item in external_item_list]
    if not os.path.exists('output/'):
        os.makedirs('output/')
    output.to_csv('output/submission.csv')
    print("Save submission ... ")
    
if __name__=='__main__':
    args = parse_args()
    main(args)