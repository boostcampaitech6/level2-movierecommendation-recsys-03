import os
from args import parse_args
import argparse

import pandas as pd
import torch

def main(args):
    
    outputs = list(map(lambda x: x.strip(), args.outputs.split(',')))
    weights = list(map(lambda x: float(x.strip()), args.weights.split(',')))
    df = pd.DataFrame(columns = ['user','item'])
    for i, output in enumerate(outputs):
        new_df = pd.read_csv(output,index_col=0,header=0)
        new_df = new_df.groupby('user').head(30).reset_index()
        score_list = list(map(lambda x: x*weights[i]*0.5, range(40, 10, -1)))
        new_df['score'] = score_list * len(new_df['user'].unique())
        df = pd.concat([df, new_df], ignore_index=True)

    sum_df = df.groupby(['user', 'item']).sum().reset_index()
    sum_df = sum_df.drop_duplicates(['user','item'])
    count_df = sum_df.sort_values(by=['user', 'score'], ascending=[True, False])
    count_df = count_df.groupby('user').head(10).reset_index()


    output = pd.read_csv('../movie/data/eval/sample_submission.csv')
    output['item'] = count_df['item']
    if not os.path.exists('output/'):
        os.makedirs('output/')
    output.to_csv('output/submission.csv')
    print("Save submission ... ")
    
if __name__=='__main__':
    args = parse_args()
    main(args)