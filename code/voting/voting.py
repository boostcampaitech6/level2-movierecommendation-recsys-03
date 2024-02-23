import os
from args import parse_args
import argparse

import pandas as pd
import torch

def main(args):
    
    outputs = list(map(lambda x: x.strip(), args.outputs.split(',')))
    df = pd.DataFrame(columns = ['user','item'])
    for output in outputs:
        new_df = pd.read_csv(os.path.join(output),index_col=0,header=0)
        df = pd.concat([df, new_df], ignore_index=True)

    count_df = df.groupby('user')['item'].value_counts().reset_index(name='count')
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