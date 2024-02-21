import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def main(data_path,output_path,submission_name):
    items = pd.read_csv(os.path.join(data_path, 'EDA_items.csv'), sep=',')
    Users = pd.read_csv(os.path.join(data_path, 'EDA_users.csv'), sep=',')
    
    sub = pd.read_csv(output_path+'submission_'+submission_name+'.csv')
    sub = pd.merge(sub, items, on='item')

    import ast
    
    def convert_str_to_list(str_list): # 문자열을 리스트로 변환하는 함수
        try:
            return ast.literal_eval(str_list)
        except ValueError: 
            return []  
        end

    submission_EDA = pd.DataFrame({'user': Users['user'].copy()})

    year_bin_pivot = sub.pivot_table(index='user', columns='year_bin', aggfunc='size', fill_value=0)
    popularity_bin_pivot = sub.pivot_table(index='user', columns='popularity_bin', aggfunc='size', fill_value=0)

    submission_EDA = pd.merge(submission_EDA, year_bin_pivot, on='user')
    submission_EDA = pd.merge(submission_EDA, popularity_bin_pivot, on='user')

    def calculate_pivot(sub, Users, column_name, submission_EDA):
        sub[column_name] = sub[column_name].apply(convert_str_to_list)
        sub_exploded = sub.explode(column_name)
        counts = sub_exploded.pivot_table(index='user', columns=column_name, aggfunc='size', fill_value=0)
        counts_merge = pd.merge(counts, Users, on='user')
        pivot_1 = counts_merge.apply(lambda row: row[row[column_name+'_1']] if row[column_name+'_1'] in counts_merge.columns else 0, axis=1)
        submission_EDA[column_name+'_1']=pivot_1.values
        pivot_2 = counts_merge.apply(lambda row: row[row[column_name+'_2']] if row[column_name+'_2'] in counts_merge.columns else 0, axis=1)
        submission_EDA[column_name+'_2']=pivot_2.values
        pivot_3 = counts_merge.apply(lambda row: row[row[column_name+'_3']] if row[column_name+'_3'] in counts_merge.columns else 0, axis=1)
        submission_EDA[column_name+'_3']=pivot_3.values
        return submission_EDA

    submission_EDA = calculate_pivot(sub,Users,'genre',submission_EDA)
    submission_EDA = calculate_pivot(sub,Users,'writer',submission_EDA)
    submission_EDA = calculate_pivot(sub,Users,'director',submission_EDA)

    submission_EDA = pd.merge(submission_EDA, Users[['user','type']], on='user')
    submission_EDA['model'] = submission_name
    submission_EDA.to_csv(os.path.join(output_path,'submission_'+submission_name+'_EDA.csv'), sep=',', index=False)


if __name__ == '__main__':
    data_path = '../../data/train'
    output_path = '../recbole/output/'
    submission_name='ease'
    main(data_path,output_path,submission_name)