import pandas as pd
import numpy as np
import os
from itertools import combinations
from multiprocessing import Pool

def calculate_correlation(pair):
        idx1, idx2 = pair
        correlation = np.corrcoef(user_data[:, idx1], user_data[:, idx2])[0, 1]
        return correlation
    
def calculation_per_user(user1):
    idx1 = user_pairs.index(user1)
    pairs = [(idx1, user_pairs.index(user2)) for user2 in inactive_user_ids]
    with Pool() as pool:
        correlations = pool.map(calculate_correlation, pairs)
    #calculate mean taste similarity
    avg = np.mean(correlations)
    #calculate taste dispersion
    std = np.std(correlations)
    user_correlation[idx1] = avg 
    taste_dispersion[idx1] = std


def mean_taste_sim(train_df):
    for user1 in user_pairs:
        calculation_per_user(user1)

    pearson_df = pd.DataFrame({'user':user_pairs, 'similarity':user_correlation, 'dispersion':taste_dispersion})
    new_df = pd.merge(train_df, pearson_df, on=['user'], how='left')
    return new_df

#get data
data_path = '/data/ephemeral/movie/data/train'
train_df = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
#except 3000 active user
inactive_user_ids = train_df.value_counts('user').index[3000:]
#user-item interaction table
table = pd.pivot_table(train_df, index=['item'], columns=['user'], aggfunc=lambda x: 1, fill_value=0)
#prepare data
user_pairs = train_df['user'].unique().tolist()
user_num = len(user_pairs)
user_correlation = np.zeros((user_num))
taste_dispersion = np.zeros((user_num))
user_data = table.to_numpy()
#save new df
new_df = mean_taste_sim(train_df)
new_df.to_csv(os.path.join(data_path, 'train_pearson_inactive.csv'), index=False)


    
