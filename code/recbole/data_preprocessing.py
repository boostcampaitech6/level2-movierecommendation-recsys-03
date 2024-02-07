import os
import yaml
import pandas as pd

def convert_into_atomic_files(config):
    
    interactions = pd.read_csv(os.path.join(config['data_path'], 'train_ratings.csv'))
    directors = pd.read_csv(os.path.join(config['data_path'], 'directors.tsv'), sep='\t')
    genres = pd.read_csv(os.path.join(config['data_path'], 'genres.tsv'), sep='\t')
    titles = pd.read_csv(os.path.join(config['data_path'], 'titles.tsv'), sep='\t')
    writers = pd.read_csv(os.path.join(config['data_path'], 'writers.tsv'), sep='\t')
    years = pd.read_csv(os.path.join(config['data_path'], 'years.tsv'), sep='\t')
    
    if not os.path.exists('/data/data/train/movie/'):
        os.makedirs('/data/data/train/movie/')
    interactions.rename(columns={'user':'user:token', 'item':'item:token', 'time':'time:float'}, inplace = True)
    interactions.to_csv(os.path.join(config['data_path'], 'movie/movie.inter'), sep='\t', index=False)
    print("Save interaction data ...")
    
    interactions[['user:token']].to_csv(os.path.join(config['data_path'], 'movie/movie.user'), sep='\t', index=False)
    print("Save user data ...")
    
    items = pd.merge(titles, years, on='item',)
    items = pd.merge(items, genres, on='item')
    items = pd.merge(items, writers, on='item', how = 'left')
    items = pd.merge(items, directors, on='item', how = 'left')
    items = items.fillna("none")
    items.rename(columns={'item':'item:token', 'title':'title:token_seq', 'year':'year:token', 'genre':'genre:token_seq', "writer":"writer:token", 'director':'director:token'}, inplace=True)
    items.to_csv(os.path.join(config['data_path'],'movie','movie.item'), sep='\t', index=False)
    print("Save item data ...")
    print("done!")

if __name__ == '__main__':
    # yaml 파일을 읽어서 dictionary로 불러오기
    with open('configs/recvae.yaml', "r") as file:
        config = yaml.safe_load(file)
        convert_into_atomic_files(config)