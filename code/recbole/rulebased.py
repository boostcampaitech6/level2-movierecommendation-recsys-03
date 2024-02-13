import os 
import numpy as np
import pandas as pd

data_path = '/data/data/train'
interactions = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
genres = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep='\t')
# interactions과 genres를 item을 기준으로 병합
merged = interactions.merge(genres, on='item')
# 사용자별로 본 장르를 리스트로 묶기
user_genre_counts = merged.groupby('user')['genre'].value_counts()
print(user_genre_counts)
most_watched_genre_per_user = user_genre_counts.groupby(level=0).idxmax()
print(most_watched_genre_per_user)


user_item_group = interactions.groupby('user')['item'].apply(list)
print(user_item_group)

item_counts = interactions['item'].value_counts()
print(item_counts)

recommendations = []

# 각 사용자에 대해서
for user, watched_items in user_item_group.items():
    # 아직 시청하지 않은 아이템을 찾는다
    unwatched_items = item_counts[~item_counts.index.isin(watched_items)]
    # 사용자가 가장 많이 시청한 장르의 아이템만 선택
    genre_items = genres[genres['genre'] == most_watched_genre_per_user[user][1]]['item']
    preferred_items = unwatched_items[unwatched_items.index.isin(genre_items)]
    # 시청 횟수가 많은 순서대로 아이템을 정렬한다
    recommended_items = preferred_items.sort_values(ascending=False)
    # 상위 10개 아이템을 추천한다
    top_10_items = recommended_items[:10]
    # 추천 결과를 저장한다
    for item in top_10_items.index:
        recommendations.append((user, item))


# 추천 결과를 DataFrame으로 변환한다
recommendations_df = pd.DataFrame(recommendations, columns=['user', 'item'])
print(recommendations_df)
recommendations_df.to_csv('output/submission_rule_genres.csv',index=False)