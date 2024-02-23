import os 
import numpy as np
import pandas as pd

data_path = '/data/data/train'
interactions = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
genres = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep='\t')
writers = pd.read_csv(os.path.join(data_path, 'writers.tsv'), sep='\t')
directors = pd.read_csv(os.path.join(data_path, 'directors.tsv'), sep='\t')


Users = pd.read_csv(os.path.join(data_path, 'EDA_users.csv'), sep=',')


user_item_group = interactions.groupby('user')['item'].apply(list)
print(user_item_group)

item_counts = interactions['item'].value_counts()
print(item_counts)

recommendations = []

# 각 사용자와 그들의 시청 항목에 대해 반복
for user, watched_items in user_item_group.items():
    # 시청하지 않은 항목을 찾음
    unwatched_items = item_counts[~item_counts.index.isin(watched_items)]

    # 사용자의 선호 타입을 확인
    user_type = Users.loc[Users['user'] == user, 'type'].values[0]

    if user_type == 'genre':
        # 사용자의 가장 많이 시청한 장르에서 시청하지 않은 항목을 선택
        genre_1 = Users.loc[Users['user'] == user, 'genre_1'].values[0]
        genre_items = genres[genres['genre'] == genre_1]['item']
        preferred_items = unwatched_items[unwatched_items.index.isin(genre_items)]
    elif user_type == 'writer':
        # 사용자의 가장 선호하는 작가의 작품 중 시청하지 않은 항목을 선택
        preferred_writer = Users.loc[Users['user'] == user, 'writer_1'].values[0]
        writer_items = writers[writers['writer'] == preferred_writer]['item']
        preferred_items = unwatched_items[unwatched_items.index.isin(writer_items)]
    elif user_type == 'director':
        # 사용자의 가장 선호하는 감독의 작품 중 시청하지 않은 항목을 선택
        preferred_director = Users.loc[Users['user'] == user, 'director_1'].values[0]
        director_items = directors[directors['director'] == preferred_director]['item']
        preferred_items = unwatched_items[unwatched_items.index.isin(director_items)]
    else:
        # 다른 타입의 경우, 사용자가 시청하지 않은 항목 중에서 선택
        preferred_items = unwatched_items

    # 시청 횟수에 따라 항목을 내림차순으로 정렬
    recommended_items = preferred_items.sort_values(ascending=False)

    # 추천을 위한 상위 10개 항목을 선택
    top_items = recommended_items[:10]

    # preferred_items의 수가 10개 미만인 경우, 추가로 시청 횟수가 많은 항목들을 추천에 포함
    if len(top_items) < 10:
        extra_items_needed = 10 - len(top_items)
        extra_items = unwatched_items[~unwatched_items.index.isin(top_items.index)].sort_values(ascending=False)[:extra_items_needed]
        top_items = pd.concat([top_items, extra_items])

    # 추천 결과를 저장
    for item in top_items.index:
        recommendations.append((user, item))



# 추천 결과를 DataFrame으로 변환한다
recommendations_df = pd.DataFrame(recommendations, columns=['user', 'item'])
print(recommendations_df)
recommendations_df.to_csv('output/submission_rule_types2.csv',index=False)
