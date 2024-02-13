import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


data_path = '/data/data/train'
interactions = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
grouped_items = interactions.groupby('user')['item'].apply(list)


data_path = '/data/data/train'
interactions = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
directors = pd.read_csv(os.path.join(data_path, 'directors.tsv'), sep='\t')
genres = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep='\t')
titles = pd.read_csv(os.path.join(data_path, 'titles.tsv'), sep='\t')
writers = pd.read_csv(os.path.join(data_path, 'writers.tsv'), sep='\t')
years = pd.read_csv(os.path.join(data_path, 'years.tsv'), sep='\t')

#Genre, Writer, Director 한칸에 넣기
grouped_genres= genres.groupby('item')['genre'].apply(list)
grouped_writers = writers.groupby('item')['writer'].apply(list)
grouped_directors = directors.groupby('item')['director'].apply(list)

items = pd.merge(titles, years, on='item')
items = pd.merge(items, grouped_genres, on='item', how = 'left')
items = pd.merge(items, grouped_writers, on='item', how = 'left')
items = pd.merge(items, grouped_directors, on='item', how = 'left')

#기존
items = pd.merge(titles, years, on='item')
items = pd.merge(items, genres, on='item')
items = pd.merge(items, writers, on='item', how = 'left')
items = pd.merge(items, directors, on='item', how = 'left')

# year_bin 추가
bins = pd.cut(items['year'], bins=10)
bin_labels = [int(b.left) for b in bins.cat.categories]
items['year_bin'] = pd.cut(items['year'], bins=10, labels=bin_labels)

# popularity_bin 추가
item_counts = interactions['item'].value_counts()
item_counts = pd.DataFrame(item_counts)
item_counts.reset_index(inplace=True)
item_counts.columns = ['item', 'count']
item_counts['popularity_bin'] = pd.cut(item_counts['count'], bins=10, labels=False)
items = items.merge(item_counts[['item', 'popularity_bin']], on='item', how='left')




item_counts_by_year_bin = items.groupby('year_bin').size()
plt.bar(item_counts_by_year_bin.index, item_counts_by_year_bin.values)
plt.savefig('year-item수.png')


plt.figure(figsize=(10, 6))
item_counts_by_popularity_bin = items.groupby('popularity_bin').size()
plt.bar(item_counts_by_popularity_bin.index, item_counts_by_popularity_bin.values)
plt.savefig('popularity-item수.png')

# interactions에서의 item 수 : 6807개
# Title에서의 item 수 : 6807개 (위와 전부 동일) -> 중복 제외 6616
# Years에서의 item수 : 6799개
# Directors에서의 item수 : 5503개
# Writers에서의 item수 : 5648개
# 모든 Item은 Ineractions(Title)에 포함된다


## title 리메이크 확인
titles['title'] = titles['title'].apply(lambda x: re.sub(r'\s*\(\d{4}\)', '', x))
titles['title'].nunique()
titles.to_csv(os.path.join(data_path, 'title_sorted'), sep='\t', index=False)

## Directors
# 5905개의 data, 5503개의 영화
# 1340개의 감독중 1169개의 감독이 10편이상 촬영
# 58개의 영화는 2명이상의 감독이 있다. 한영화의 최대 감독수는 14명
directors['director'].value_counts
directors['item'].nunique()

## Writers
# 11307개의 data, 5648개의 영화
# Writers와 Directors는 4973개 일치
# 1307개의 영화는 2명이상의 작가가 있고. 한영화의 최대 작가수는 24명
writers['writer'].value_counts()    
writers['item'].nunique()

merged_df = pd.merge(directors, writers, on='item', how='inner')
common_items_count = merged_df['item'].nunique()
print("두 데이터프레임에서 공통된 item의 수:", common_items_count)


item_counts = writers.groupby('item').size()
print(len(item_counts[item_counts>10]))




interactions.to_csv(os.path.join(data_path, 'EDAinter.csv'), sep='\t', index=False)
print("Save interaction data ...")
print("Save user data ...")
   
items = pd.merge(titles, years, on='item',)
items = pd.merge(items, genres, on='item')
items = pd.merge(items, writers, on='item', how = 'left')
items = pd.merge(items, directors, on='item', how = 'left')
items = items.fillna("none")
items.to_csv(os.path.join(data_path,'EDAitem.csv'), sep='\t', index=False)
print("Save item data ...")


#Data 확인
min_time = interactions['time'].min()
max_time = interactions['time'].max()
# 1970년 1월 1일 자정(UTC)을 기준 : 2005년 3월 ~ 2015년 3월
# min : 1113220585  / max : 1427781052

#item_genre
item_genre_group = genres.groupby('item')['genre'].apply(list)
print(item_genre_group)
# interactions과 genres를 item을 기준으로 병합
merged = interactions.merge(genres, on='item')
#사용자별로 본 영화의 년도수 묶기
merged_df = pd.merge(interactions, items, on='item')
user_year_count = merged_df.groupby(['user', 'year']).size().reset_index(name='counts')
user_year_count.to_csv(os.path.join(data_path, 'user_year_count.csv'))
# 년도를 기준으로 2000년 이하와 이후로 구분
user_year_count['period'] = pd.cut(user_year_count['year'], bins=[0, 2000, user_year_count['year'].max()], labels=['2000년 이하', '2000년 이후'])

# 사용자별로 2000년 이하와 이후의 아이템을 본 카운트
user_period_count = user_year_count.groupby(['user', 'period'])['counts'].sum().reset_index()
user_period_count.to_csv(os.path.join(data_path, 'user_year_count_bin.csv'))


# 사용자별로 본 장르를 리스트로 묶기
user_genre_counts = merged.groupby('user')['genre'].value_counts()
print(user_genre_counts)
user_genre_counts.to_csv(os.path.join(data_path, 'item_genre_group.csv'))


# item별 시청횟수
# 6807개의 item중 2000개는 100회 이하 시청했다.
num_items = interactions['item'].nunique()
print('아이템의 개수:', num_items)

item_counts = interactions['item'].value_counts()
print(item_counts)

item_counts_over_100 = item_counts[item_counts >= 5000]
print(item_counts_over_100)

item_counts = interactions['item'].value_counts()
plt.figure(figsize=(12,6))
plt.bar(item_counts.index, item_counts.values)
plt.title('Item Interaction Counts')
plt.xlabel('Item ID')
plt.ylabel('Count')
plt.savefig('item_counts.png')

print(item_counts[10])




#User별 시청횟수
user_view_counts = interactions['user'].value_counts()
print(user_view_counts)

user_view_counts_over = user_view_counts[user_view_counts >= 100]
print(user_view_counts_over)


### Submission 결과 확인

submission = pd.read_csv(os.path.join('output/submission_EASE.csv'))
train = pd.read_csv('data/data/train/train_ratings.csv')

## 각 Train과 Submission 내에는 중복이 없음

submission_duplicates = submission.duplicated(subset=['user', 'item'])
train_duplicates = train.duplicated(subset=['user', 'item'])
print('submission 내부에 중복된 경우가 있는지:', submission_duplicates.any())
print('train 내부에 중복된 경우가 있는지:', train_duplicates.any())


## Sequential submission과 train에는 중복이 있음

# submission과 train 데이터셋에서 'user'와 'item' 열만 추출합니다.
submission_user_item = submission[['user', 'item']]
train_user_item = train[['user', 'item']]
# 두 데이터셋을 병합하고, 'user'와 'item' 열이 중복되는지 확인합니다.
merged = pd.concat([submission_user_item, train_user_item])
duplicates = merged.duplicated(subset=['user', 'item'])
# 중복된 경우가 있는지 확인합니다.
print('Sequential submission과 train에 중복된 경우가 있는지:', duplicates.any())

# 중복된 데이터의 개수를 확인합니다.
num_duplicates = duplicates.sum()
print('Sequential submission과 train에서 중복된 데이터의 개수:', num_duplicates)