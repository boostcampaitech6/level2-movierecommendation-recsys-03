import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

data_path = '../../data/train'
interactions = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))
directors = pd.read_csv(os.path.join(data_path, 'directors.tsv'), sep='\t')
genres = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep='\t')
titles = pd.read_csv(os.path.join(data_path, 'titles.tsv'), sep='\t')
writers = pd.read_csv(os.path.join(data_path, 'writers.tsv'), sep='\t')
years = pd.read_csv(os.path.join(data_path, 'years.tsv'), sep='\t')


## items 파일

#Genre, Writer, Director 한칸에 넣기
grouped_genres= genres.groupby('item')['genre'].apply(list)
grouped_writers = writers.groupby('item')['writer'].apply(list)
grouped_directors = directors.groupby('item')['director'].apply(list)
items = pd.merge(titles, years, on='item')
items = pd.merge(items, grouped_genres, on='item', how = 'left')
items = pd.merge(items, grouped_writers, on='item', how = 'left')
items = pd.merge(items, grouped_directors, on='item', how = 'left')
# year_bin 추가
year_counts = interactions.merge(items[['item','year']], on='item',how='left')['year'].value_counts()
year_counts = pd.DataFrame(year_counts)
year_counts.reset_index(inplace=True)
year_counts.columns = ['year', 'count']
year_counts['cumulative_count'] = year_counts['count'].cumsum()
labels = [f'year_{i}' for i in range(3,0,-1)]  # 10개 구간 레이블
year_counts['year_bin'] = pd.cut(year_counts['cumulative_count'], bins=3, labels=labels)
items = items.merge(year_counts[['year', 'year_bin']], on='year', how='left')
# popularity_bin 추가
item_counts = interactions['item'].value_counts()
item_counts = pd.DataFrame(item_counts)
item_counts.reset_index(inplace=True)
item_counts.columns = ['item', 'count']
#item count 기준
item_counts['cumulative_count'] = item_counts['count'].cumsum()
labels = [f'pop_{i}' for i in range(3,0,-1)]  # 10개 구간 레이블
item_counts['popularity_bin'] = pd.cut(item_counts['cumulative_count'], bins=3, labels=labels)
items = items.merge(item_counts[['item', 'popularity_bin']], on='item', how='left')




## Users 파일

#item 추가
Users = interactions.groupby('user')['item'].agg(list).reset_index()
Users['item_count'] = Users['item'].apply(len)
merged_interactions = interactions.merge(items, on='item', how='left')
#popularity 추가
popularity_filled = merged_interactions.groupby(['user', 'popularity_bin']).size().unstack(fill_value=0)
#new_column_names = {pop_bin: f'pop_{int(pop_bin)}' for pop_bin in popularity_filled.columns}
#popularity_filled = popularity_filled.rename(columns=new_column_names)
Users = pd.merge(Users, popularity_filled, on='user')
#year 추가
year_filled = merged_interactions.groupby(['user', 'year_bin']).size().unstack(fill_value=0)
#new_column_names = {year_bin: f'year_{int(year_bin)}' for year_bin in year_filled.columns}
#year_filled = year_filled.rename(columns=new_column_names)
Users = pd.merge(Users, year_filled, on='user')
#genre,writer,director 추가
def process_genres(df, top_n=3, label='genre'):
    rows = []  # 결과를 저장할 리스트 초기화
    # 각 사용자별로 반복
    for user, row in df.iterrows():
        # 각 장르(또는 대체 용어)의 상호작용 횟수를 내림차순으로 정렬
        sorted_df = row.sort_values(ascending=False)
        # 상위 N개 장르와 해당 상호작용 횟수 추출
        top_categories = sorted_df[:top_n].index.tolist()
        top_counts = sorted_df[:top_n].values.tolist()
        # 결과 딕셔너리 생성
        row_dict = {'user': user}
        for i in range(top_n):
            row_dict[f'{label}_{i+1}'] = top_categories[i] if i < len(top_categories) else None
            row_dict[f'{label}_{i+1}_count'] = top_counts[i] if i < len(top_counts) else 0
        # 나머지 장르(또는 대체 용어)의 상호작용 횟수 합산
        etc_count = sorted_df[top_n:].sum()
        row_dict[f'{label}_etc'] = 'etc'
        row_dict[f'{label}_etc_count'] = etc_count
        # 결과 리스트에 추가
        rows.append(row_dict)
    # 결과 리스트를 DataFrame으로 변환
    result_df = pd.DataFrame(rows)
    result_df.set_index('user', inplace=True)
    return result_df

interactions_genres = interactions.merge(genres, on='item', how='left')
genre_filled = interactions_genres.groupby(['user', 'genre']).size().unstack(fill_value=0)
interactions_directors = interactions.merge(directors, on='item', how='left')
director_filled = interactions_directors.groupby(['user', 'director']).size().unstack(fill_value=0)
interactions_writers = interactions.merge(writers, on='item', how='left')
writer_filled = interactions_writers.groupby(['user', 'writer']).size().unstack(fill_value=0)

Users = pd.merge(Users, process_genres(genre_filled,label='genre'), on='user')
Users = pd.merge(Users, process_genres(director_filled,label='director'), on='user')
Users = pd.merge(Users, process_genres(writer_filled,label='writer'), on='user')

# percent 추가
for column in Users.columns:
    if column in ['pop_3','pop_2','pop_1','year_1','year_2','year_3','genre_1_count','genre_2_count','genre_3_count'
                      ,'director_1_count','director_2_count','director_3_count','writer_1_count','writer_2_count','writer_3_count']:
        Users[column+'_percent'] = Users[column].div(Users['item_count'],axis=0)

# type 추가 
Users['type_popular'] = Users['pop_3_percent'] >= 0.7
Users['type_unpopular'] = Users['pop_1_percent'] >= 0.6
Users['type_new'] = Users['year_3_percent'] >= 0.7
Users['type_old'] = Users['year_1_percent'] >= 0.6
Users['type_genre'] = Users['genre_1_count_percent'] >= 0.7
Users['type_director'] = Users['director_1_count_percent'] >= 0.08
Users['type_writer'] = Users['writer_1_count_percent'] >= 0.08
Users['type_mania'] = Users['item_count']>300
Users['type_novice'] = Users['item_count']<50

true_counts = {
    'type_popular': Users['type_popular'].sum(),
    'type_unpopular': Users['type_unpopular'].sum(),
    'type_new': Users['type_new'].sum(),
    'type_old': Users['type_old'].sum(),
    'type_genre': Users['type_genre'].sum(),
    'type_director': Users['type_director'].sum(),
    'type_writer': Users['type_writer'].sum(),
    'type_mania': Users['type_mania'].sum(),
    'type_novice': Users['type_novice'].sum()
}

print(true_counts)

Users['All_users'] = True

Users['type'] = 'popular'
Users['type'] = Users.apply(lambda Users: 'genre' if Users['type_genre'] else Users['type'], axis=1)
Users['type'] = Users.apply(lambda Users: 'writer' if Users['type_writer'] else Users['type'], axis=1)
Users['type'] = Users.apply(lambda Users: 'director' if Users['type_director'] else Users['type'], axis=1)
Users['type'].value_counts()

Users = Users.drop('item', axis=1)
items.to_csv(os.path.join(data_path, 'EDA_items.csv'), sep=',', index=False)
Users.to_csv(os.path.join(data_path, 'EDA_users.csv'), sep=',', index=False)



## Submission 분석 #
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

data_path = 'data/train'
items = pd.read_csv(os.path.join(data_path, 'EDA_items.csv'), sep=',')
Users = pd.read_csv(os.path.join(data_path, 'EDA_users.csv'), sep=',')

output_path = 'code/recbole/output/'

submission_name='easer'
sub = pd.read_csv(output_path+'submission_'+submission_name+'.csv')
sub = pd.merge(sub, items, on='item')

import ast
# 문자열을 리스트로 변환하는 함수 정의
def convert_str_to_list(str_list):
    try:
        return ast.literal_eval(str_list)
    except ValueError:  # 문자열이 올바른 리스트 형식이 아닐 경우
        return []  # 빈 리스트 반환
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


## EDA ##

#year, popularity PLOT
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