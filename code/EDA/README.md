# EDA : data와 submission 분석

## Setup
```
bash
cd code/EDA
conda init
(base) . ~/.bashrc
(base) conda create -n eda python=3.10 -y
(base) conda activate eda
(eda) pip install -r requirements.txt
```

## Files
`code/EDA`
* `data_EDA.py` : item,user에 대한 EDA파일을 생성합니다.
* `submission_EDA.py` : submission에 대한 EDA파일을 생성합니다.
* `rule_based_model.py` : User type을 기반으로 하는 rule based model입니다.

## 실행방법
* EDA_items.csv, EDA_users.csv 파일 생성
```
python data_EDA.py 
```
* submission_model_EDA.csv 파일 생성
```
python submission_EDA.py
```