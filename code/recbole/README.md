# Recbole Baseline Code

영화 추천 대회를 위해 recbole library를 사용한 베이스라인 코드입니다.

## Installation

```
pip install -r requirements.txt
```

## How to run

1. data preprocessing
   recbole에서 요구하는 데이터 형식에 맞추기 위해 변환한 파일을 저장하는 코드입니다.
   ```
   python data_preprocessing.py
   ```
2. Set Config file
   recbole에서 제공하는 모델 중 하나를 골라서 {model_name}.yaml 파일로 저장합니다.  
   참고 사이트: https://recbole.io/docs/recbole/recbole.model.html
    
3. Training
    ```
    python train.py --model model_name
    ```
4. Inference  
   training 이후 저장된 모델(.pth)의 경로와 함께 submission 파일을 저장합니다.
     
   ```
   python inference.py --model_path model_path
   ```
