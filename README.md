# Movie Recommendation

## OverView
본 프로젝트는 'Movie Recommendation'으로, 사용자의 영화 시청 이력 데이터를 토대로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측하는 기존 대회들과는 다르게 사용자의 영화 시청 이력에서 10개를 랜덤으로 제거한 후 제거된 시청 이력이 무엇인지를 맞춰야 하는 대회이다.

모든 사용자에게 10개의 영화를 추천하며, 이때 추천 리스트의 정확성(Recall@10)을 평가 기준으로 삼는다.

평가를 위한 정답(ground-truth) 데이터는 Sequential Recommendation 시나리오를 바탕으로 사용자의 Time-Ordered Sequence에서 일부 Item이 누락(dropout)된 상황을 상정한다
<br><br>

## Component

### 프로젝트 디렉토리 구조 
```
📦level2-movierecommendation-recsys-03-main
 ┗ 📂code
   ┣ 📂EASER
   ┣ 📂EDA
   ┣ 📂bert4rec
   ┣ 📂custom
   ┣ 📂multi
   ┣ 📂recbole
   ┃ ┗ 📂configs - .yaml
   ┃   ┣ ADMMSLIM, DIFFRec, EASE, deepfM
   ┃   ┣ fm, lightgcn, ract, recvae, 
   ┃   ┗ seq, slim
   ┣ 📂s3rec
   ┗ 📂voting
```
### 데이터셋 구조
```
📦level2-movierecommendation-recsys-03-main
 ┗ 📂train
	 ┣ 📜Ml_item2attributes.json
	 ┣ 📜directors.tsv
	 ┣ 📜genres.tsv
	 ┣ 📜titles.tsv
	 ┣ 📜train_ratings.csv
	 ┣ 📜writers.tsv
	 ┗ 📜years.tsv
```


## Team
<br>
<table align="left">
  <tr height="155px">
    <td align="center" width="150px">
      <a href="https://github.com/ksb3966"><img src="https://github.com/ksb3966.png" width="100px;" alt=""/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/SiwooPark00"><img src="https://github.com/SiwooPark00.png" width="100px;" alt=""/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/arctic890"><img src="https://github.com/arctic890.png" width="100px;" alt=""/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/JaeGwon-Lee"><img src="https://github.com/JaeGwon-Lee.png" width="100px;" alt=""/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/jinmin111"><img src="https://github.com/jinmin111.png" width="100px;" alt=""/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/chris3427"><img src="https://github.com/chris3427.png" width="100px;" alt=""/></a>
    </td>
  </tr>
  <tr height="80px">
    <td align="center" width="150px">
      <a href="https://github.com/ksb3966">김수빈_T6021</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/SiwooPark00">박시우_T6060</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/arctic890">백승빈_T6075</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/JaeGwon-Lee">이재권_T6131</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/jinmin111">이진민_T6139</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/chris3427">장재원_T6149</a>
    </td>
  </tr>
</table>
&nbsp;
<br>

## Role

| 이름 | 역할 |
| --- | --- |
| 김수빈 | EDA, 모델 선정 및 튜닝, EASER, SASRec, Bert4Rec 실험 수행   |
| 박시우 | EDA, s3rec baseline 정리, recbole baseline 구축, RecVAE, ADMMSLIM, soft voting 앙상블 구현 |
| 백승빈 | EDA, EASE, SLIMElastic, DiffRec 모델 실험 및 튜닝, hard voting기반 앙상블 구현 |
| 이재권 | EDA, LightGCN 모델 실험 및 튜닝 |
| 이진민 | EDA, Multi-DAE, Multi-VAE 코드 모듈화 및 실험 |
| 장재원 | EDA, Sequential models, Type based model, 모델들 성능 비교 및 분석 |
<br>

## Experiment Result

### Single Model Result
|  | Public Recall@10 | Private Recall@10 |
| --- | --- | --- |
| Popular item rule based model | 0.0673 | 0.0671 |
| Genre rule based model | 0.0619 | 0.0626 |
| Type based model | 0.0687 | 0.0696 |
| GRU4Rec | 0.0970 | 0.0809 |
| SASRec | 0.0884 | 0.0833 |
| S3Rec(Pretrained) | 0.0829 | 0.0743 |
| BERT4Rec | 0.0687 | 0.0676 |
| LightGCN | 0.1302 | 0.1316 |
| DiffRec | 0.1413 | 0.1431 |
| RecVAE | 0.1349 | 0.1362 |
| Multi-VAE | 0.1394 | 0.1377   |
| Multi-DAE (with side information) | 0.1427 | 0.1413 |
| EASE | 0.1566 | 0.1565 |
| ADMMSLIM | 0.1524 | 0.1541 |
| SLIMElastic | 0.1562 | 0.1562  |
| EASER | 0.1612 | 0.1603 |

### Ensemble Result
|  | Private Recall@10 | Public Recall@10 |
| --- | --- | --- |
| EASER, 앙상블 모델*에 type별 추천 적용 | 0.1614 | 0.1605 |
| 앙상블 모델*(EASE, ADMMSLIM, RecVAE) | 0.1613 | 0.1611 |

최종적으로는 가장 높은 Public score 값에 해당하는 두 결과 값을 제출하였다.
<br><br>

## Wrap-Up Report
[MovieREc Wrap-up Report - Suggestify.pdf](./DKT.Wrap-up.Report.-.Suggestify.pdf)
<br>
