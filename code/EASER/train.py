import sys
sys.path.append("../")

from model import EASER
import pandas as pd
import os
from datetime import datetime
from pytz import timezone
from train_args import parse_args
from dataloader import *
from datetime import datetime
from preprocessing import MakeMatrixDataSet
from utils import evaluate

import warnings
warnings.filterwarnings(action="ignore")


def main(args):
    data_loader = Dataloader(config = args)
    train_df, users, items = data_loader.dataloader()
    
    model = EASER(args)
    model.fit(train_df)
    # HIT_RATE, NDCG = evaluate(pred, train_df, users)
    # print("Hit Rate: ", HIT_RATE, "NDCG : ", NDCG)
    result_df = model.predict(train_df, users, items, args.K)

    file_name = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
    result_df[["user", "item"]].to_csv(args.output_dir + f"{file_name}_{args.K}.csv", index=False)

   
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
    