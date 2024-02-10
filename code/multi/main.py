from utils import Logger, Setting
from preprocessing import data_load, DataLoader
from trainer import run
import pandas as pd
import argparse
from models import MultiDAE, MultiVAE
from inference import predict
import warnings
import os
import wandb



def main(args):
    wandb.login()
    wandb.init(project=args.wandb_project_name, entity = 'suggestify_lv2', config=vars(args))

    Setting.seed_everything(args.seed)

    ######################## DATA LOAD
    print(f'--------------- {args.model} Load Data ---------------')
    id2show, id2profile = data_load(args)

    loader = DataLoader(args.data)

    n_items = loader.load_n_items()


    ####################### Setting for Log
    setting = Setting()

    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)

    logger = Logger(args, log_path)
    logger.save_args()

        
    filename = setting.get_submit_filename(args)

    wandb.run.name = filename[9:-4]
    wandb.run.save()

    ######################## Model Load
    print(f'--------------- INIT {args.model} ---------------')
    p_dims = [200, 600, n_items]
    if args.model=='Multi-DAE':
        model = MultiDAE(p_dims).to(args.device)
    elif args.model=='Multi-VAE':
        model = MultiVAE(p_dims).to(args.device)

    ######################## TRAIN
    print(f'--------------- {args.model} TRAINING ---------------')
    model = run(args, model, loader, logger, setting)
    
    ######################## INFERENCE
    print(f'--------------- {args.model} PREDICT ---------------')
    train_data = loader.load_data('train')
    predicts = predict(args, model, train_data)

    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    # submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    # submission['rating'] = predicts

    result = pd.DataFrame(predicts, columns=['user', 'item'])

    result['user'] = result['user'].apply(lambda x : id2profile[x])
    result['item'] = result['item'].apply(lambda x : id2show[x])
    result = result.sort_values(by='user')

    write_path = os.path.join(filename)
    result.to_csv(write_path, index=False)

    wandb.finish()


    # submission.to_csv(filename, index=False)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    ## 각종 파라미터 세팅
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')

    parser.add_argument('--data', type=str, default='../../../data/train/',
                        help='Movielens dataset location')
    parser.add_argument('--model', type=str, default='Multi-DAE',
                        help='model name')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='use gpu')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='./saved',
                        help='path to save the final model')
    parser.add_argument("--wandb_project_name", default="movierec-multi", type=str, help="Setting WandB Project Name")
    
    args = parser.parse_args([])
    
    main(args)