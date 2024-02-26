from train_args import parse_args
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import tqdm

from preprocess import preprocess

from data_loaders import SeqDataset
from model.model import BERT4Rec
from trainer.trainer import *
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
)

import wandb
def main(args):
    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + "train_ratings.csv"
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, _ = get_user_seqs(
        args.data_file
    )

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    '''
    seq_dataset = SeqDataset(user_train, num_user, num_item, max_len, mask_prob)
    data_loader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True) # TODO4: pytorch의 DataLoader와 seq_dataset을 사용하여 학습 파이프라인을 구현해보세요.
    '''
    
    num_user, num_item, df, user_train, user_valid = preprocess(args)
    
    train_dataset = SeqDataset(user_train, num_user, num_item, args.max_seq_length, args.mask_prob)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size,
    )

    valid_dataset = SeqDataset(user_valid, num_user, num_item, args.max_seq_length, args.mask_prob)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(
        valid_dataset, sampler=valid_sampler, batch_size=args.batch_size
    )

    test_dataset = SeqDataset(df, num_user, num_item, args.max_seq_length, args.mask_prob)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.batch_size
    )

    model = BERT4Rec(num_user, num_item, args.hidden_units, 
                     args.num_attention_heads, args.num_hidden_layers, 
                     args.max_seq_length, args.attention_probs_dropout_prob, args.device)

    wandb.login()
    wandb.init(project = args.project_name,
               name = args.run_name,
               entity = args.entity_name,
               config = {
                   "epochs": args.epochs,
                   "learning_rate": args.lr,
                   "batch_size": args.batch_size,
                   "using_pretrain": args.using_pretrain,
                   "hidden_size": args.hidden_size,
                   "num_hidden_layers": args.num_hidden_layers,
                   "attention_probs_dropout_prob": args.attention_probs_dropout_prob,
                   "hidden_dropout_prob": args.hidden_dropout_prob,
                   "max_seq_length": args.max_seq_length,
               })
    
    
        #-- loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # default: cross_entropy # mask 안한 거 빼고 함
    
    #-- optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    #-- Earlystopping
    patience = args.patience
    stop_counter = 0
    best_val_acc = 0
    best_val_loss = np.inf
    
    print (f"[DEBUG] Start of TRAINING")
    
    for epoch in range(1, args.epochs + 1):
        #-- training
        loss_sum = 0
            
        tqdm_bar = tqdm.tqdm(train_dataloader)
            
        for idx, (log_seqs, labels) in enumerate(tqdm_bar):
            logits = model(log_seqs)
                
            # size matching
            logits = logits.view(-1, logits.size(-1))   # [51200, 6808]
            labels = labels.view(-1).to(args.device)         # 51200
                
            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss_sum += loss
            loss.backward()
            optimizer.step()
                
            tqdm_bar.set_description(f'Epoch: {epoch + 1:3d}| Step: {idx:3d}| Train loss: {loss:.5f}')
            
        loss_avg = loss_sum / len(train_dataloader)

        #-- validataion
        torch.cuda.empty_cache()
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            masked_cnt = 0
            correct_cnt = 0

            for _log_seqs, _labels in valid_dataloader:
                _logits = model(_log_seqs)

                y_hat = _logits[:,:].argsort()[:,:,-1].view(-1)

                # size matching
                _logits = _logits.view(-1, _logits.size(-1))   # [51200, 6808]
                _labels = _labels.view(-1).to(args.device)         # 51200

                _loss = criterion(_logits, _labels)
                            
                correct_cnt += torch.sum((_labels == y_hat) & (_labels != 0))
                masked_cnt += _labels.count_nonzero()
                valid_loss += _loss
                        
            valid_loss_avg = valid_loss / len(valid_dataloader)
            valid_acc = correct_cnt / masked_cnt
                
            if valid_loss_avg < best_val_loss:
                print(f"New best model for val loss : {valid_loss_avg:.5f}! saving the best model..")
                torch.save(model, f"{args.output_dir}/best.pth")
                best_val_loss = valid_loss_avg
                best_val_acc = valid_acc
                stop_counter = 0
    
            else:
                stop_counter += 1
                print (f"!!! Early stop counter = {stop_counter}/{patience} !!!")
                
            torch.save(model, f"{args.output_dir}/last.pth")
                
            print(
                f"[Val] acc : {valid_acc:4.2%}, loss: {valid_loss_avg:.5f} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:.5f}"
            )

            if stop_counter >= patience:
                print("Early stopping")
                break
        #-- [MLflow] save last model artifacts to mlflow
    
    '''
    for epoch in range(args.epochs):
        trainer.train(epoch)

        scores, _ = trainer.valid(epoch)

        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    trainer.args.train_matrix = test_rating_matrix
    print("---------------Change to test_rating_matrix!-------------------")
    # load the best model
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    scores, result_info = trainer.test(0)
    print(result_info)
    '''
    
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
