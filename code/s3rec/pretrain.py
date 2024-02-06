from pretrain_args import parse_args
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from datasets import PretrainDataset
from models import S3RecModel
from trainers import PretrainTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs_long,
    set_seed,
)

import wandb

def main(args):

    set_seed(args.seed)
    check_path(args.output_dir)

    args.checkpoint_path = os.path.join(args.output_dir, "Pretrain.pt")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # args.data_file = args.data_dir + args.data_name + '.txt'
    args.data_file = args.data_dir + "train_ratings.csv"
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"
    # concat all user_seq get a long sequence, from which sample neg segment for SP
    user_seq, max_item, long_sequence = get_user_seqs_long(args.data_file)

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    args.item2attribute = item2attribute

    model = S3RecModel(args=args)
    trainer = PretrainTrainer(model, None, None, None, None, args)

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)

    wandb.login()
    wandb.init(project = args.project_name,
               name = args.run_name,
               entity = args.entity_name,
               config = {
                   "epochs": args.pre_epochs,
                   "batch_size": args.pre_batch_size,
                   "hidden_size": args.hidden_size,
                   "num_hidden_layers": args.num_hidden_layers,
                   "attention_probs_dropout_prob": args.attention_probs_dropout_prob,
                   "hidden_dropout_prob": args.hidden_dropout_prob,
                   "max_seq_length": args.max_seq_length,
                   "mask_probability": args.mask_p,
                   "aap_weight": args.aap_weight,
                   "mip_weight": args.mip_weight,
                   "map_weight": args.map_weight,
                   "sp_weight": args.sp_weight,
               })
    for epoch in range(args.pre_epochs):

        pretrain_dataset = PretrainDataset(args, user_seq, long_sequence)
        pretrain_sampler = RandomSampler(pretrain_dataset)
        pretrain_dataloader = DataLoader(
            pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size
        )

        losses = trainer.pretrain(epoch, pretrain_dataloader)

        ## comparing `sp_loss_avg``
        early_stopping(np.array([-losses["sp_loss_avg"]]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
