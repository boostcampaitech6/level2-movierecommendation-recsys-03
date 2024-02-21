from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
#from recbole.model.general_recommender import EASE
from team.code.custom.EASE import EASE
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import pandas as pd
import os

if __name__ == '__main__':

    # configurations initialization
    config = Config(
        model='EASE',
        dataset = 'movie',
        config_file_list=['../recbole/configs/EASE.yaml'],
    )

    #bring interactions
    interactions = pd.read_csv('../../data/train/train_pearson.csv')

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = EASE(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)
    print(test_result)