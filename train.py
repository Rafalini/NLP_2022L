import pandas as pd
import torch
from trainer import OurTrainer
import consts
import utils
import os

# use_cuda = torch.cuda.is_available()
use_cuda = False

if __name__ == '__main__':
    utils.prepare_environment()
    print(os.getcwd())

    data = pd.read_csv(consts.CONL_PREPROC_TEST)
    data = utils.prepare_multilabel_data(data, consts.INIT_TRAIN_SIZE)

    train_size = consts.INIT_TRAIN_SIZE
    # while train_size <= consts.MAX_TRAIN_SIZE:
    torch.cuda.empty_cache()
    print("=" * 100)
    print(f"Training for nrows={train_size}")

    model = utils.prepare_bert_train(train_size, use_cuda)
    trainer = OurTrainer(model)
    trainer.train(data)

        # train_size += consts.STEP
