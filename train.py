import pandas as pd
import torch
from simpletransformers.classification import ClassificationModel

from models.bert import BertWrapper
from trainer import OurTrainer
import consts
import utils
import os

# use_cuda = torch.cuda.is_available()
use_cuda = False


def prepare_bert(number_of_rows: int, num_labels: int) -> BertWrapper:
    bert_args = consts.BERT_ARGS
    bert_args.output_dir = f"{consts.BERT_OUTPUT}-{number_of_rows}"

    bert = ClassificationModel(
        consts.BERT_MODEL_TYPE,
        consts.BERT_MODEL_NAME,
        args=bert_args,
        use_cuda=use_cuda,
        num_labels=num_labels
    )
    return BertWrapper(bert)


if __name__ == '__main__':
    utils.prepare_environment()
    print(os.getcwd())
    raw_train_data = pd.read_csv(consts.CONL_PREPROC_TRAIN)
    # test_data = pd.read_csv(consts.TEST_DATA)
    # test_inputs, test_labels = utils.prepare_evaluation_data(test_data)

    train_size = consts.INIT_TRAIN_SIZE
    while train_size <= consts.MAX_TRAIN_SIZE:
        torch.cuda.empty_cache()
        print("=" * 100)
        print(f"Training for nrows={train_size}")
        data = raw_train_data[0: train_size]
        print(data['label1'].value_counts())
        model = prepare_bert(train_size, len(data['label1'].unique())+1)
        trainer = OurTrainer(model)
        trainer.train(data)

        # test_preds = trainer.model.predict(test_inputs)
        # print(f" Number of positives: {sum(test_preds)}")

        train_size += consts.STEP
