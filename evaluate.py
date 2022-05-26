import pandas as pd
from simpletransformers.classification import ClassificationModel
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from models.bert import BertWrapper
import consts
import utils

# use_cuda = torch.cuda.is_available()
use_cuda = False

def prepare_bert(number_of_rows: int, num_labels: int) -> BertWrapper:
    bert_args = consts.BERT_ARGS
    bert_args.output_dir = f"{consts.BERT_OUTPUT}-{number_of_rows}-eval"

    bert = ClassificationModel(
        consts.BERT_MODEL_TYPE,
        f"{consts.BERT_OUTPUT}-{number_of_rows}",
        args=bert_args,
        use_cuda=use_cuda,
        num_labels=num_labels
    )
    return BertWrapper(bert)



if __name__ == '__main__':
    utils.prepare_environment()

    if consts.ENCODING == "BIO":
        data = pd.read_csv(consts.CONL_PREPROC_TEST_BIO)
    if consts.ENCODING == "IO":
        data = pd.read_csv(consts.CONL_PREPROC_TEST_IO)
    if consts.ENCODING == "IOB":
        data = pd.read_csv(consts.CONL_PREPROC_TEST_IOB)

    inputs, labels = utils.prepare_evaluation_data(data)

    train_size = consts.INIT_TRAIN_SIZE

    while train_size <= consts.MAX_TRAIN_SIZE:
        torch.cuda.empty_cache()
        print("="*100)
        print(f"Evaluating for nrows={train_size}")

        model = prepare_bert(train_size, len(data['entityTag'].unique()))
        inputs = inputs[1: train_size]
        labels = labels[1: train_size]

        predictions = model.predict(inputs)
        print(inputs)
        print(labels)
        print(predictions)

        print(f" -Precision:  {precision_score(labels, predictions, average='macro')}")
        print(f" -Accuracy:   {accuracy_score(labels, predictions) }")
        print(f" -Recall:     {recall_score(labels, predictions, average='macro')}")
        print(f" -F1 score:   {f1_score(labels, predictions, average='macro')}")

        train_size += consts.STEP
