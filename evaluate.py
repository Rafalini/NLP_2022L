import pandas as pd
import torch

import consts
import utils

# use_cuda = torch.cuda.is_available()
use_cuda = False


if __name__ == '__main__':
    utils.prepare_environment()

    data = pd.read_csv(consts.CONL_PREPROC_TEST)
    data = utils.prepare_multilabel_data(data, consts.INIT_TRAIN_SIZE)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    print(data.head)
    # inputs = data['text'].astype(str).values.tolist()
    # labels = data['labels'].astype(int).values.tolist()

    # while train_size <= consts.MAX_TRAIN_SIZE:
    torch.cuda.empty_cache()
    print("="*100)
    print(f"Evaluating for nrows={consts.INIT_TRAIN_SIZE}")

    model = utils.prepare_bert_eval(consts.INIT_TRAIN_SIZE, use_cuda)
    result = model.eval_model(data)


    # print(f" -Number of predicted sarcasms: {sum(predictions)}\n")
    #
    # print(f" -Precision:  {precision_score(labels, predictions, average='micro')}")
    # print(f" -Accuracy:   {accuracy_score(labels, predictions)}")
    # print(f" -Recall:     {recall_score(labels, predictions, average='weighted')}")
    # print(f" -F1 score:   {f1_score(labels, predictions, average='macro')}")
    #
    #     train_size += consts.STEP
