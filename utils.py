import torch
import numpy as np
import os
import random
import warnings

from simpletransformers.classification import MultiLabelClassificationModel
from simpletransformers.config.model_args import MultiLabelClassificationArgs
from sparknlp.training import CoNLL
from sparknlp import SparkSession
import sparknlp

import consts
from models.bert import BertWrapper


def seed_torch(seed=2137):
    #  taken from: https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def turn_off_stupid_warnings():
    # ugly, but simpletransfomers.T5 throws some stupid
    # deprecation warnings if everything is done the way
    # the official tutorial says: https://simpletransformers.ai/docs/t5-model/
    warnings.filterwarnings("ignore", category=FutureWarning)


def prepare_environment():
    turn_off_stupid_warnings()
    seed_torch()


def getDataset(path):
    sparknlp.start()
    spark = SparkSession.builder \
        .appName("Spark NLP")\
        .master("local[4]")\
        .config("spark.driver.memory","16G")\
        .config("spark.driver.maxResultSize", "0") \
        .config("spark.kryoserializer.buffer.max", "2000M")\
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.4.4")\
        .getOrCreate()

    return (CoNLL().readDataset(spark, path)).toPandas()


def prepare_evaluation_data(data):
    inputs = data['text'].astype(str).values.tolist()
    labels = data['label1'].astype(int).values.tolist()
    return inputs, labels


def prepare_multilabel_data(data, datasize: int):
    data = data[:datasize] #reduce size

    partOfSpeach = data['partOfSpeach'].unique()
    syntactic = data['syntactic'].unique()
    entityTag = data['entityTag'].unique()

    labels = []

    for index, row in data.iterrows():
        label = [0]*(len(partOfSpeach) + len(syntactic) + len(entityTag))
        label[np.where(partOfSpeach == row['partOfSpeach'])[0][0]] = 1
        label[np.where(syntactic == row['syntactic'])[0][0] + len(partOfSpeach)] = 1
        label[np.where(entityTag == row['entityTag'])[0][0] + len(partOfSpeach) + len(syntactic)] = 1
        labels.append(label)

    data = data.drop(columns=['partOfSpeach', 'syntactic', 'entityTag'])
    data['labels'] = labels
    return data


def prepare_bert_train(number_of_rows: int, use_cuda) -> BertWrapper:
    bert_args = MultiLabelClassificationArgs(
        num_train_epochs=consts.EPOCHS,
        overwrite_output_dir=True,
        output_dir=f"{consts.BERT_OUTPUT}-{number_of_rows}",
    )
    bert = MultiLabelClassificationModel(
        consts.BERT_MODEL_TYPE,
        consts.BERT_MODEL_NAME,
        use_cuda=use_cuda,
        num_labels=3,
        args=bert_args
    )
    return BertWrapper(bert)

def prepare_bert_eval(number_of_rows: int, use_cuda) -> BertWrapper:
    bert_args = MultiLabelClassificationArgs(
        num_train_epochs=consts.EPOCHS,
        overwrite_output_dir=True,
        output_dir=f"{consts.BERT_OUTPUT}-{number_of_rows}-eval",
    )
    bert = MultiLabelClassificationModel(
        consts.BERT_MODEL_TYPE,
        f"{consts.BERT_OUTPUT}-{number_of_rows}",
        use_cuda=use_cuda,
        num_labels=3,
        args=bert_args
    )
    return BertWrapper(bert)
