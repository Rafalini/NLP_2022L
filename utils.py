import torch
import numpy as np
import os
import random
import warnings
from sparknlp.training import CoNLL
from sparknlp import SparkSession
import sparknlp


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
    # trainingData = (CoNLL().readDataset(spark, path)).toPandas()
    # print(type(trainingData))
    # trainingData.selectExpr(
    #     "text",
    #     "token.result as tokens",
    #     "pos.result as pos",
    #     "label.result as label"
    # ).show(3, False)


def prepare_evaluation_data(data):
    inputs = data['text'].astype(str).values.tolist()
    labels = data['entityTag'].astype(int).values.tolist()
    return inputs, labels
