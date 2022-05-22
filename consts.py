import os
from simpletransformers.classification import ClassificationArgs

DATA_DIR = './data/sourceData'
CONL_TRAIN_DATASET = os.path.join(DATA_DIR, 'conll2003', 'train.txt')
CONL_TEST_DATASET = os.path.join(DATA_DIR, 'conll2003', 'test.txt')
CONL_VALID_DATASET = os.path.join(DATA_DIR, 'conll2003', 'valid.txt')

INIT_TRAIN_SIZE = 500
MAX_TRAIN_SIZE = 1000
STEP = 100

#####################################################################
BERT_MODEL_TYPE = 'roberta'  # bert, roberta, xlm, ...
BERT_MODEL_NAME = 'roberta-base'
BERT_ARGS = ClassificationArgs(
    model_type=BERT_MODEL_TYPE,
    # overwrite_output_dir=True
)
BERT_OUTPUT = 'bert-output'
