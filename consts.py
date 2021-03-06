import os
from simpletransformers.classification import MultiLabelClassificationModel

ENCODING_IOB = "IOB"
ENCODING_IO = "IO"
ENCODING_BIO = "BIO"

DATA_DIR = 'data/sourceData'
CONL_TRAIN_DATASET = os.path.join(DATA_DIR, 'conll2003', 'train.txt')
CONL_TEST_DATASET = os.path.join(DATA_DIR, 'conll2003', 'test.txt')
CONL_VALID_DATASET = os.path.join(DATA_DIR, 'conll2003', 'valid.txt')

DATA_PROCESSED_DIR = 'data/preprocessedData'

CONL_PREPROC_TRAIN = os.path.join(DATA_PROCESSED_DIR, 'conll2003', 'preprocessed_train.csv')
CONL_PREPROC_TEST = os.path.join(DATA_PROCESSED_DIR, 'conll2003', 'preprocessed_test.csv')
CONL_PREPROC_VALID = os.path.join(DATA_PROCESSED_DIR, 'conll2003', 'preprocessed_valid.csv')

CONL_PREPROC_TRAIN_IOB = os.path.join(DATA_PROCESSED_DIR, 'conll2003', 'preprocessed_train_IOB.csv')
CONL_PREPROC_TEST_IOB = os.path.join(DATA_PROCESSED_DIR, 'conll2003', 'preprocessed_test_IOB.csv')
CONL_PREPROC_VALID_IOB = os.path.join(DATA_PROCESSED_DIR, 'conll2003', 'preprocessed_valid_IOB.csv')

CONL_PREPROC_TRAIN_IO = os.path.join(DATA_PROCESSED_DIR, 'conll2003', 'preprocessed_train_IO.csv')
CONL_PREPROC_TEST_IO = os.path.join(DATA_PROCESSED_DIR, 'conll2003', 'preprocessed_test_IO.csv')
CONL_PREPROC_VALID_IO = os.path.join(DATA_PROCESSED_DIR, 'conll2003', 'preprocessed_valid_IO.csv')

CONL_PREPROC_TRAIN_BIO = os.path.join(DATA_PROCESSED_DIR, 'conll2003', 'preprocessed_train_BIO.csv')
CONL_PREPROC_TEST_BIO = os.path.join(DATA_PROCESSED_DIR, 'conll2003', 'preprocessed_test_BIO.csv')
CONL_PREPROC_VALID_BIO = os.path.join(DATA_PROCESSED_DIR, 'conll2003', 'preprocessed_valid_BIO.csv')

#####################################################################
# SELECTED OPTIONS FOR DATASET
SELECTED_ENCODING = ENCODING_IOB
SELECTED_TRAIN_DATASET = CONL_TRAIN_DATASET
SELECTED_TEST_DATASET = CONL_TRAIN_DATASET
SELECTED_VALID_DATASET = CONL_TRAIN_DATASET

#####################################################################
# SELECTED OPTIONS FOR MODEL
BERT_MODEL_TYPE = 'roberta'  # bert, roberta, xlm, ...
BERT_MODEL_NAME = 'roberta-base'
BERT_OUTPUT = 'bert-output'

#####################################################################
# SELECTED OPTIONS FOR LEARNING
# INIT_TRAIN_SIZE = 204565
# MAX_TRAIN_SIZE = 204565
INIT_TRAIN_SIZE = 10
MAX_TRAIN_SIZE = 10
STEP = 100
EPOCHS = 10