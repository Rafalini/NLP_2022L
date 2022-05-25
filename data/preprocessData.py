import pandas as pd
import re
from sparknlp.training import CoNLL
from sparknlp import SparkSession
import sparknlp

test_or_train = ['train', 'test', 'valid']


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


def conllCleanup(input, output):
    df = getDataset(input)
    df = df.drop(columns=['document', 'label', 'sentence'])
    print(df.head)
    # df = df[['text', 'pos']]
    # df.to_csv(output)


def simpleCSV(input, output, encoding):
    df = pd.read_csv(filepath_or_buffer=input, sep=' ')
    # df['-X-'] = df['-X-'].rank(method='dense', ascending=False).astype(int)
    # df['-X-.1'] = df['-X-.1'].rank(method='dense', ascending=False).astype(int)
    # df['O'] = df['O'].rank(method='dense', ascending=False).astype(int)-1
    df = df.drop(columns=['-X-']).drop(columns=['-X-.1'])
    df.rename(columns={'O': 'label1', '-DOCSTART-': 'text'}, inplace=True)
    if encoding == 'IO':
        df['label1'] = IOB2IO(df['label1'])
    if encoding == 'BIO':
        df['label1'] = IOB2BIO(df['label1'])
    df['label1'] = pd.factorize(df['label1'])[0]
    df.to_csv(output)


def newCSV(input, output):
    df = pd.read_csv(filepath_or_buffer=input, sep=' ')
    df.rename(columns={'-DOCSTART-': 'text', '-X-': 'partofSpeach', '-X-.1': 'syntactic', 'O': 'entityTag'}, inplace=True)
    print(df.head)
    df.to_csv(output)


def IOB2IO(values):
    i = 0
    for value in values:
        if type(value) is not str:
            values[i] = ''
            i += 1
            continue
        values[i] = value.replace('B-', 'I-')
        i += 1
    return values


def IOB2BIO(values):
    i = 0
    prevO = True # if previous value wa O
    for value in values:
        if type(value) is not str:
            values[i] = ''
            i += 1
            continue
        if prevO:
            values[i] = value.replace('I-', 'B-')
        if value == 'O':
            prevO = True
        else :
            prevO = False
        i += 1
    return values


if __name__ == '__main__':
#option 1
    # newCSV('./sourceData/conll2003/test.txt', './preprocessedData/conll2003/preprocessed_test.csv')
    # newCSV('./sourceData/conll2003/train.txt', './preprocessedData/conll2003/preprocessed_train.csv')
    # newCSV('./sourceData/conll2003/valid.txt', './preprocessedData/conll2003/preprocessed_valid.csv')
#option2
    for encoding in ['IOB', 'IO', 'BIO']:
        for data_type in test_or_train:
            print('Processing: '+encoding+' '+data_type)
            input_file = f'./sourceData/conll2003/{data_type}.txt'
            output_file = f'./preprocessedData/conll2003/preprocessed_{data_type}_{encoding}.csv'
            simpleCSV(input_file, output_file, encoding)
