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
    df = initial_preprocess(df)
    print(df.head)
    df = df[['text', 'pos']]
    df.to_csv(output)


def initial_preprocess(df: pd.DataFrame):
    df['text'] = df['text'].astype('string')
    # make everything uppercase
    df['text'] = df['text'].str.upper()
    # remove punctuation
    for i, text in enumerate(df['pos']):
        df.loc[i, 'text'] = re.sub(r'[^\w\s#]', '', text)  # DO NOT REMOVE HASH
    return df

def simpleCSV(input, output):
    df = pd.read_csv(filepath_or_buffer=input, sep=' ')
    # df['-X-'] = df['-X-'].rank(method='dense', ascending=False).astype(int)
    # df['-X-.1'] = df['-X-.1'].rank(method='dense', ascending=False).astype(int)
    # df['O'] = df['O'].rank(method='dense', ascending=False).astype(int)-1
    df = df.drop(columns=['O'])
    df.rename(columns={'-X-': 'label1', '-X-.1': 'label2', '-DOCSTART-': 'text'}, inplace=True)
    df['label1'] = pd.factorize(df['label1'])[0]
    df['label2'] = pd.factorize(df['label2'])[0]
    df.to_csv(output)

if __name__ == '__main__':
#option 1
    # conllCleanup(input_file, output_file)
#option2
    for type in test_or_train:
        input_file = f'./sourceData/conll2003/{type}.txt'
        output_file = f'./preprocessedData/conll2003/preprocessed_{type}.csv'
        simpleCSV(input_file, output_file)
