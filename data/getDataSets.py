import os
import tarfile
import zipfile

import requests


def downloadGeniaDataset():
    urls = [
        "http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Relation/GENIA_relation_annotation_training_data.tar.gz",
        "http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Relation/GENIA_relation_annotation_development_data.tar.gz",
        "http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Relation/GENIA_relation_annotation_test_data.tar.gz"]

    print('Downloading GENIA dataset, urls: '+str(len(urls)))
    geniaPath = os.path.join(os.getcwd() + '/sourceData/genia')

    if not os.path.exists(geniaPath):
        os.mkdir(geniaPath)

    for url in urls:
        datasetName = url.split('/')[-1]
        response = requests.get(url)
        open(os.path.join(geniaPath, datasetName), "wb").write(response.content)
        archive = tarfile.open(os.path.join(geniaPath, datasetName))
        archive.extractall(geniaPath)
        os.remove(os.path.join(geniaPath, datasetName))


def downloadBlurbDataset():
    urls = ["https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip",
            "https://biocreative.bioinformatics.udel.edu/media/store/files/2015/BC5CDR-converter-0.0.2.zip"]

    print('Downloading BLURB dataset, urls: '+str(len(urls)))
    blurbPath = os.path.join(os.getcwd() + '/sourceData/blurb')

    if not os.path.exists(blurbPath):
        os.mkdir(blurbPath)

    for url in urls:
        datasetName = url.split('/')[-1]
        response = requests.get(url)
        open(os.path.join(blurbPath, datasetName), "wb").write(response.content)
        archive = zipfile.ZipFile(os.path.join(blurbPath, datasetName))
        archive.extractall(blurbPath)


def downloadConllDataset():
    urls = ["https://data.deepai.org/conll2003.zip"]
            # "http://www.cnts.ua.ac.be/conll2003/eng.raw.tar",
            # "http://www.cnts.ua.ac.be/conll2003/deu.raw.tar"]

    print('Downloading CoNLL 2003 dataset, urls: '+str(len(urls)))
    conllPath = os.path.join(os.getcwd() + '/sourceData/conll2003')

    if not os.path.exists(conllPath):
        os.mkdir(conllPath)

    for url in urls[1:]:
        datasetName = url.split('/')[-1]
        response = requests.get(url)
        open(os.path.join(conllPath, datasetName), "wb").write(response.content)
        archive = zipfile.ZipFile(os.path.join(conllPath, datasetName))
        archive.extractall(conllPath)

    os.remove(os.path.join(conllPath, 'conll2003.zip'))


if __name__ == "__main__":
    # downloadGeniaDataset()
    # downloadBlurbDataset()
    downloadConllDataset()
