# NLP_2022L


Project structure:

root/
    / Data /
    |      /sourceData        - unmodified data from internet
    |      /preprecessedData  - preprocessed data
    |
    / docs /                  - documentation
    |
    / models /                - models, weights, configuration

## How to run

1) download data - use script in {project}/data/getDataSets.py
2) process data (generate csv's) run {project}/data/preprocessData.py it will generate csv files in {project}/data/preprocessedData/conll2003/
3) Run train script {project}/train.py