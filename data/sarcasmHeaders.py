import json
import pandas as pd

data = []
with open('./sourceData/Sarcasm_Headlines_Dataset_v2.json') as f:
    for line in f:
        data.append(json.loads(line))

print(data[0]['is_sarcastic'])

f1 = pd.DataFrame(data)
f1 = f1.drop(columns=['article_link'])
f1.rename(columns={'is_sarcastic': 'sarcasm_label', 'headline': 'text'}, inplace=True)
f1.to_csv('headlines.csv')
