import pandas as pd
import os

folder_path = 'annotations\clean'
uploaded = os.listdir(folder_path)

li = []

for fn in uploaded:
    print(f'files:\n{fn}')
    df = pd.read_csv(os.path.join(folder_path, fn), encoding='UTF-8', index_col=None)
    df.drop_duplicates(keep='first', inplace=True)
    li.append(df)

data = pd.concat(li)
data.drop('Unnamed: 0', inplace=True, axis=1)
data.drop('Story', inplace=True, axis=1)
data.drop_duplicates(keep='first', inplace=True)
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

data.to_csv('descriptive_dataset.csv', sep=';')
df = pd.read_csv('descriptive_dataset.csv', sep=';')

print('\n', df.head())
