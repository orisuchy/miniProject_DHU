import pandas as pd


df = pd.read_csv("annotations\CatmaTry.csv", sep=';', header=None, usecols=[2, 4, 10],
                 names=["Story", "Sentence", "Type"])
df['Type'] = df['Type'].str.replace('/', '')

# Init dict with 0's
new_data = {"Sentence": df['Sentence'],
            "Descriptive": [0] * len(df),
            "NotDescriptive": [0] * len(df),
            "MightDescriptive": [0] * len(df)}

# set 1 according to type
for i, t in enumerate(df['Type']):
    new_data[t][i] = 1

new_df = pd.DataFrame(new_data)
new_df.to_csv("goodCSV.csv")
