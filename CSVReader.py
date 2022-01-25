import pandas as pd

df = pd.read_csv("annotations\CatmaTry.csv", sep=';', header=None, usecols=[2, 4, 10], names=["Story", "Sentence", "Type"])
df['Type'] = df['Type'].str.replace('/', '')

descriptive_vect = []
Notdescriptive_vect = []
Mightdescriptive_vect = []

for t in df['Type']:
    if t == 'Descriptive':
        descriptive_vect.append(1)
        Notdescriptive_vect.append(0)
        Mightdescriptive_vect.append(0)
    elif t == 'NotDescriptive':
        descriptive_vect.append(0)
        Notdescriptive_vect.append(1)
        Mightdescriptive_vect.append(0)
    elif t == 'MightDescriptive':
        descriptive_vect.append(0)
        Notdescriptive_vect.append(0)
        Mightdescriptive_vect.append(1)

new_data = {"Sentence": df['Sentence'],
            "Descriptive": descriptive_vect,
            "NotDescriptive": Notdescriptive_vect,
            "MightDescriptive": Mightdescriptive_vect}

new_df = pd.DataFrame(new_data)

new_df.to_csv("goodCSV.csv")
