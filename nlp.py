import pandas as pd

dataframe = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t',quoting = 3)
print dataframe.head()
