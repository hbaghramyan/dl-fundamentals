# 1 Loading the dataset into DataFrames

# pip install datasets

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from local_dataset_utilities import download_dataset, load_dataset_into_to_dataframe, partition_dataset

download_dataset()

df = load_dataset_into_to_dataframe()
partition_dataset(df)

df_train = pd.read_csv("train.csv")
df_val = pd.read_csv("val.csv")
df_test = pd.read_csv("test.csv")

## 2 Bag-of-Words Model
cv = CountVectorizer(lowercase=True, max_features=10_000, stop_words="english")

cv.fit(df_train["text"])

X_train = cv.transform(df_train["text"])
X_val = cv.transform(df_val["text"])
X_test = cv.transform(df_test["text"])