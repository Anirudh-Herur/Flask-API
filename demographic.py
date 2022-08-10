import pandas as pd
import numpy as np

df2 = pd.read_csv('articles.csv', encoding='utf8')
df2 = df2.sort_values(['total_events'], ascending=False)
output = df2[["url", "title", "text", "lang",
                     "total_events"]].head(20).values.tolist()
