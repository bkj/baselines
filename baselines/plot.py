import json
import pandas as pd
import numpy as np

from rsub import *
import seaborn as sns
from matplotlib import pyplot as plt

inpath = 'monitor.json'
gen = map(json.loads, open(inpath))
next(gen)

df = pd.DataFrame(list(gen))
df['z'] = np.floor(df.index / 10)
df = df[:-1]

x = np.vstack(df.groupby('z').r.apply(list))
_ = sns.tsplot(data=x.T)
show_plot()