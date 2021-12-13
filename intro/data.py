import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('../data/airline-passengers.csv', index_col=0, parse_dates=True)
# print(df)
# print(type(df.index[0]))
# print(list(df['Passengers']))
# print(df['Passengers'].values)
# print(np.array(df['Passengers']))

plt.plot(df)
plt.show()

# values = df['Passengers'].values
# plt.plot(values, 'r--')
# plt.show()