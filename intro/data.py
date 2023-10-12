import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('../data/airline-passengers.csv', index_col=0, parse_dates=True)

values = df['Passengers'].values
plt.plot(values, 'r--')
plt.show()