import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv('data/airline-passengers.csv')
values = df['Passengers'].values

plt.figure(figsize=(15, 10))
plt.plot(values, label='training data')

for x in [0, 1, 2]:
    model = ARIMA(values, order=(1, 1, x), seasonal_order=(1, 1, 1, 12))
    res = model.fit()
    p = res.predict(start=0, end=len(values) + 100)
    plt.plot(p, label=str(x))

plt.legend()
plt.show()