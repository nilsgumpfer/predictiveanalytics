import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv('../data/airline-passengers.csv')
values = df['Passengers'].values
model = ARIMA(values[:50], order=(1, 1, 1), seasonal_order=(1, 1, 1, 2))
res = model.fit()
p = res.predict(start=0, end=144)
plt.plot(p)
plt.plot(values)
plt.show()
