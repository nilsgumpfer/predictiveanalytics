import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv('../data/airline-passengers.csv')
# values = df['Passengers'].values
# model = ARIMA(values, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
# res = model.fit()
# p = res.predict()
