import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


# Data loading
df = pd.read_csv('../data/airline-passengers.csv')
values = df['Passengers'].values

# Model definition
model = ARIMA(values[:50], order=(1, 2, 2), seasonal_order=(1, 1, 1, 12))

# Model training
res = model.fit()

# Model usage
p = res.predict(start=0, end=144)

# Plot
plt.plot(p)
plt.plot(values)
plt.show()
