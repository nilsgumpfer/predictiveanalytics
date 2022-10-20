import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Load data from CSV, create dataframe
df = pd.read_csv('../data/airline-passengers.csv')

# Extract passenger values from dataframe
values = df['Passengers'].values

# Set train set size to 80
train_size = 80

# Split into train and test data
values_train = values[:train_size]  # use first 80 values
values_test = values[train_size:]  # extract values from index 80 up to end

print('Train:', len(values_train))
print('Test:', len(values_test))

# Plot training data
plt.figure(figsize=(15, 10))
plt.plot(values_train, label='training data')
plt.plot(np.arange(start=train_size, stop=len(values)), values_test, label='validation data')

# Train different models
for x in [0, 1, 2]:
    model = ARIMA(values_train, order=(1, x, 1), seasonal_order=(1, 1, 2, 12))
    res = model.fit()
    p_all = res.predict(start=0, end=len(values))
    p_test = res.predict(start=train_size+1, end=len(values))
    mse = mean_squared_error(p_test, values_test)
    plt.plot(p_all, '--', label='model {} (MSE={:.2f})'.format(x, mse))

plt.legend()
plt.show()
