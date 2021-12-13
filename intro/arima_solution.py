import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load data
df = pd.read_csv('../data/airline-passengers.csv', header=0)

# Extract column 'Passengers'
values = df['Passengers'].values

# Add line to plot containing passenger values
plt.plot(values, label='Passengers')

# Initialize models
model_1 = ARIMA(values, order=(2, 2, 2), seasonal_order=(1, 1, 1, 12))
model_2 = ARIMA(values, order=(2, 2, 2), seasonal_order=(0, 1, 0, 12))

# Train models
results_1 = model_1.fit()
results_2 = model_2.fit()

# Predict values in range
predicted_values_1 = results_1.predict(start=0, end=250)
predicted_values_2 = results_2.predict(start=0, end=250)

# Add line to plot containing predicted passenger values
plt.plot(predicted_values_1, label='Model 1')
plt.plot(predicted_values_2, label='Model 2')

# Add legend to plot
plt.legend()

# Show plot
plt.show()
