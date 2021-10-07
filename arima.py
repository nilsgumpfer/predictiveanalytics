import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load data
df = pd.read_csv('data/airline-passengers.csv', header=0, index_col=0, parse_dates=True)

# Extract column 'Passengers'
values = df['Passengers'].values

# Add line to plot containing passenger values
plt.plot(values, label='Passengers')

# Prepare hyperparameters for SARIMA model
P, D, Q, s = 2, 2, 1, 4

# Initialize model
model = ARIMA(values, seasonal_order=(P, D, Q, s))

# Train model
results = model.fit()

# Predict values in range
predicted_values = results.predict(start=1, end=300)

# Add line to plot containing predicted passenger values
plt.plot(predicted_values, label=str((P, D, Q, s)))

# Add legend to plot
plt.legend()

# Show plot
plt.show()
