import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load data
df = pd.read_csv('data/airline-passengers.csv', header=0, index_col=0, parse_dates=True)

# Extract column 'Passengers'
values = df['Passengers'].values

# Add line to plot containing passenger values
plt.plot(values, label='Passengers')

i = 0

for a in range(2):
    for b in range(2):
        for c in range(2):
            d, e, f = 1, 1, 1
            # Initialize models
            model = ARIMA(values, order=(a, b, c), seasonal_order=(d, e, f, 12))

            # Train models
            results = model.fit()

            # Predict values in range
            predicted_values = results.predict(start=0, end=250)

            # Add line to plot containing predicted passenger values
            plt.plot(predicted_values, label='Model {}'.format(i))
            i = i + 1
            print(i)

# Add legend to plot
plt.legend()

# Show plot
plt.show()
