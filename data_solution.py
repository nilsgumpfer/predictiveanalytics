import pandas as pd
from matplotlib import pyplot as plt

# Load data
df = pd.read_csv('data/airline-passengers.csv', header=0, index_col=0, parse_dates=True)

# Print data
print(df)

# Plot data
plt.plot(df)

# Show plot
plt.show()
