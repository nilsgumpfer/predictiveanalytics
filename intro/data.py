import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('../data/airline-passengers.csv')

print(df)

months = df['Month'].values
passengers = df['Passengers'].values

print(months)
print(passengers)

plt.plot(months, passengers)
plt.xticks([months[0], months[-1]])
plt.show()
