import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv('data/airline-passengers.csv', header=0)
