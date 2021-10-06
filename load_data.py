from matplotlib import pyplot as plt
from pandas import read_csv
import numpy as np


def main():
    series = read_csv('data/airline-passengers.csv', index_col=0, parse_dates=True)
    values = np.array(series['Passengers'])
    # print(len(values))
    # plt.plot(values, label='values')
    # plt.plot(values[80:90], label='cutout')
    # plt.legend()
    # plt.show()
    X, Y = generate_training_data(values)


def generate_training_data(values, window_size=20, dist=5):
    inputs = []
    outputs = []

    for start in range(len(values) - (window_size + dist)):
        # print(start)
        x = values[start:start + window_size]
        y = values[start + dist:start + dist + window_size]
        print('x: {}:{}, y: {}:{}'.format(start, start + window_size, start + dist, start + dist + window_size))
        inputs.append(x)
        outputs.append(y)

    return inputs, outputs


if __name__ == '__main__':
    main()
