import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def aufgabe1():
    mylist = ['a', '1', 1, 2.0]
    myarray = np.array(mylist)

    for i, v in enumerate(myarray):
        print(i, v, type(v))


def aufgabe2():
    np.random.seed(1)
    myarray = np.random.random(20)
    print(myarray.min(), myarray.max(), myarray.mean(), myarray.std())
    arrayA = myarray[:5]  # 0 to 5
    arrayB = myarray[5:]  # 5 to end

    print(len(arrayA), len(arrayB))


def aufgabe3():
    mydict = {'name': ['Gumpfer', 'Doe', 'Mustermann'],
              'vorname': ['Nils', 'John', 'Max'],
              'alter': [29, 66, 90]}

    df = pd.DataFrame(mydict)
    df = df.sort_values('alter', ascending=False)

    print(df)

    df.to_csv('../data/mydf.csv')
    df2 = pd.read_csv('../data/mydf.csv', index_col=0)

    print(df2)


def aufgabe4():
    val1 = np.arange(start=0, stop=100, step=2)
    val2 = val1 ** 2
    val3 = val1 ** 3
    print(val1)
    plt.plot(val1, label='linear', color='red')
    plt.plot(val2, label='quadratic', color='green')
    plt.plot(val3, label='cubic', color='blue')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('MyPlot')
    # plt.show()
    plt.savefig('../data/plots/myplot.pdf')

# aufgabe1()
# aufgabe2()
# aufgabe3()
aufgabe4()
