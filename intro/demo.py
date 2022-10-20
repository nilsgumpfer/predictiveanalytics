import numpy as np


def loops(a1, e, b, c, x, d=5.0):
    for element in x:
        print(element)

    for i in range(len(x)):
        print(x[i])


def run():
    s = "Hello world"
    c = 'X'
    mynumber = 123
    f = 1.23

    x = [s, c, mynumber, f, 5, 'hello', 4.5]
    a = np.array(x)

    loops(a1=s, b=c, c=f, e=1, x=x)

def run_mydict():
    d = {'A': ['Alfred', 'Anne'], 'Z': ['Zonk', 'Zacharias']}
    print(d['Z'])

    # d = {5: 'Hello', 'x': 6, 3.7: ['Python', 'Version']}

    # print(d.keys())
    # print(d.values())

    for k, v, g in zip(d.keys(), d.values()):
        print(k, v)


# run()
# loops(['4', 4, 5, 6.7])
# run_mydict()
loops()