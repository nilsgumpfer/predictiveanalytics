import numpy as np

def numpy_examples():
    mylist = [[1, 2, 3],
              [1, 2, 3]]
    x = np.array(mylist, dtype=float)
    print(type(mylist), type(x), x, mylist)
    print(np.mean(x))
    print(np.mean(x, axis=0))
    print(np.mean(x, axis=1))
    print(x[:,2])

def main():
    # print("Hello!", 123, [123, "a'a", 'a"a'])
    a = "Hello!"
    b = float(123)
    c = [123, "a'a", 'a"a']
    g = ['A', 'B', 'C']
    b = int(b)
    b = str(b)
    print(a, b, c)
    print(type(a), type(b), type(c))

    d = {"a": 1, "b": 2, "c": c, 3.0: "three"}
    print(d)

    if type(a) == str:
        print(a, "is a string")

    print("Length of c:", len(c))

    for i in range(len(c)):
        print(c[i])

    print("--------")

    for element in c:
        print(element)

    print("--------")

    for k in d:
        print(k, d[k])

    counter = 0
    while counter < 10:
        print(counter)

        # if counter == 5:
        #     continue

        myfunction(counter, exponent=2)
        myfunction(v=counter, exponent=3)

        counter += 1

    print("--------")

    for i, x in enumerate(c):
        print(i, g[i], x)

    for x, y in zip(c, g):
        print(x, y)

    for i, (x, y) in enumerate(zip(c, g)):
        print(i, x, y)


def myfunction(v, exponent=2):
    def greeting(name_to_greet):
        print("Hello {:.4f}: {}!".format(v, name_to_greet))
        # print("Hello: " + name_to_greet + "!")

    greeting("Nils")
    print("Function call!", v**exponent)


if __name__ == '__main__':
    # main()
    numpy_examples()