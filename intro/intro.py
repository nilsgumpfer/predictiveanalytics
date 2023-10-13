import numpy as np

# x = {'train': [5, 3, 2, 5, 7], 'val': [2, 3, 5], 'test': [1, 2, 3]}
#
# print(x['train'], type(x['train']))
# print(x['val'], type(x['val']))
# print(x['test'], type(x['test']))
#
# print(['ABC', 1, 2.0, {'A': 1}], type(['ABC', 1, 2.0, {'A': 1}]))
#
# print(np.array([1, 2, 3]), type(np.array([1, 2, 3])))
#
# print(type('Hello'))
#
# x_s = str(x)
# print(x_s)
#
# print(x)
#
# y = np.array([1, 2, 3], dtype=float)
# print(y, type(y))
#
# print(y.min(), y.max())
#
# print(type(float(1.2)))
#
# for v in y:
#     print(v, type(v))

# mylist = ['a', '1', 1, 2.0, {'A': 1, 'B': 2}, [[1], [1, 2], {1.0: 2}]]
# mydict = {'A': 1, 'B': 2}
# inputs = ['Nils', 'Peter', 'Anne']
# outputs = np.array(['m', 'm', 'w'])

# for v in mylist:
#     print(v, type(v))

# for k in mydict:
#     print(k, mydict[k])

# d = {}
# for i, o in zip(inputs, outputs):
#     print(i, o)
#     d[i] = o
#
# print(d)

# for i, v in enumerate(mylist):
#     print(i+1, 'von', len(mylist), v)

# for i, v in enumerate(inputs):
#     print(v, outputs[i])



def myfunction(name, age, greeting='Hello'):
    print(greeting, name, '(', age, ') !')


def main():
    myfunction('Nils', 29, 'Hi')  # positional arguments
    myfunction(age=29, greeting='Goodbye', name='Nils')  # keyword arguments


main()
