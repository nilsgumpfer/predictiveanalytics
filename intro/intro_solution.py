import numpy as np

x = [1, 2, 3, 4, 5]
y = np.array([x * 2])
d = {1: 0, 'x': 7, 7: 5.0, 100: "text"}

print(type(x), type(y))

for i in range(100):
    print(i)

for v in x:
    print(v)


for k in d:
    print(k)


for k in d:
    print(d[k])