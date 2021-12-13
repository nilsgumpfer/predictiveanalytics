import numpy as np

# Array definition
# x = np.array([[1, 2, 3], [1, 2, 3], [3, 4, 6]])

# Array print
# print(x.shape)

# ALT+ENTER -> context

# d = {1: "A", 2: "B", 3: "C", 4: [1, 2, 3], 5: {'A': 1, 'C': 3}}

# for k in d:
#     print(d[k])
#
# for v in d.values():
#     print(v)

# for k, v in d.items():
#     print(k, v)

# for k, v in zip(d.keys(), d.values()):
#     print(k, v)

# print(list(zip(d.keys(), d.values())))

x = np.array([1, 2, 3], dtype=int)
y = np.array(x, dtype=float)
y = y * 2

print(y)