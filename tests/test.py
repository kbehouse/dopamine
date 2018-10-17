import numpy as np

x = np.arange(10)
print(x)
print(np.roll(x, 2))
print(np.roll(x, -1))
print(np.roll(x, -3))
ary = np.roll(x, -3)
ary[-3:] = [99,98,88]

print(ary)