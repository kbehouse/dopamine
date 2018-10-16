import numpy as np
from numpy import linalg as LA

a = np.arange(9) - 4
b = a.reshape((3, 3))


sum = 0
for i in a:
    j = i
    print('j=', j)
    sum+= j*j

print('sum = ', sum)
print('np.sqrt(sum) = ', np.sqrt(sum) )

print(LA.norm(a) )
# 7.745966692414834
print(LA.norm(b))
# 7.745966692414834