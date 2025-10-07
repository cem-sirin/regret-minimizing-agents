import numpy as np
from rich import print

v = np.array([0.5])
print(v.shape)
print(v.shape == (1,))
print(len(v.shape))


v = np.array(v)
print(v.shape)
print(v.shape == (1,))
print(len(v.shape))


v_list = np.random.uniform(0, 1, 3)
print(v_list)
print(type(v_list))
