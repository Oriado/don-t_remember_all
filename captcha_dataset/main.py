import torch
import numpy as np

data = [[1,2], [3,4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(x_np.shape)
print(x_data.shape)

t1 = torch.cat((x_np, x_data), dim=0) # dim - по чому ми об'єднуємо
print(t1.shape)
print(t1)

y1 = x_np @ x_data.T

print(y1.shape)
print(y1)

y2 = x_np * x_data
print(y2.shape)
print(y2)

y3 = x_np - x_data
print(y3.shape)
print(y3)

y4 = x_np + x_data
print(y4.shape)
print(y4)
