import numpy as np
from matplotlib import pyplot as plt


data_x=np.random.randint(-500,500,size=(10000))
data_y=np.power(data_x,2)###np.power数组的次方
print(data_x)
print(data_y)

plt.scatter(data_x,data_y,c='r',s=5,)
plt.xlabel('x')
plt.ylabel('y')
plt.show()