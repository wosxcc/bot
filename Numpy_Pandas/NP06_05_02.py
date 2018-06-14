###数组属性
import numpy as np


a =np.array([[1,2,3],[4,5,6]])
a.shape=(3,2)       # #数组类型转换
print(a)
b =a.reshape(3,2) # #数组类型转换
print(b)

a=np.arange(10,100,5,dtype=np.float32)          ##建立一个从10开始到100步幅为5类型为float32的数组
print(a)

b=a.reshape(3,3,2)
print(b)
print(b.itemsize)  ##输出b类型的字节长度


#############flags属性
# 1.	C_CONTIGUOUS (C) 数组位于单一的、C 风格的连续区段内
# 2.	F_CONTIGUOUS (F) 数组位于单一的、Fortran 风格的连续区段内
# 3.	OWNDATA (O) 数组的内存从其它对象处借用
# 4.	WRITEABLE (W) 数据区域可写入。 将它设置为flase会锁定数据，使其只读
# 5.	ALIGNED (A) 数据和任何元素会为硬件适当对齐
# 6.	UPDATEIFCOPY (U) 这个数组是另一数组的副本。当这个数组释放时，源数组会由这个数组中的元素更新

x=np.array([1,2,3,4,5])
print(a.flags)

xx=np.empty([2,3],dtype=int) ###定义未初始化的数组
print('xx',xx)

##numpy.zeros(shape, dtype = float, order = 'C')

x=np.zeros((2,2),dtype=[('x','i4'),('y','i4')])
print(x)

x= np.ones(5)
print(x.dtype)



# 将列表转化为ndarray
x=[3,4,5]
a= np.asarray(x,dtype= float)
print(a)

x=(123,456,789)
a = np.asarray(x)
print(a)

x=[(11,22.3),(22,33.445,55)]
a= np.asarray(x)
print(a)



##np.frombuffer(buffer, dtype = float, count = -1, offset = 0)


# 1.	buffer 任何暴露缓冲区借口的对象
# 2.	dtype 返回数组的数据类型，默认为float
# 3.	count 需要读取的数据数量，默认为-1，读取所有数据
# 4.	offset 需要读取的起始位置，默认为0
s= b'wos xcc'
a = np.frombuffer(s, dtype='S1')
print(s)
print(a)


list =range(5)
print(list)
it= iter(list)
print(it)
x =np.fromiter(it,dtype=float)
print(x)

##numpy.arange(start, stop, step, dtype)
X= np.arange(10,20,2,dtype=float)   ##注意前闭后开
print(X)


