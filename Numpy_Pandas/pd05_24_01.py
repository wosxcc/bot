import  numpy as np

a = [1,2,3,4,5]
b=a
b[0]=222

print('b',b)
print('a',a)

c=np.zeros((5),np.int32)
c=a
c[0]=3333
print('c',c)
print('a',a)
d=np.zeros((5),np.int32)
for i in range(len(a)):
    d[i]=a[i]

d[0]=444
print('d',d)
print('a',a)


