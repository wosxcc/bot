# -*- coding:utf-8 -*-
import numpy as np

aline= np.array([[1,2,3],[2,3,4]]) ##数组初始化
print('aline= np.array([[1,2,3],[2,3,4]]):',aline)

aline = np.array([1,2,3,4,5],ndmin=2)   #转化为二维数组  #ndmin=2 最小维度为2
print('aline = np.array([1,2,3,4,5],ndmin=2):',aline)

a =np.array([1,2,3],dtype = complex) ##类型为复数
print('np.array([1,2,3],dtype = complex):',a)


### np.dtype(object,align,copy)

## object 被转换为数据类型的对象
## align：如果为true，则向字段添加间隔，使其类似 C 的结构体。
##Copy : 生成dtype对象的新副本，如果为flase，结果是内建数据类型对象的引用。
dt = np.dtype(np.int32) ##定图类型
dt = np.dtype('i4')     #int8，int16，int32，int64 可替换为等价的字符串 'i1'，'i2'，'i4'，以及其他。
print(dt)
dt = np.dtype('>i4')
print(dt)

dt = np.dtype([('age',np.int8)])  #创建结构化数据类型
dt = np.dtype([('name',np.int8)])
a = np.array([(10,),(20.1,),(30.0,)],dtype=dt)  #dt的类型采用了
a=  np.array([(100,),(110.1,),(30.0,)],dtype=dt)
print('np.dtype([(age,np.int8)])',dt)
print('类型是：',a.dtype,'a是:',a)

print('内容是：',a['name'])


student=np.dtype([('name','S20'),('age','i1'),('marks','f4')])

a=np.array([('xcc',18,50.12345),('wll',21,87.361)],dtype=student)
print(a)


