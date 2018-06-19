#-*- coding:utf-8 -*-
#coding=gbk
import re
xx =  b'\\u5929\\u54ea\\uff0c\\u4f60\\u5403\\u8fc7\\u4e86\\u5417\\uff1f'
# xx = b'\u5929\u54ea\uff0c\u4f60\u5403\u8fc7\u4e86\u5417\uff1f'
print (xx.encode('utf-8'))
print(u''+xx)
#
# xxc = xx.split('\\')
# strcc =u''
# for i in  xxc:
#        print(u'\'+i)
#        strcc+=i+r'\u'
#
# print(strcc)
# print(strcc.encode("utf-8"))