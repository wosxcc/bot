# def num():
#     print([x*3 for x in range(6)])
#     return [lambda y:x*y for x in range(6)]
# print([m(3) for m in num()])
# # [15, 15, 15, 15, 15, 15]




# func = lambda y:3*y
#
# print(func(7))
# for i in range(7):
#     print(func(i))



def sumx():
    return  [lambda y:y*x for x in range(9)]


print([m(3) for m in sumx()])

for i in sumx():
    print(sumx(i))

##变量值互换
a = 1
b = 2
(a, b) = (b, a)
print(a,b)