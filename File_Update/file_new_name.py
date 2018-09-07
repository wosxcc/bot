import os

path = 'E:/Desk_why'
count=1
for file in os.listdir(path):
    print(file)
    print('E:/fujian/'+str(count)+'.jpg')
    os.rename('E:/Desk_why/'+file,'E:/Desk_why/'+str(count)+'.jpg')
    count+=1