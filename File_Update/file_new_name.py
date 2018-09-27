import os

path = 'E:/2018-09-12'

path = '../crete_data/yy22/'
count=620
for file in os.listdir(path):
    print(file)
    print('E:/fujian/'+str(count)+'.jpg')
    if file[-3:] =='jpg':
        print(path + '/' + file,path + '/' + str(count) + '.jpg')

        print(path + '/' + file[:-3]+'txt',path + '/' + str(count) + '.txt')
        os.rename(path + '/' + file,path + '/' + str(count) + '.jpg')

        os.rename(path + '/' + file[:-3]+'txt',path + '/' + str(count) + '.txt')
    count+=1