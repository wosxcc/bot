import  os

path_file = 'D:/pproject/bot/my_tf/gan_img'

counts=1
for filess in  os.listdir(path_file):
    os.rename(path_file+'/'+filess,'D:/pproject/bot/my_tf/gangan/'+str(counts)+'.jpg')
    counts +=1