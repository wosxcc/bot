import os
paths =['D:/bot/my_tf/img/train/cat','D:/bot/my_tf/img/train/dog']

out_txt = ''
i=0
for apath in paths:
    for file_name in os.listdir(apath):
        out_txt+= apath+'/'+file_name+ ' '+str(i)+ '\n'
        print(apath+'/'+file_name+ ' '+str(i))
    i+=1
txtout = open('dogcat.txt','w')
txtout.write(out_txt)
txtout.close()