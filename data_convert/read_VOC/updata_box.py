import  os
import numpy as np



paths='E:/BOT_Person/train'

for file in os.listdir(paths):
    if file[-4:]=='.txt':
        new_box=''
        new_txt =open(paths+'/'+file)
        old_data = new_txt.read()
        for bbox in old_data.split('\n'):
            box=bbox.split(' ')
            if len(box)==5:
                new_box+=box[0]+' '+box[1]+' '+box[2]+' '+str(float(box[3])*2)+' '+str(float(box[4])*2)+'\n'
        new_txt.close()
        write_txt = open(paths+'/'+file, 'w')
        write_txt.write(new_box)
        write_txt.close()
