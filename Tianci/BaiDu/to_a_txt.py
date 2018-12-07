import os


img_path ='E:/Model/deeplab/Database/SegmentationClass'

txt_path = 'E:/Model/deeplab/Database/ImageSets/Segmentation/'


countt=0

train_txt = ''
val_txt = ''
text_txt = ''
for fimg in os.listdir(img_path):
    if countt%29==1:
        val_txt+=fimg[:-4]+'\n'
    elif countt%49==1:
        text_txt+=fimg[:-4]+'\n'
    else:
        train_txt += fimg[:-4]+'\n'


    countt+=1


train_save= open(txt_path+'train.txt','w')
train_save.write(train_txt)
train_save.close()

val_save= open(txt_path+'val.txt','w')
val_save.write(val_txt)
val_save.close()

text_save= open(txt_path+'trainval.txt','w')
text_save.write(text_txt)
text_save.close()


print(text_txt)