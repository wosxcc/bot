import  os




count=0
test_txt = open('E:/BOT_Txt/train.txt', 'w')
# train_txt = open('E:/COCO/train.txt', 'w')
for file_name in os.listdir('E:/BOT_Txt/train'):
    if file_name[-4:]=='.jpg':
        # print (file_name[:-4]+'.txt')
        # if file_name[:-4]+'.txt' in os.listdir('E:/BOT_Person/train'):
        # if count <14000:
        # train_txt.write( 'E:/COCO/train/'+file_name + '\n')
        # else:
        test_txt.write('E:/BOT_Txt/train/' + file_name + '\n')
        count += 1
        print(file_name)
test_txt.close()
# train_txt.close()