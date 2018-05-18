import  os




count=0
test_txt = open('I:/BOT_Car/train.txt', 'w')
# train_txt = open('E:/COCO/train.txt', 'w')
for file_name in os.listdir('I:/BOT_Car/train'):
    if file_name[-4:]=='.jpg':
        # if count <14000:
        # train_txt.write( 'E:/COCO/train/'+file_name + '\n')
        # else:
        test_txt.write('I:/BOT_Car/train/' + file_name + '\n')
        count += 1
        print(file_name)
test_txt.close()
# train_txt.close()