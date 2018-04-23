import  os




count=0
test_txt = open('test.txt', 'w')
train_txt = open('train.txt', 'w')
for file_name in os.listdir('E:/BOT_train/train'):
    if file_name[-4:]=='.jpg':
        if count <14000:
            train_txt.write( 'D:/BOT_train/train/'+file_name + '\n')
        else:
            test_txt.write('D:/BOT_train/train/' + file_name + '\n')
        count += 1
        print(file_name)
test_txt.close()
train_txt.close()