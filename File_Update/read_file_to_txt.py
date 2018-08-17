import  os



count=0

file_path='E:/BOT_Person/trainb480'
file_path='E:/BOT_Car/train'

test_txt = open(file_path+'.txt', 'w')
# train_txt = open('E:/COCO/train.txt', 'w')
for file_name in os.listdir(file_path):
    if file_name[-4:]=='.jpg':
        test_txt.write(file_path+'/' + file_name + '\n')
        count += 1
        print(file_name)
test_txt.close()

print('累计数量：',count)
# train_txt.close()