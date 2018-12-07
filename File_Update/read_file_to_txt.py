import os
count=0
file_path='E:/Desk_Set/train'
test_txt = open(file_path+'.txt', 'w')     # 需要操作的文件夹
# train_txt = open('E:/COCO/train.txt', 'w')
for file_name in os.listdir(file_path):
    if file_name[-4:]=='.jpg':
        test_txt.write(file_path+'/' + file_name + '\n')
        count += 1
        print(file_name)
test_txt.close()

print('累计数量：',count)
# train_txt.close()