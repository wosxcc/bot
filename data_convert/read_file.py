import  os




count=0
test_txt = open('E:/COCO/test.txt', 'w')
# train_txt = open('E:/COCO/train.txt', 'w')
for file_name in os.listdir('E:/COCO/test'):
    if file_name[-4:]=='.jpg':
        # if count <14000:
        # train_txt.write( 'E:/COCO/train/'+file_name + '\n')
        # else:
        test_txt.write('E:/COCO/test/' + file_name + '\n')
        count += 1
        print(file_name)
test_txt.close()
# train_txt.close()