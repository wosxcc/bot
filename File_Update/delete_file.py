import  os



count=0


file_path=  'E:/BOT_Person/trainb'

file_path = 'E:/dectect/dectect/face68'
file_path = 'E:/BOT_Person/trainb480'
##'F:/YOLO训练数据/person/trainc'

for file_name in os.listdir(file_path):
    if file_name[-4:]=='.txt':
        if file_name[:-4]+'.jpg' not in os.listdir(file_path):
            os.remove(file_path+'/'+file_name)
            print(file_name)

    if file_name[-4:]=='.jpg':
        if file_name[:-4]+'.txt' not in os.listdir(file_path):
            print(file_name)
            os.remove(file_path+'/'+file_name)
