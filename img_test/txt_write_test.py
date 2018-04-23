import os
new_txt='444\n'


txt_name='E:/COCO/train/test.txt'

if os.path.exists(txt_name):
    exist_txt = open(txt_name)
    old_data = exist_txt.read()
    new_txt=old_data+new_txt
    exist_txt.close()
    write_txt = open(txt_name, 'w')
    write_txt.write(new_txt)
    write_txt.close()
else:
    write_txt = open(txt_name, 'w')
    write_txt.write(new_txt)
    write_txt.close()
