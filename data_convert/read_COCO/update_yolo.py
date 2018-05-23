import os


paths='E:/xbot/crete_data/train'
for file in os.listdir(paths):
    if file[-4:] == '.txt':
        new_box = ''
        new_txt = open(paths + '/' + file)
        old_data = new_txt.read()
        new_txt.close()
        for bbox in old_data.split('\n'):
            if len(bbox)>2:
                if bbox[0]=='0':
                    new_box+='1'+bbox[1:]+'\n'
                else:
                    new_box +=bbox+'\n'
        write_txt = open(paths + '/' + file, 'w')
        write_txt.write(new_box)
        write_txt.close()

