import numpy as np
import os
import math
cccc ='29 26 24 21 38 36 33 31 9 7 4 1 19 17 14 12 10 14 15 17 2 4 5 8 29 32 34 36 20 22 24 27'
xcxc ='0 29 26 24 21 38 36 33 31 9' \
      ' 7 0 0 19 17 14 12 10 12 15 ' \
      '17 0 0 0 7 29 32 34 36 20 ' \
      '22 24 27 29 26 24 21 38 36 33 ' \
      '31 9 7 0 0 19 17 14 12 10 ' \
      '12 15 17 0 0 0 8 29 32 34 ' \
      '36 20 22 24 27 29 26 24 21 38' \
      ' 36 33 31 9 7 0 0 19 17 14' \
      ' 12 10 12 15 17 0 0 0 8 29' \
      ' 32 34 36 20 22 24 27 28 26 24' \
      ' 21 38 36 33 31 9 7 0 0 19 17 14 12 10 12 15 17 0 0 0 8 29 32 34 36 20 22 24 27 29 26 24 21 38 36 33 31 9 7 0 0 19 17 14 12 10 12 15 17 0 0 0 7 29 32 34 36 20 22 24 27 29 26 24 21 38 36 33 31 9 7 0 0 19 16 14 12 10 12 15 17 0 0 0 8 29 32 34 36 20 22 24 27 29 26 24 21 38 36 33 31 9 7 0 0 19 17 14 12 10 12 15 17 0 0 0 7 29 32 34 36 20 22 24 27 29 26 24 21 38 36 33 31 9 7 0 0 19 16 14 12 10 12 15 17 0 0 0 8 29 32 34 36 20 22 24 27 29 26 24 21 38 36 33 31 9 7 0 0 19 17 14 12 10 12 15 17 0 0 0 7 29 32 34 36 20 22 24 27 29 26 24 21 38 36 33 31 9 7 0 0 19 16 14 12 10 12 15 17 0 0 0 8 29 32 34 37 20 22 24 27 29 26 24 21 38 36 33 31 9 7 0 0 19 17 14 12 10 12 15 17 0 0 0 7 29 32 34 36 20 22 24 27 29 26 24 21 38 36 33 31 9 7 0 0 19 17 14 12 10 12 15 17 0 0 0 8 29 32 34 36 20 22 24 27 29 26 24 21 38 36 33 31 9 7 0 0 19 17 14 12 10 12 15 17 0 0 0 7 29 32 34 36 20 22 24 27 29 26 24 21 38 36 33 31 9 7 0 0 19 16 14 12 10 12 15 17 0 0 0 8 29 32 34 36 20 22 24 27 29 26 24 21 38 36 33 31 9 7 0 0 19 17 14 12 10 12 15 17 0 0 0 7 29 32 34 36 20 22 24 27 29 26 24 21 38 36 33 31 9 7 0 0 19 16 14 12 10 12 15 17 0 0 0 8 29 32 34 36 20 22 24'

beilv= np.array(cccc.split(' '),dtype=np.int)

id_number = np.zeros([21398,512],dtype=np.float)
txt_path ='E:/about_Face/faceID1'
count_face =[]
save_path='E:/about_Face/face_id/'
countt = 0

xmin = 0.0
xmax = 0.0
for txt_flie in os.listdir(txt_path):
    txt_open = open(txt_path + "/" + txt_flie)
    txt_read = txt_open.read()
    face_init = txt_read.split(' ')
    txt_open.close()

    output_txt = ''
    outss = np.array(face_init[154:154+512],dtype=float)
    outxx = np.zeros([512], dtype=np.float)
    #print(out512[:5])
    countcc=0
    for ii in outss:
        # print('第几次循环',countcc)
        # print(float(ii) * (10 ** beilv[countcc]))
        # print(math.pow( 10, -beilv[countcc]))

        if ii==None:
            print("None")
            outxx[countcc]=0.0


        if  -1<ii  and ii<1:
            outzhuan = outss[countcc] * math.pow( 10, beilv[countcc%32])
        else:
            outzhuan = outss[countcc] * math.pow( 10, -beilv[countcc%32])

        if outzhuan < -500.0:
            break


        if outzhuan  > 500.0:
            break

        outxx[countcc]=(outzhuan+ 499.67599999999993)/(499.67599999999993+499.803)
        output_txt +=str((outzhuan+ 499.67599999999993)/(499.67599999999993+499.803))+ ' '
        countcc+=1
    if countcc<512:
        continue
    # print(out512[:5])
    # print(output_txt)
    # write_txt = open(save_path + txt_flie[:-6] + '.txt', 'w')
    # write_txt.write(output_txt)
    # write_txt.close()

    # if np.nanmin(outxx)<-500.0:
    #     list_a = outxx.tolist()
    #     print(txt_flie, '最小值',np.nanmin(outxx))
    #     continue
    #
    # if np.nanmax(outxx)>500.0:
    #     list_a = outxx.tolist()
    #     print(txt_flie, '最大值',np.nanmax(outxx))
    #     continue

    write_txt = open(save_path + txt_flie[:-6] + '.txt', 'w')
    write_txt.write(output_txt)
    write_txt.close()

    id_number[countt] = outxx
    count_face.append(outxx)
    countt+=1

# print(xmax,xmin)
xxxxxx = np.array(count_face,np.float)
print(np.min(xxxxxx),np.max(xxxxxx),np.nanmax(xxxxxx),np.nanmin(xxxxxx))
print(np.nanmax(id_number),np.nanmin(id_number))






