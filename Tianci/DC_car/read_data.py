import csv
import os
import numpy as np
from Tianci.DC_car.date import *
flie_path = "train_new.csv"


# out= open("my_train.csv",'a',newline='')
# csv_write=csv.writer(out,dialect='excel')
#
cvs_read = csv.reader(open(flie_path))
scount = 0

class_id =0
class_name=''
for lista in cvs_read:
    if scount==0:
        scount+=1
        continue
    if scount == 1:
        class_name = lista[1]
    if class_name!=lista[1] and scount>1:
        class_id+=1
        class_name =lista[1]

    # m1, d1, w1, h1, ms1, s1 = get_week_day(lista[2])
    # m2, d2, w2, h2, ms2, s2 = get_week_day(lista[3])
    # time_c = time_jian(lista[2],lista[3])
    # csv_write.writerow([class_id, m1, d1, w1, h1, ms1, s1,m2, d2, w2, h2, ms2, s2,time_c,lista[4],lista[5],lista[6],lista[7]])
    scount += 1



print(class_id)
print(scount)
