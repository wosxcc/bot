
import numpy as np

facea = 'E:/about_Face/face_number/Strom_Thurmond_0003.txt'
faceb = 'E:/about_Face/face_number/Strom_Thurmond_0002.txt'

sfacea =open(facea).read().split(' ')
sfaceb=open(faceb).read().split(' ')
print(sfacea)
print(sfaceb)
facea_list = np.array(sfacea[:512],dtype=np.float)
where_are_nan = np.isnan(facea_list)
facea_list[where_are_nan] = 0.5
faceb_list =  np.array(sfaceb[:512],dtype=np.float)
where_are_nanb = np.isnan(faceb_list)
faceb_list[where_are_nanb] =  0.5
dist = np.sqrt(np.sum(np.square(facea_list - faceb_list)))





print(dist)