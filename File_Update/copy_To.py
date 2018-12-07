import os ,shutil


yfile_patch="E:/about_Face/facenet-master/data/casia_maxpy_mtcnnpy_182"
mfile_patch ="D:/faceID/"

for bfile in os.listdir(yfile_patch):
    for imgfile in os.listdir(yfile_patch+'/'+bfile):
        shutil.copyfile(yfile_patch+'/'+bfile+"/"+imgfile, mfile_patch+imgfile)