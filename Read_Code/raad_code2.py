import cv2 as cv
import os
import numpy as np



def _get_dynamic_binary_image(filedir, img_name):
    # filename =   './out_img/' + img_name.split('.')[0] + '-binary.jpg'
    img_name = filedir + '/' + img_name
    print('.....' + img_name)
    im = cv.imread(img_name)
    im = cv.cvtColor(im,cv.COLOR_BGR2GRAY) #灰值化
    # 二值化
    th1 = cv.adaptiveThreshold(im, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 1)
    return th1

def clear_border(img):
  h, w = img.shape[:2]
  for y in range(0, w):
    for x in range(0, h):
      if y < 2 or y > w - 2:
        img[x, y] = 255
      if x < 2 or x > h -2:
        img[x, y] = 255
  return img



def interference_line(img):
   h, w = img.shape[:2]
   # ！！！opencv矩阵点是反的
   # img[1,2] 1:图片的高度，2：图片的宽度
   for y in range(1, w - 1):
     for x in range(1, h - 1):
       count = 0
       if img[x, y - 1] > 245:
         count = count + 1
       if img[x, y + 1] > 245:
         count = count + 1
       if img[x - 1, y] > 245:
         count = count + 1
       if img[x + 1, y] > 245:
         count = count + 1
       if count > 2:
         img[x, y] = 255
   return img

def interference_point(img, x = 0, y = 0):
    """
    9邻域框,以当前点为中心的田字框,黑点个数
    :param x:
    :param y:
    :return:
    """
    cur_pixel = img[x,y]# 当前像素点的值
    height,width = img.shape[:2]

    for y in range(0, width - 1):
      for x in range(0, height - 1):
        if y == 0:  # 第一行
            if x == 0:  # 左上顶点,4邻域
                # 中心点旁边3个点
                sum = int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])
                if sum <= 2 * 245:
                  img[x, y] = 0
            elif x == height - 1:  # 右上顶点
                sum = int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1])
                if sum <= 2 * 245:
                  img[x, y] = 0
            else:  # 最上非顶点,6邻域
                sum = int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])
                if sum <= 3 * 245:
                  img[x, y] = 0
        elif y == width - 1:  # 最下面一行
            if x == 0:  # 左下顶点
                # 中心点旁边3个点
                sum = int(cur_pixel) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y - 1]) \
                      + int(img[x, y - 1])
                if sum <= 2 * 245:
                  img[x, y] = 0
            elif x == height - 1:  # 右下顶点
                sum = int(cur_pixel) \
                      + int(img[x, y - 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y - 1])

                if sum <= 2 * 245:
                  img[x, y] = 0
            else:  # 最下非顶点,6邻域
                sum = int(cur_pixel) \
                      + int(img[x - 1, y]) \
                      + int(img[x + 1, y]) \
                      + int(img[x, y - 1]) \
                      + int(img[x - 1, y - 1]) \
                      + int(img[x + 1, y - 1])
                if sum <= 3 * 245:
                  img[x, y] = 0
        else:  # y不在边界
            if x == 0:  # 左边非顶点
                sum = int(img[x, y - 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y - 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])

                if sum <= 3 * 245:
                  img[x, y] = 0
            elif x == height - 1:  # 右边非顶点
                sum = int(img[x, y - 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x - 1, y - 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1])

                if sum <= 3 * 245:
                  img[x, y] = 0
            else:  # 具备9领域条件的
                sum = int(img[x - 1, y - 1]) \
                      + int(img[x - 1, y]) \
                      + int(img[x - 1, y + 1]) \
                      + int(img[x, y - 1]) \
                      + int(cur_pixel) \
                      + int(img[x, y + 1]) \
                      + int(img[x + 1, y - 1]) \
                      + int(img[x + 1, y]) \
                      + int(img[x + 1, y + 1])
                if sum <= 4 * 245:
                  img[x, y] = 0
    return img

# def cfs(im,x_fd,y_fd):
#   '''用队列和集合记录遍历过的像素坐标代替单纯递归以解决cfs访问过深问题
#   '''
#
#   # print('**********')
#
#   xaxis=[]
#   yaxis=[]
#   visited =set()
#   q = Queue()
#   q.put((x_fd, y_fd))
#   visited.add((x_fd, y_fd))
#   offsets=[(1, 0), (0, 1), (-1, 0), (0, -1)]#四邻域
#
#   while not q.empty():
#       x,y=q.get()
#
#       for xoffset,yoffset in offsets:
#           x_neighbor,y_neighbor = x+xoffset,y+yoffset
#
#           if (x_neighbor,y_neighbor) in (visited):
#               continue  # 已经访问过了
#
#           visited.add((x_neighbor, y_neighbor))
#
#           try:
#               if im[x_neighbor, y_neighbor] == 0:
#                   xaxis.append(x_neighbor)
#                   yaxis.append(y_neighbor)
#                   q.put((x_neighbor,y_neighbor))
#
#           except IndexError:
#               pass
#   # print(xaxis)
#   if (len(xaxis) == 0 | len(yaxis) == 0):
#     xmax = x_fd + 1
#     xmin = x_fd
#     ymax = y_fd + 1
#     ymin = y_fd
#
#   else:
#     xmax = max(xaxis)
#     xmin = min(xaxis)
#     ymax = max(yaxis)
#     ymin = min(yaxis)
#     #ymin,ymax=sort(yaxis)
#
#   return ymax,ymin,xmax,xmin
#
# def detectFgPix(im,xmax):
#   '''搜索区块起点
#   '''
#
#   h,w = im.shape[:2]
#   for y_fd in range(xmax+1,w):
#       for x_fd in range(h):
#           if im[x_fd,y_fd] == 0:
#               return x_fd,y_fd
#
# def CFS(im):
#   '''切割字符位置
#   '''
#
#   zoneL=[]#各区块长度L列表
#   zoneWB=[]#各区块的X轴[起始，终点]列表
#   zoneHB=[]#各区块的Y轴[起始，终点]列表
#
#   xmax=0#上一区块结束黑点横坐标,这里是初始化
#   for i in range(10):
#
#       try:
#           x_fd,y_fd = detectFgPix(im,xmax)
#           # print(y_fd,x_fd)
#           xmax,xmin,ymax,ymin=cfs(im,x_fd,y_fd)
#           L = xmax - xmin
#           H = ymax - ymin
#           zoneL.append(L)
#           zoneWB.append([xmin,xmax])
#           zoneHB.append([ymin,ymax])
#
#       except TypeError:
#           return zoneL,zoneWB,zoneHB
#   return zoneL,zoneWB,zoneHB

path= 'H:/Chrome_drown/caffe_verify_code/train_notsplit'
for file in os.listdir(path):
    img=_get_dynamic_binary_image(path, file)
    img=clear_border(img)
    img2=interference_line(img)
    img3=interference_point(img2)
    cv.imshow('img',img3)
    cv.waitKey()
    cv.destroyAllWindows()