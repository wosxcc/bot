# -*- coding: utf-8 -*-
                    ####截图翻译####
import  http.client as httplib
import hashlib
import urllib
import random
import cv2 as cv
import numpy as np
from PIL import ImageGrab, Image, ImageDraw, ImageFont
from win32api import GetSystemMetrics
import  os

if __name__ == "__main__":
    drawing=False
    mode=True
    ix,iy=-1,-1             # 初始化位置
    rex,rey=-1,-1           # 初始化位置
    class0='hand'
    class1='face'
    windows_name='E:/xbot/pachong/car'   # 文件位置
    img_copy=[]
    sum_init=[]
    # global count_e



    appid = '20180615000176703'
    secretKey = 'IU2YJT3o6y_AfqLWXhmw'
    httpClient = None
    myurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    fromLang = 'auto'
    toLang = 'zh'

    from aip import AipOcr
    import base64
    import  cv2 as cv
    """ 你的 APPID AK SK """
    APP_ID = '11402288'
    API_KEY = 'iQMH1TCFcZLpglS0tfxrLr9O'
    SECRET_KEY = 'VcUPEubVGoGIOXYfA9oZYC71DCuu7RGx'
    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
    def get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    def fanyi_img(img_path):
        salt = random.randint(32768, 65536)
        myurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
        imgss = get_file_content(img_path)
        img_txt = client.general(imgss)
        img = cv.imread(img_path)

        count = 1
        en_txt = ''
        for txt_line in img_txt['words_result']:
            # print(count, txt_line['words'])
            q = txt_line['words']
            sign = appid + q + str(salt) + secretKey
            m = hashlib.md5()
            m.update(sign.encode("utf-8"))
            sign = m.hexdigest()
            myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign
            httpClient = httplib.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', myurl)
            response = httpClient.getresponse()
            sssread = response.read()
            xxx = sssread.decode()
            zwen = xxx.split('dst')[1][3:-4]
            print(count, zwen.encode("utf-8").decode('unicode_escape'))
            en_txt += txt_line['words']
            img_PIL = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            ##打印文字
            ssscc = zwen.encode("utf-8").decode('unicode_escape')           # 图片上显示中文
            font = ImageFont.truetype("simhei.ttf", int(txt_line['location']['height']), encoding="utf-8")  ##第二个为字体大小
            # 字体颜色
            fillColor = (255, 0, 0)
            # 文字输出位置
            position = (int(txt_line['location']['left']), int(txt_line['location']['top'])+int(txt_line['location']['height']))
            draw = ImageDraw.Draw(img_PIL)
            draw.text(position, ssscc, font=font, fill=fillColor)

            img=cv.cvtColor(np.asarray(img_PIL), cv.COLOR_RGB2BGR)
            # img = cv.putText(img, str(count), (int(txt_line['location']['left']), int(txt_line['location']['top'] + 20)),
            #                  cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            count += 1
        print(en_txt)
        q = en_txt
        sign = appid + q + str(salt) + secretKey
        m = hashlib.md5()
        m.update(sign.encode("utf-8"))
        sign = m.hexdigest()
        myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
            q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign
        httpClient = httplib.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)

    # response是HTTPResponse对象
    response = httpClient.getresponse()
    sssread = response.read()
    xxx = sssread.decode()
    zwen = xxx.split('dst')[1][3:-4]
    print(zwen.encode("utf-8").decode('unicode_escape'))

    if os.path.exists(img_path):
        # 删除文件，可使用以下两种方法。
        os.remove(img_path)
    cv.imshow('img', img)
    cv.waitKey()




    def draw_circle(event,x,y,flags,param):
        global ix,iy,rex,rey,drawing,mode
        if event==cv.EVENT_LBUTTONDOWN:
            drawing =True
            rex, rey=x,y
            ix,iy=x,y
        elif event==cv.EVENT_MOUSEMOVE and  flags==cv.EVENT_FLAG_LBUTTON:
            if drawing ==True:
                if mode==True:
                    im_draw = np.copy(img_copy)
                    # cv.putText(im_draw, str(count_e), (rex, rey+20),  cv.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2)
                    # cv.line(im_draw, (rex, rey), (x, y), (0, 255, 255), 2)
                    cv.rectangle(im_draw, (rex, rey), (x, y), (255, 0, 0), 2)
                    ix, iy = x, y
                    cv.imshow(windows_name, im_draw)
                else:
                    im_draw = np.copy(img_copy)
                    # cv.putText(im_draw,class1,(rex, rey+20),cv.FONT_HERSHEY_SIMPLEX,1,(255, 0, 255),2)
                    # cv.line(im_draw, (rex, rey), (x, y), (0, 255, 255), 2)
                    cv.rectangle(im_draw, (rex, rey), (x, y), (255, 0, 255), 2)
                    ix, iy = x, y
                    cv.imshow(windows_name, im_draw)

        if event==cv.EVENT_LBUTTONUP:
            # img_copy = np.copy(img)
            x = max(x, 0)
            y = max(y, 0)
            x = min(x, img_copy.shape[1])
            y = min(y, img_copy.shape[0])
            mx=(rex + x) / 2.0
            my=(rey + y) / 2.0
            min_x = rex if (x > rex) else x
            max_x = rex if (x < rex) else x
            min_y = rey if (y > rey) else y
            max_y = rey if (y < rey) else y
            xwidth = abs(rex - x)
            yheight = abs(rey - y)
            if xwidth*yheight>=200:
                # print('结束时的位置：rex,rey,x,y',mx,my,xwidth,yheight)
                if mode == True:                                # 如果是人类型是0
                    sum_init.append([0,mx,my,xwidth,yheight])
                    # cv.putText(img_copy, class0, (int(mx-xwidth/2), int(my-yheight/2)+20), cv.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0),2)
                    cv.rectangle(img_copy, (int(mx-xwidth/2), int(my-yheight/2)), (int(mx+xwidth/2), int(my+yheight/2)), (255, 0, 0), 1)
                else:                                            # 如果是车类型是1
                    sum_init.append([1,mx,my,xwidth,yheight])
                    # cv.putText(img_copy, class1, (int(mx - xwidth / 2), int(my - yheight / 2)+20), cv.FONT_HERSHEY_SIMPLEX, 1,
                    #            (255, 0, 255), 2)
                    cv.rectangle(img_copy, (int(mx - xwidth / 2), int(my - yheight / 2)),
                                 (int(mx + xwidth / 2), int(my + yheight / 2)), (255, 0, 255), )
                drawing=False

    # keep_num=

bbox = (0, 0,  GetSystemMetrics (0), GetSystemMetrics (1))   #####(sx,sy,ex,ey)
im = ImageGrab.grab(bbox)   ###获取屏幕图像
# im.save('as.jpg')
img = np.array(im)       ###转化为array数组
img =cv.cvtColor(img,cv.COLOR_BGR2RGB)
# img=cv.resize(img,(1400,1000), interpolation=cv.INTER_CUBIC)
img_copy = np.copy(img)
# if count_c%3==0:
cv.namedWindow(windows_name)
cv.setMouseCallback(windows_name,draw_circle)
sum_init = []
mode = True
while(1):
    cv.imshow(windows_name, img_copy)
    k=cv.waitKey(0)&0xFF
    if k==ord('c'): # c转换类型
        print('转换成功')
        mode=not mode
        cv.imshow(windows_name, img_copy)
    elif k==ord('s'):       # s保存
        if len(sum_init)>0:
            for bbox in sum_init:
                bboxss = [int(xx) for xx in bbox]
                imgwrite_name ='ls.jpg'
                # cv.imshow('xximg',img[int(bboxss[2]-bboxss[4]/2):int(bboxss[2]+bboxss[4]/2),int(bboxss[1]-bboxss[3]/2):int(bboxss[1]+bboxss[3]/2),:])
                cv.imwrite(imgwrite_name,img[int(bboxss[2]-bboxss[4]/2):int(bboxss[2]+bboxss[4]/2),int(bboxss[1]-bboxss[3]/2):int(bboxss[1]+bboxss[3]/2),:])
                fanyi_img(imgwrite_name)
            break
    elif k == ord('d'):             # 清屏
        img_copy = np.copy(img)
        sum_init=[]
    elif k==32:  # 空格跳到下一帧
        break

cv.destroyAllWindows()
