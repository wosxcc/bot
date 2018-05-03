import cv2 as cv
import numpy as np
import math

cap =cv.VideoCapture('E:/dectect/dectect/cut_image/1.mp4')

def get_rect(im, title='get_rect'):   #   (a,b) = get_rect(im, title='get_rect')
    mouse_params = {'tl': None, 'br': None, 'current_pos': None,
        'released_once': False}

    cv.namedWindow(title)
    cv.moveWindow(title, 100, 100)

    def onMouse(event, x, y, flags, param):

        param['current_pos'] = (x, y)

        if param['tl'] is not None and not (flags & cv.EVENT_FLAG_LBUTTON):
            param['released_once'] = True

        if flags & cv.EVENT_FLAG_LBUTTON:
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']

    cv.setMouseCallback(title, onMouse, mouse_params)
    cv.imshow(title, im)

    while mouse_params['br'] is None:
        im_draw = np.copy(im)

        if mouse_params['tl'] is not None:
            cv.rectangle(im_draw, mouse_params['tl'],
                mouse_params['current_pos'], (255, 0, 0))

        cv.imshow(title, im_draw)
        _ = cv.waitKey(10)

    cv.destroyWindow(title)

    tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
        min(mouse_params['tl'][1], mouse_params['br'][1]))
    br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
        max(mouse_params['tl'][1], mouse_params['br'][1]))

    return (tl, br)  #tl=(y1,x1), br=(y2,x2)

def draw_bbox(event,x,y,flags,param):
    global  sx,sy,ex,ey

    if event==cv.EVENT_LBUTTONDOWN:
        sx, xy=x,y
        ex, ey=x,y
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
        print('sx,sy,x,y',sx,sy,x,y)
        cv.imshow('hahah',img)
        cv.line(img,(sx,sy),(x,y),(0,0,255),2)
        cv.rectangle(img,(sx,sy),(x,y),(0,255,255),2)
    # if event==cv.EVENT_LBUTTONUP:

while True:
    res ,img =cap.read()
    # cv.imshow('img', img)
    # cv.waitKey()

    while True:
        os = cv.waitKey(1) & 0xff
        (a, b) = get_rect(img, title='get_rect')
        # cv.namedWindow(u'获取位置')
        # cv.setMouseCallback(u'获取位置', draw_bbox)
        cv.imshow(u'获取位置',img)
        if os == 27:
            break




