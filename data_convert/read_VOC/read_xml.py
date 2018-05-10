import  xml.dom.minidom
import  cv2 as cv


dom = xml.dom.minidom.parse('E:\BaiduNetdiskDownload\VOC 2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\Annotations/2007_000027.xml')



#得到文档元素对象
root = dom.documentElement
names = root.getElementsByTagName('name')
for name in names:
    print(name.firstChild.data)





#
# xmins = root.getElementsByTagName('xmin')
# for xmin in xmins:
#     xmin.firstChild.data
# ymins= root.getElementsByTagName('ymin')
# for ymin in ymins:
#     ymin.firstChild.data
# xmaxs= root.getElementsByTagName('xmax')
# for xmax in xmaxs:
#     xmax.firstChild.data
# ymaxs= root.getElementsByTagName('ymax')
# for ymax in ymaxs:
#     ymax.firstChild.data



xmins = root.getElementsByTagName('xmin')
ymins= root.getElementsByTagName('ymin')
xmaxs= root.getElementsByTagName('xmax')
ymaxs= root.getElementsByTagName('ymax')
count_box=[]

for i in range(len(ymaxs)):
    bbox = {}
    print('int(xmins[i].firstChild.data)',int(xmins[i].firstChild.data))
    bbox['x1'] = int(xmins[i].firstChild.data)
    bbox['y1'] = int(ymins[i].firstChild.data)
    bbox['x2'] = int(xmaxs[i].firstChild.data)
    bbox['y2'] = int(ymaxs[i].firstChild.data)
    print(bbox)
    count_box.append(bbox)

print(count_box)

img=cv.imread('E:\BaiduNetdiskDownload\VOC 2012\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages/2007_000027.jpg')
cv.imshow('imgxx',img)
for j in range(len(count_box)):
    print(count_box[j]['x1'],count_box[j]['y1'],count_box[j]['x2'],count_box[j]['y2'])
    img=cv.rectangle(img,(count_box[j]['x1'],count_box[j]['y1']),(count_box[j]['x2'],count_box[j]['y2']),(0,0,255),2)
cv.imshow('imgxxxx',img)

cv.waitKey()
cv.destroyAllWindows()

