import  http.client as httplib
import hashlib
import urllib
import random
import json
appid = '20180615000176703'
secretKey = 'IU2YJT3o6y_AfqLWXhmw'



httpClient = None
myurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'

fromLang = 'auto'
toLang = 'zh'
salt = random.randint(32768, 65536)
# salt = 'My God, have you eaten'






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


img_path='txt.jpg'
# img_path='timg.jpg'
imgss =get_file_content(img_path)
img_txt=client.general(imgss)
img =cv.imread(img_path)


count=1
en_txt=''
for txt_line in img_txt['words_result']:
    print(count,txt_line[ 'words'])

    q = txt_line[ 'words']
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
    print(count,zwen.encode("utf-8").decode('unicode_escape'))
    en_txt+= txt_line[ 'words']
    img=cv.putText(img,str(count),(int(txt_line['location']['left']),int(txt_line['location']['top']+20)),cv.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),2)
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

cv.imshow('img',img)
cv.waitKey()



