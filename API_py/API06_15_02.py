import  http.client as httplib
import hashlib
import urllib
import random
import json
appid = '20180615000176703'
secretKey = 'IU2YJT3o6y_AfqLWXhmw'


httpClient = None
myurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
q = 'My God, have you eaten'
fromLang = 'auto'
toLang = 'zh'
salt = random.randint(32768, 65536)
# salt = 'My God, have you eaten'

sign = appid + q + str(salt) + secretKey
print(sign)
m = hashlib.md5()
m.update(sign.encode("utf-8"))
sign =m.hexdigest()

#
# m1 = hashlib.md5()
# # m1 = md5.new()
# m1.update(sign)
# sign = m1.hexdigest()

myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(salt) + '&sign=' + sign

try:
    httpClient = httplib.HTTPConnection('api.fanyi.baidu.com')
    httpClient.request('GET', myurl)

    # response是HTTPResponse对象
    response = httpClient.getresponse()
    sssread=response.read()
    xxx =sssread.decode()
    print(xxx.encode("utf-8").decode('unicode_escape'))
    zwen = xxx.split('dst')[1][3:-4]
    print(zwen.encode("utf-8").decode('unicode_escape'))
except Exception:
    print('报错了')
finally:
    if httpClient:
        httpClient.close()
