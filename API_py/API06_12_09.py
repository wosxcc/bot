import base64
import urllib,urllib3
import json
client_id = 'q2bA1wTuhPtoGRoYp0ROXUwQ'
client_secret = '0T37uW8DXMCftWSPboo5ZjGd3QgYYpqo'
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s' % (
client_id, client_secret)
f = open('../face_into/face68/image_test/sface_3140.jpg', 'rb')
img = base64.b64encode(f.read())

params=json.dumps([{'image':img,'image_type':'BASE64','face_type':'LIVE','quality_control':'LOW'}])
params = urllib.urlencode(params)

request_url='https://aip.baidubce.com/rest/2.0/face/v3/detect'

request_url = request_url + "?access_token=" + access_token
request = urllib2.Request(url=request_url, data=params)
request.add_header('Content-Type', 'application/json')
response = urllib2.urlopen(request)
content = response.read()
if content:
    print content