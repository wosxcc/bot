import sys
import ssl
import base64
from urllib import request, parse


# client_id 为官网获取的AK， client_secret 为官网获取的SK
# 获取token
def get_token():
    client_id = 'q2bA1wTuhPtoGRoYp0ROXUwQ'
    client_secret = '0T37uW8DXMCftWSPboo5ZjGd3QgYYpqo '
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=%s&client_secret=%s' % (
    client_id, client_secret)
    req = request.Request(host)
    req.add_header('Content-Type', 'application/json; charset=UTF-8')
    response = request.urlopen(req)
    # 获得请求结果
    content = response.read()
    # 结果转化为字符
    content = bytes.decode(content)
    # 转化为字典
    content = eval(content[:-1])
    return content['access_token']


# 转换图片
# 读取文件内容，转换为base64编码
# 二进制方式打开图文件
def imgdata(file1path):
    f = open('image/timg.jpg', 'rb')
    img = base64.b64encode(f.read())
    return img


# 提交进行对比获得结果
def img(file1path, file2path):
    token = get_token()
    # 人脸识别API
    # url = 'https://aip.baidubce.com/rest/2.0/face/v2/detect?access_token='+token
    # 人脸对比API
    url = 'https://aip.baidubce.com/rest/2.0/face/v3/detect=' + token
    params = imgdata(file1path)
    print(params)
    # urlencode处理需提交的数据
    data = parse.urlencode(params).encode('utf-8')
    print(data)
    req = request.Request(url, data=data)
    print(req)
    req.add_header('Content-Type', 'application/x-www-form-urlencoded')
    response = request.urlopen(req)
    print(response)
    content = response.read()
    print('content',content)
    content = bytes.decode(content)
    print('content2', content)
    content = eval(content)
    print('content3', content)
    # 获得分数
    score = content['result'][0]['score']
    if score > 80:
        return '照片相似度：' + str(score) + ',同一个人'
    else:
        return '照片相似度：' + str(score) + ',不是同一个人'


if __name__ == '__main__':
    file1path = '../face_into/face68/image_test/sface_2877.jpg'
    file2path = '../face_into/face68/image_test/sface_2877f.jpg'
    res = img(file1path, file2path)
    print(res)