#coding=utf-8
import socket
import re
HOST = ''
PORT = 8000#Readindex.html,putintoHTTPresponsedata
index_content = '''HTTP/1.x200ok
Content-Type: text/html'''
file=open('axcc.html','rb')

file_xx =file.read().decode("utf-8")
index_content+=file_xx
file.close()

# #Readreg.html,putintoHTTPresponsedata
# reg_content = '''HTTP/1.x200ok
# Content-Type:text/html'''
# file=open('reg.html','r')
# reg_content+=file.read()
# file.close()#Readpicture,putintoHTTPresponsedatafile=open('T-mac.jpg','rb')
# pic_content='''HTTP/1.x200ok
# Content-Type:image/jpg'''
# pic_content+=file.read()
# file.close()#Configuresocket


sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock.bind((HOST,PORT))
sock.listen(100)#infiniteloop
while True:
    #maximumnumberofrequestswaiting
    conn, addr = sock.accept()


    request=conn.recv(1024)
    request =str(request, encoding="utf8")

    print("你有什么请求",request)
    method=request.split(' ')[0]
    src=request.split(' ')[1]

    print('有什么需求',src)
    #dealwihtGETmethod
    if method=='GET':
        if src=='/axcc.html':
            content=index_content
        elif re.match('^/\?.*$',src):
            entry=src.split('?')[1]#maincontentoftherequest
            content='HTTP/1.x200ok\r\nContent-Type:text/html\r\n\r\n'
            content+=entry
            content+='<br/><fontcolor="green"size="7">registersuccesss!</p>'
        else:
            continue

    #dealwithPOSTmethod
    elif method=='POST':
        form=request.split('\r\n')
        entry=form[-1]#maincontentoftherequest
        content='HTTP/1.x200ok\r\nContent-Type:text/html\r\n\r\n'
        content+=entry
        content+='<br/><fontcolor="green"size="7">registersuccesss!</p>'

#######Moreoperations,suchasputtheformintodatabase#...######
    else:
        continue
    conn.sendall(content)

#closeconnection
    conn.close()