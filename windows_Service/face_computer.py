# -*- coding: UTF-8 -*-
import socket
def connection_create(HOST,PORT):
    asocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)  #定义socket类型，网络通信，TCP
    try:
        asocket.connect((HOST,PORT)) ## 要连接的IP与端口
    except socket.error as e:
        return 0
    return asocket

def send_request(asocket,cmd):
    print('Video is：', cmd)
    while True:
        try:
            aaa = asocket
            #
            # print(asocket)
            asocket.sendall(str(cmd).encode("utf-8"))
            # print(asocket)
            data = aaa.recv(200000)
        except socket.error as e:
            print('lian jie zhong duan')
            return 0

        sdata = str(data, encoding="utf8")
        print(sdata[:5000])


if __name__=="__main__":
    HOST = input("Your HOST:")
    PORT = int(input("Your PORT:"))
    for i in range(16):
        astock = connection_create(HOST, PORT)
        if astock != 0:
            break
        else:
            if (i % 3 == 0):
                print('Unable to connect to server')
    while True:
        cmd = input("Video path:")

        if cmd == 'exit':
            astock.close()
            break
        if astock!=0:
            print('connect to server')
            # print(astock)
            send_request(astock,cmd)

        else:
            print('server id connect to ')


#  D:/bot_opencv/dectect/dectect/image/4.mp4

