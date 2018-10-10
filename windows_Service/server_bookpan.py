import socket
HOST='192.168.0.149'
PORT=8000       #56857
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)  #定义socket类型，网络通信，TCP
s.connect((HOST,PORT)) ## 要连接的IP与端口
while   1:
    cmd=input("输入cmd命令:")
    print('命令为：',cmd)
    s.sendall(str(cmd).encode("utf-8"))
    while 1:
        data=s.recv(200000)
        sdata = str(data, encoding="utf8")
        print(sdata[:600])

s.close()


