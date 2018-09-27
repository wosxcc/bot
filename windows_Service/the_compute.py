import socket
def connection_create(HOST,PORT):
    asocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)  #定义socket类型，网络通信，TCP
    try:
        asocket.connect((HOST,PORT)) ## 要连接的IP与端口
    except socket.error as e:
        return 0
    return asocket

def send_request(asocket,cmd):
    try:
        aaa = asocket
        print('命令为：', cmd)
        print(asocket)
        asocket.sendall(str(cmd).encode("utf-8"))
        print(asocket)
        data = aaa.recv(200000)
    except socket.error as e:
        print('连接已中断')
        return 0

    sdata = str(data, encoding="utf8")
    print(sdata[:5000])


if __name__=="__main__":
    HOST = '127.0.0.1'
    PORT = 8000
    for i in range(16):
        astock = connection_create(HOST, PORT)
        if astock != 0:
            break
        else:
            if (i % 3 == 0):
                print('无法连接到服务')
    while True:
        cmd = input("输入cmd命令:")

        if cmd == 'exit':
            astock.close()
            break
        if astock!=0:
            print('已连接服务')
            print(astock)
            send_request(astock,cmd)

        else:
            print('请检查服务是否开启')


# <socket.socket fd=420, family=AddressFamily.AF_INET, type=SocketKind.SOCK_STREAM, proto=0, laddr=('127.0.0.1', 3253), raddr=('127.0.0.1', 8000)>
#
# <socket.socket fd=420, family=AddressFamily.AF_INET, type=SocketKind.SOCK_STREAM, proto=0, laddr=('0.0.0.0', 3253), raddr=('127.0.0.1', 8000)>