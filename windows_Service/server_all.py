import socket
def connection_create(HOST,PORT):
    asocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)  #定义socket类型，网络通信，TCP
    try:
        asocket.connect((HOST,PORT)) ## 要连接的IP与端口
    except socket.error as e:
        return 0
    return asocket

def send_request(asocket):

    while True:
        cmd = input("输入cmd命令:")
        if cmd == 'exit':
            break
        print('命令为：', cmd)
        try:
            print(asocket)
            asocket.sendall(str(cmd).encode("utf-8"))
            print(asocket)
            data = asocket.recv(200000)
            print(asocket)
        except socket.error as e:
            print('连接已中断')
            return 0

        sdata = str(data, encoding="utf8")
        print(sdata[:5000])
    asocket.close()

if __name__=="__main__":
    HOST = '127.0.0.1'
    PORT = 8000



    for i in range(16):
        astock = connection_create(HOST,PORT)
        if astock!=0:
            break
        else:
            if(i%3==0):
                print('无法连接到服务')
    if astock!=0:
        print('已连接服务')
        send_request(astock)

    else:
            print('请检查服务是否开启')