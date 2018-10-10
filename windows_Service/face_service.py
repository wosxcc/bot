import socket   #socket模块
import subprocess
import threading,getopt,sys,string
from win32api import GetSystemMetrics


def start_service(HOST,PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 定义socket类型，网络通信，TCP
    try:
        s.bind((HOST, PORT))  # 套接字绑定的IP与端口
    except socket.error as e:
        print('IP或端口号号错误，服务开启失败。')
        return 0
    s.listen(10)  # 开始TCP监听,监听10个请求
    return s


def jonnys(client,addres):

    client.settimeout(200)
    print("已连接", addres)
    while 1:
        try:
            data = client.recv(200000)  # 把接收的数据实例化
            sdata = str(data, encoding="utf8")
            print(sdata)
            output_list =sdata.split(' ')
            output_txt = output_list[-1]+' '+output_list[0]
            print(output_txt)
            if len(output_txt.strip()) == 0:  # 如果输出结果长度为0，则告诉客户端完成。此用法针对于创建文件或目录，创建成功不会有输出信息
                try:
                    client.sendall('Done.'.encode("utf-8"))
                except socket.error as e:
                    print('连接', addres, '已断开')
            else:
                try:
                    client.sendall(output_txt.encode("utf-8"))  # 否则就把结果发给对端（即客户端）
                except socket.error as e:
                    print('客户', addres, '连接已断开')
        except socket.timeout:
            print("超时")
    client.close()

#     return conn
if __name__ == "__main__":
    HOST = ''
    PORT = 8000
    aservice =start_service(HOST,PORT)
    threads=[]
    while True:
        conn, addr = aservice.accept()
        thread = threading.Thread(target=jonnys,args=(conn,addr))
        # threads.append(thread1)
        # thread2 = threading.Thread(target=jonnys, args=(conn, addr))
        # threads.append(thread2)
        thread.start()


    # 关闭连接