import socket   #socket模块
import subprocess

def start_service(HOST,PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 定义socket类型，网络通信，TCP
    try:
        s.bind((HOST, PORT))  # 套接字绑定的IP与端口
    except socket.error as e:
        print('IP或端口号号错误，服务开启失败。')
        return 0
    s.listen(10)  # 开始TCP监听,监听10个请求
    return s


def service_event(asocket):
    while 1:
        conn, addr = asocket.accept()  # 接受TCP连接，并返回新的套接字与IP地址
        print("已连接", addr)
        while 1:
            try:
                data = conn.recv(200000)  # 把接收的数据实例化
            except socket.error as e:
                print('客户', addr, '连接已断开')
                break
            sdata = str(data, encoding="utf8")
            # print("输入命令",sdata)
            cmd_status, cmd_result = subprocess.getstatusoutput(sdata)
            if len(cmd_result.strip()) == 0:  # 如果输出结果长度为0，则告诉客户端完成。此用法针对于创建文件或目录，创建成功不会有输出信息
                try:
                    conn.sendall('Done.'.encode("utf-8"))
                except socket.error as e:
                    print('连接', addr, '已断开')
                    break
            else:
                output_txt = cmd_result.encode("utf-8")
                try:
                    conn.sendall(output_txt)  # 否则就把结果发给对端（即客户端）
                except socket.error as e:
                    print('客户', addr, '连接已断开')
                    break
                    # print(output_txt)
                    # print(len(output_txt))
    conn.close()
if __name__ == "__main__":
    HOST = ''
    PORT = 8000
    asocket =start_service(HOST,PORT)

    service_event(asocket)


    # 关闭连接