# coding:utf-8
import socket
from multiprocessing import Process

HTML_ROOT_DIR = ""


def handle_client(client_socket):
    """处理客户端请求"""
    # 获取客户端请求数据
    request_data = client_socket.recv(1024) #从socket接收数据，注意是byte类型，bufsize指定一次最多接收的数据大小，
    print("请求的数据:", request_data)



    # 构造响应数据
    response_start_line = "HTTP/1.1 200 OK\r\n"
    response_headers = "Server: My server\r\n"
    response_body ='<p id="label_keyLang" >http:	<input  value="0" id="key_eng_default" name="keyLang" /></p><button type="submit" id="start" class="button">submit</button>'


    #'<br/><fontcolor="green"size="7">registersuccesss!</p>'
    #"""<input type="hidden" value="doc" name="action" id="action_scan" /><p id="label_ocrLang">识别语言:	<input  value="2" id="ocr_china" name="ocrLang"/></p><p id="label_keyLang" >输出语言:	<input  value="0" id="key_eng_default" name="keyLang" /></p><button type="submit" id="start" class="button">submit</button>"""


    response = response_start_line + response_headers + "\r\n" + response_body
    print("response data:", response)

    # 向客户端返回响应数据
    client_socket.send(bytes(response, "utf-8"))    # 发送数据到socket，前提是已经连接到远程socket,返回值是发送数据的量，检查数据是否发送完是应用的责任

    # 关闭客户端连接
    client_socket.close()


if __name__=="__main__":

    # 创建socket服务

    # socket.AF_UNIX 只能够用于单一的Unix系统进程间通信
    # socket.AF_INET 服务器之间网络通信
    # socket.AF_INET6 IPv6
    # socket.SOCK_STREAM 流式socket , for TCP
    # socket.SOCK_DGRAM 数据报式socket, for UDP

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("", 8000)) #将socket对象绑定到一个地址，但这个地址必须是没有被占用的，否则会连接失败。这里的address一般是一个ip,port对，如（‘localhost’, 10000）
    server_socket.listen(120) #开始监听TCP传入连接。backlog指定在拒绝连接之前，操作系统可以挂起的最大连接数量。该值至少为1，大部分应用程序设为5就可以了。


    while True:
        client_socket, client_address = server_socket.accept() #接受一个连接，但前提是socket必须已经绑定了一个地址，在等待连接。返回值是一个（conn, addresss）的值对，这里的conn是一个socket对象，可以用来改送或接收数据.而address是连接另一端绑定的地址，socket.getpeername()函数也能返回该地址。
        # print("[%s, %s]用户连接上了"%client_addrest[0],client_address[1])
        # print("[%s, %s]用户连接上了" % client_address)
        handle_client_process = Process(target=handle_client, args=(client_socket,))
        print(handle_client_process)
        handle_client_process.start()
        client_socket.close()       # 关闭连接，当socket.close()执行时，与这个连接相关的底层操作也会关闭（如文件描述符），一旦关闭，再对相关的文件对象操作都会失败。


