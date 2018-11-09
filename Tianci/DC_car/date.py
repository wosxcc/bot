import json
import urllib.request
import datetime
import  http.client as httplib


def re_feast(date):
    server_url = "http://www.easybots.cn/api/holiday.php?d="
    httpClient = urllib.request.urlopen(server_url+date)
    vop_data = json.loads(httpClient.read())
    return int(float(vop_data[date]))


def get_week_day(date):
    datee=datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    m = int(date[5:7])
    d = int(date[8:10])
    w = int(datee.weekday())
    h = int(date[11:13])
    ms = int(date[14:16])
    s = int(date[17:])
    #print(m,d,h,ms,s)
    return m,d,w,h,ms,s


def time_jian (dateA,dateB):
    timec = (datetime.datetime.strptime(dateB, "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(dateA,"%Y-%m-%d %H:%M:%S")).seconds
    house = float(timec)/3600.0
    return house