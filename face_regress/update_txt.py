import os
import io



file_yuan=open('train.txt')
get=file_yuan.read()
result=get.split('\n')
other_result=get.splitlines()
try:
    with open('train_no.txt', 'w') as file_write:
        for i in  range(len(other_result)):
            data_list=other_result[i].split(' ')
            # print(data_list)
            # write_str = data_list[0] + '\t1\t' + data_list[1] + '\t' + data_list[2] + '\t' + data_list[3] + '\t' + str(
            #     1.00) + '\n'
            # if float(data_list[3])>1.0:
            #     write_str = data_list[0] + '\t1\t'  + data_list[1] + '\t'+data_list[2]+ '\t'+data_list[3]+ '\t'+str(1.00)+'\n'
            write_str=data_list[0] + '\t' + data_list[1] + '\t-1\t-1\t-1\t-1\n'
            file_write.write(write_str)
            print(write_str)
except:
    print('Error')