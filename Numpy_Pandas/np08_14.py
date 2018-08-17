
for g in range(3+1,21):
    print(g)

count =0
for g in range(1,21):
    for h in range(g+2,21):
        for i in range(h+2,21):
            for j in range(i+2,21):
                count += 1
                # if j-i>0 and i-h>0 and h-g>0:
                #     count+=1
                #     print(g,h,i,j)
print('统计',count)
