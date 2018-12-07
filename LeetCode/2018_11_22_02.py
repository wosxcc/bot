
def isMatch(s, p):
    if len(s) == 0:
        return True
    i = p.find(s[0])
    if i<0:
        i=p.find('.')
        if i<0:
            return False
    #print('我看看啦',i,len(p),i>=len(p))
    while(i<len(p)):
        #print('有进入循环么')
        for c in range(len(s)):
            if (i==len(p)):
                return False

            if p[i]=='.':
                i+=1
                continue
            if s[c]==p[i] and p[i]!='*':
                i+=1
                continue

            if p[i] == '*' and p[i - 1]=='.':
                if c ==len(s)-1:
                    return True
                if s[c-1] != s[c]:
                    j = p[i + 1:].find(s[0])
                    # print(i)
                    if j < 0:
                        j = p.find('.')
                        if j < 0:
                            return False
                    i = i + 1 + j
                    c=0
                    break
                if s[c+1] != s[c]:
                    i += 1
                    continue
            if p[i]=='*' and s[c]==p[i-1]:

                if c ==len(s)-1:
                    return True
                if s[c+1] != p[i - 1]:
                    i+=1
                    continue
            if s[c]!=p[i] and p[i]!='*':
                j=p[i+1:].find(s[0])
                if j < 0:
                    j = p.find('.')
                    if j < 0:
                        return False
                i=i+1+j
                c = 0
                break
        if c ==len(s)-1:
            return True
    return False


s="mississippi"
p=".*"
print(isMatch(s,p))



