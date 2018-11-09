def divPrime(num):
    lt = []
    print(num, '=', end=' ')
    while num != 1:
        for i in range(2, int(num + 1)):
            if num % i == 0:  # i是num的一个质因数
                lt.append(i)
                num = num / i  # 将num除以i，剩下的部分继续分解
                break
    for i in range(0, len(lt) - 1):
        print(lt[i], '*', end=' ')

    print(lt[-1])
divPrime(99721)