def fib_num(length):
    if(length < 0):
        return []
    fib_num = [0,1]
    for i in range(2,length+1):
        fib_num.append(fib_num[i-2] + fib_num[i-1])
    return fib_num[:length+1]

def fib_num_rec(list_num, length):
    #print(list_num)
    if(length == len(list_num)):
        print(list_num)
        return list_num
    len_list = len(list_num)
    list_num.append(list_num[len_list - 1] + list_num[len_list - 2])
    #print(list_num)
    fib_num_rec(list_num, length)

#print(fib_num(0))
#print(fib_num(1))
#print(fib_num(12))
l = fib_num_rec([0,1], 4)
print(l)