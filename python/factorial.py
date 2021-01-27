def factorial_rec(num):
    if(num != 0):
        num = num * factorial_rec(num-1)
    else:
        return 1
    return num  

def factorial_it(num):
    res = 1
    for i in range(num,1,-1):
        res = res * i
    return res 
print(factorial_rec(4))
print(factorial_it(4))